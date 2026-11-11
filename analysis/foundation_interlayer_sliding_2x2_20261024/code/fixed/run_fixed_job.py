#!/usr/bin/env python3
"""Run and audit one fixed-path native-MACE evaluation row."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_at(path: Path, index: int) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return rows[index]


def registry_row(path: Path, model_id: str) -> dict[str, str]:
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["model_id"] == model_id:
                return row
    raise KeyError(f"model_id not found in registry: {model_id}")


def xyz_frame_counts(path: Path) -> list[int]:
    lines = path.read_text().splitlines()
    counts = []
    cursor = 0
    while cursor < len(lines):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            break
        count = int(lines[cursor].strip())
        counts.append(count)
        cursor += 2 + count
    return counts


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("ase", "mace-torch", "torch", "numpy", "pandas"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_normalized_profile(path: Path, expected_images: int, model_label: str) -> tuple[bool, str]:
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        e0 = float(rows[0]["predicted_energy_eV"])
        output = path.with_name("fixed_profile.csv")
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_index", "image_fraction", "model_name", "predicted_energy_eV", "relative_energy_eV"])
            writer.writeheader()
            for row in rows:
                index = int(row["image_index"])
                energy = float(row["predicted_energy_eV"])
                writer.writerow({
                    "image_index": index,
                    "image_fraction": index / (expected_images - 1),
                    "model_name": model_label,
                    "predicted_energy_eV": energy,
                    "relative_energy_eV": energy - e0,
                })
        return True, str(output)
    except Exception as exc:
        return False, f"profile_normalization_error:{type(exc).__name__}:{exc}"


def prediction_gate(path: Path, expected_images: int, model_label: str) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing_prediction_csv"
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_images:
            return False, f"expected_{expected_images}_rows_found_{len(rows)}"
        indices = [int(row["image_index"]) for row in rows]
        if indices != list(range(expected_images)):
            return False, f"image_indices_not_ordered:{indices}"
        names = {row.get("model_name", "") for row in rows}
        if names != {model_label}:
            return False, f"model_name_mismatch:{sorted(names)}"
        values = [float(row["predicted_energy_eV"]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            return False, "nonfinite_prediction"
    except Exception as exc:
        return False, f"prediction_parse_error:{type(exc).__name__}:{exc}"
    return True, ""


def failure_class(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("deserial", "torch.jit", "unpickl", "load model")):
        return "MODEL_LOAD_OR_DESERIALIZATION"
    if any(token in lowered for token in ("nan", "inf", "nonfinite", "non-finite")):
        return "NONFINITE"
    if "not found" in lowered or "no such file" in lowered:
        return "INPUT_OR_PATH"
    return "RUNNER_EXCEPTION"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-table", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    row = row_at(args.job_table, args.index)
    scheduler_job = os.environ.get("SLURM_JOB_ID", f"LOCAL_{args.index}")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID", str(args.index))
    out = (
        args.root
        / "runs"
        / "fixed"
        / row["system"]
        / row["model_id"]
        / f"attempt_{scheduler_job}_{array_task}"
    )
    out.mkdir(parents=True, exist_ok=True)
    script = args.root / "code/fixed/mace_neb_1-calc.py"
    original_script = args.root / "code/fixed" / ("mace_neb_1-calc.cr-in-gap.original.py" if row["system"] == "cr_in_gap" else "mace_neb_1-calc.original.py")
    freeze = args.root / "ACCEPTANCE_FREEZE_R13.json"
    model_registry = args.root / "MODEL_REGISTRY.tsv"
    registry = registry_row(model_registry, row["model_id"])
    xyz = Path(row["xyz_path"])
    neb_out = Path(row["neb_out_path"])
    model = Path(row["model_path"])
    output_csv = out / "neb_predictions.csv"
    provenance_script = args.root / "code/capture_runtime_provenance.py"
    runtime_provenance = out / "runtime_provenance.json"
    geometry_gate = "PASS"
    geometry_gate_reason = ""
    try:
        counts = xyz_frame_counts(xyz)
        if counts != [int(row["atoms"])] * int(row["n_images"]):
            geometry_gate = "FAIL"
            geometry_gate_reason = f"expected_{row['n_images']}x{row['atoms']}_found_{counts}"
    except Exception as exc:
        geometry_gate = "FAIL"
        geometry_gate_reason = f"xyz_parse_error:{type(exc).__name__}:{exc}"
    observed_model_sha = sha256(model)
    model_gate = observed_model_sha == registry["sha256"]
    if not model_gate:
        geometry_gate_reason = ";".join(filter(None, [geometry_gate_reason, f"model_sha256_mismatch:{observed_model_sha}:{registry['sha256']}" ]))
    env = os.environ.copy()
    env.update(
        {
            "MACE_MODEL_NAME": row["model_label"],
            "MACE_MODEL_PATH": str(model),
            "MACE_XYZ_PATH": str(xyz),
            "MACE_NEB_OUT_PATH": str(neb_out),
            "MACE_OUTPUT_CSV": str(output_csv),
        }
    )
    command = [sys.executable, str(script)]
    (out / "command.json").write_text(json.dumps({"command": command, "env": {
        key: env[key] for key in ("MACE_MODEL_NAME", "MACE_MODEL_PATH", "MACE_XYZ_PATH", "MACE_NEB_OUT_PATH", "MACE_OUTPUT_CSV")
    }}, indent=2) + "\n")
    capture_command = [
        sys.executable, str(provenance_script), "--output", str(runtime_provenance),
        "--binary", sys.executable,
        "--input", f"model={model}",
        "--input", f"xyz={xyz}",
        "--input", f"neb_out={neb_out}",
        "--input", f"model_registry={model_registry}",
        "--input", f"job_table={args.job_table}",
        "--input", f"freeze={freeze}",
        "--input", f"runner={script}",
        "--input", f"wrapper={Path(__file__).resolve()}",
    ]
    capture_result = subprocess.run(capture_command, cwd=out, text=True, capture_output=True, check=False)
    runtime_gate = capture_result.returncode == 0 and runtime_provenance.is_file()
    runtime_sha = sha256(runtime_provenance)
    started = datetime.now(timezone.utc).isoformat()
    if geometry_gate == "PASS" and model_gate and runtime_gate:
        result = subprocess.run(command, cwd=out, env=env, text=True, capture_output=True, check=False)
    else:
        result = subprocess.CompletedProcess(command, 2, "", ";".join(filter(None, [geometry_gate_reason, "runtime_provenance_gate_failed" if not runtime_gate else ""])))
    (out / "runner_stdout.txt").write_text(result.stdout)
    (out / "runner_stderr.txt").write_text(result.stderr)
    prediction_pass, prediction_reason = prediction_gate(output_csv, int(row["n_images"]), row["model_label"])
    profile_pass = False
    profile_reason = "prediction_gate_failed"
    profile_path = out / "fixed_profile.csv"
    if prediction_pass:
        profile_pass, profile_result = write_normalized_profile(output_csv, int(row["n_images"]), row["model_label"])
        profile_reason = "" if profile_pass else profile_result
    gate_pass = geometry_gate == "PASS" and model_gate and runtime_gate and result.returncode == 0 and prediction_pass and profile_pass
    gate_reason = "" if gate_pass else ";".join(filter(None, [geometry_gate_reason, "runtime_provenance_gate_failed" if not runtime_gate else "", prediction_reason, profile_reason]))
    scientific_status = "FINITE_DIAGNOSTIC" if result.returncode == 0 and gate_pass else "FAILURE_MODE"
    manifest = {
        "schema_version": "foundation_fixed_path_native_mace_v1",
        "campaign_id": "foundation_interlayer_sliding_2x2_20261024",
        "record_id": row["record_id"],
        "observable": row["observable"],
        "system": row["system"],
        "plot_column": row["plot_column"],
        "atoms": int(row["atoms"]),
        "n_images_target": int(row["n_images"]),
        "dft_reference_id": row["dft_reference_id"],
        "dft_reference_status": row["dft_status"],
        "model_id": row["model_id"],
        "model_label": row["model_label"],
        "model_path": str(model),
        "model_sha256_observed": observed_model_sha,
        "model_sha256_registry": registry["sha256"],
        "model_registry_path": str(model_registry),
        "model_registry_sha256": sha256(model_registry),
        "input_xyz": str(xyz),
        "input_xyz_sha256": sha256(xyz),
        "input_neb_out": str(neb_out),
        "input_neb_out_sha256": sha256(neb_out),
        "runner_path": str(script),
        "runner_sha256": sha256(script),
        "original_recovery_script_path": str(original_script),
        "original_recovery_script_sha256": sha256(original_script),
        "wrapper_path": str(Path(__file__).resolve()),
        "wrapper_sha256": sha256(Path(__file__).resolve()),
        "acceptance_freeze": str(freeze),
        "acceptance_freeze_sha256": sha256(freeze),
        "runtime_provenance_path": str(runtime_provenance),
        "runtime_provenance_sha256": runtime_sha,
        "environment_sha256": runtime_sha,
        "runtime_provenance_capture_return_code": capture_result.returncode,
        "job_table_path": str(args.job_table),
        "job_table_sha256": sha256(args.job_table),
        "created_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": scheduler_job,
        "slurm_array_task_id": array_task,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "environment_versions": package_versions(),
        "return_code": result.returncode,
        "prediction_csv": str(output_csv),
        "prediction_csv_sha256": sha256(output_csv),
        "fixed_profile_csv": str(profile_path),
        "fixed_profile_csv_sha256": sha256(profile_path),
        "geometry_gate": geometry_gate,
        "model_registry_gate": "PASS" if model_gate else "FAIL",
        "prediction_gate": "PASS" if prediction_pass else "FAIL",
        "profile_normalization_gate": "PASS" if profile_pass else "FAIL",
        "runtime_provenance_gate": "PASS" if runtime_gate else "FAIL",
        "prediction_gate_reason": gate_reason,
        "scientific_status": scientific_status,
        "publication_status": "PRIMARY_EVIDENCE" if scientific_status == "FINITE_DIAGNOSTIC" else "DIAGNOSTIC_ONLY",
        "failure_class": "NA" if scientific_status != "FAILURE_MODE" else failure_class(result.stdout + "\n" + result.stderr),
        "failure_reason": "" if scientific_status != "FAILURE_MODE" else gate_reason,
    }
    manifest_path = out / "fixed_path_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"record_id": row["record_id"], "scientific_status": scientific_status, "prediction_gate": gate_pass}, indent=2))


if __name__ == "__main__":
    main()
