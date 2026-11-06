#!/usr/bin/env python3
"""Run and gate one fixed-path MACE single-point row."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def registry_row(path: Path, model_id: str) -> dict[str, str]:
    for row in read_rows(path):
        if row["model_id"] == model_id:
            return row
    raise KeyError(model_id)


def xyz_counts(path: Path) -> list[int]:
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
    out = {}
    for name in ("ase", "mace-torch", "torch", "numpy"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def fail_class(text: str) -> str:
    low = text.lower()
    if any(x in low for x in ("deserial", "torch.jit", "unpickl", "load model")):
        return "MODEL_LOAD_OR_DESERIALIZATION"
    if any(x in low for x in ("nan", "inf", "nonfinite", "non-finite")):
        return "NONFINITE"
    if any(x in low for x in ("no such file", "not found", "input")):
        return "INPUT_OR_PATH"
    return "RUNNER_EXCEPTION"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-table", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    row = read_rows(args.job_table)[args.index]
    scheduler_job = os.environ.get("SLURM_JOB_ID", f"LOCAL_{args.index}")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID", str(args.index))
    out = args.root / "runs" / "fixed" / row["system"] / row["model_id"] / f"attempt_{scheduler_job}_{array_task}"
    out.mkdir(parents=True, exist_ok=True)

    registry_path = args.root / "MODEL_REGISTRY.tsv"
    freeze = args.root / "ACCEPTANCE_FREEZE_R01.json"
    runner = args.root / "code/fixed/eval_fixed_ase.py"
    provenance_script = args.root / "code/capture_runtime_provenance.py"
    registry = registry_row(registry_path, row["model_id"])
    model = Path(row["model_path"])
    xyz = Path(row["xyz_path"])
    neb_out = Path(row["neb_out_path"])
    prediction = out / "fixed_predictions.csv"
    profile = out / "fixed_profile.csv"
    runtime = out / "runtime_provenance.json"
    started = datetime.now(timezone.utc).isoformat()

    geometry_gate = "PASS"
    geometry_reason = ""
    try:
        counts = xyz_counts(xyz)
        expected = [int(row["atoms"])] * int(row["n_images"])
        if counts != expected:
            geometry_gate = "FAIL"
            geometry_reason = f"expected_{expected}_found_{counts}"
    except Exception as exc:
        geometry_gate = "FAIL"
        geometry_reason = f"xyz_parse_error:{type(exc).__name__}:{exc}"
    model_observed = sha256(model)
    model_gate = model_observed == registry["model_sha256"]
    if not model_gate:
        geometry_reason = ";".join(filter(None, [geometry_reason, f"model_sha256_mismatch:{model_observed}:{registry['model_sha256']}"]))

    capture = [
        sys.executable, str(provenance_script), "--output", str(runtime), "--binary", sys.executable,
        "--input", f"model={model}", "--input", f"xyz={xyz}", "--input", f"neb_out={neb_out}",
        "--input", f"registry={registry_path}", "--input", f"job_table={args.job_table}",
        "--input", f"freeze={freeze}", "--input", f"runner={runner}", "--input", f"wrapper={Path(__file__).resolve()}",
    ]
    capture_result = subprocess.run(capture, cwd=out, text=True, capture_output=True, check=False)
    runtime_gate = capture_result.returncode == 0 and runtime.is_file()

    command = [
        sys.executable, str(runner), "--model", str(model), "--model-label", row["model_label"],
        "--xyz", str(xyz), "--neb-out", str(neb_out), "--output", str(prediction),
    ]
    if row.get("model_head", ""):
        command += ["--head", row["model_head"]]
    (out / "command.json").write_text(json.dumps({"command": command, "row": row}, indent=2, sort_keys=True) + "\n")
    if geometry_gate == "PASS" and model_gate and runtime_gate:
        result = subprocess.run(command, cwd=out, text=True, capture_output=True, check=False)
    else:
        reason = ";".join(filter(None, [geometry_reason, "runtime_provenance_gate_failed" if not runtime_gate else ""]))
        result = subprocess.CompletedProcess(command, 2, "", reason)
    (out / "runner_stdout.txt").write_text(result.stdout)
    (out / "runner_stderr.txt").write_text(result.stderr)

    prediction_gate = False
    prediction_reason = "missing_prediction"
    rows = []
    if prediction.is_file():
        try:
            with prediction.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            indices = [int(r["image_index"]) for r in rows]
            values = [float(r["predicted_energy_eV"]) for r in rows]
            forces = [float(r["max_force_eV_A"]) for r in rows]
            names = {r.get("model_name", "") for r in rows}
            prediction_gate = (
                result.returncode == 0 and len(rows) == int(row["n_images"]) and
                indices == list(range(int(row["n_images"]))) and names == {row["model_label"]} and
                all(math.isfinite(x) for x in values + forces)
            )
            prediction_reason = "" if prediction_gate else "prediction_schema_or_finite_gate_failed"
        except Exception as exc:
            prediction_reason = f"prediction_parse_error:{type(exc).__name__}:{exc}"
    profile_gate = False
    profile_reason = "prediction_gate_failed"
    if prediction_gate:
        e0 = float(rows[0]["predicted_energy_eV"])
        with profile.open("w", newline="") as handle:
            fields = ["image_index", "image_fraction", "model_name", "predicted_energy_eV", "relative_energy_eV"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in rows:
                index = int(item["image_index"])
                energy = float(item["predicted_energy_eV"])
                writer.writerow({
                    "image_index": index,
                    "image_fraction": index / (int(row["n_images"]) - 1),
                    "model_name": item["model_name"],
                    "predicted_energy_eV": f"{energy:.16g}",
                    "relative_energy_eV": f"{energy - e0:.16g}",
                })
        profile_gate = profile.is_file() and profile.stat().st_size > 0
        profile_reason = "" if profile_gate else "profile_write_failed"

    gate = geometry_gate == "PASS" and model_gate and runtime_gate and prediction_gate and profile_gate
    status = "FINITE_DIAGNOSTIC" if gate else "FAILURE_MODE"
    reason = "" if gate else ";".join(filter(None, [geometry_reason, "model_registry_gate_failed" if not model_gate else "", "runtime_provenance_gate_failed" if not runtime_gate else "", prediction_reason, profile_reason]))
    manifest = {
        "schema_version": "foundation_fixed_path_10model_v1",
        "campaign_id": "foundation_interlayer_sliding_10model_20261024",
        "record_id": row["record_id"], "observable": row["observable"], "system": row["system"],
        "plot_column": row["plot_column"], "atoms": int(row["atoms"]), "n_images_target": int(row["n_images"]),
        "dft_reference_id": row["dft_reference_id"], "dft_reference_status": row["dft_status"],
        "model_id": row["model_id"], "model_label": row["model_label"], "model_path": str(model),
        "model_head": row.get("model_head", ""), "model_sha256_observed": model_observed,
        "model_sha256_registry": registry["model_sha256"], "model_registry_path": str(registry_path),
        "model_registry_sha256": sha256(registry_path), "input_xyz": str(xyz), "input_xyz_sha256": sha256(xyz),
        "input_neb_out": str(neb_out), "input_neb_out_sha256": sha256(neb_out), "runner_path": str(runner),
        "runner_sha256": sha256(runner), "wrapper_path": str(Path(__file__).resolve()),
        "wrapper_sha256": sha256(Path(__file__).resolve()), "acceptance_freeze": str(freeze),
        "acceptance_freeze_sha256": sha256(freeze), "runtime_provenance_path": str(runtime),
        "runtime_provenance_sha256": sha256(runtime), "environment_sha256": sha256(runtime),
        "job_table_path": str(args.job_table), "job_table_sha256": sha256(args.job_table),
        "job_id": scheduler_job, "array_task_id": array_task, "python_executable": sys.executable,
        "python_version": sys.version, "environment_versions": package_versions(), "return_code": result.returncode,
        "prediction_csv": str(prediction), "prediction_csv_sha256": sha256(prediction), "fixed_profile_csv": str(profile),
        "fixed_profile_csv_sha256": sha256(profile), "geometry_gate": geometry_gate,
        "model_registry_gate": "PASS" if model_gate else "FAIL", "runtime_provenance_gate": "PASS" if runtime_gate else "FAIL",
        "prediction_gate": "PASS" if prediction_gate else "FAIL", "profile_gate": "PASS" if profile_gate else "FAIL",
        "scientific_status": status, "publication_status": "PRIMARY_EVIDENCE" if gate else "DIAGNOSTIC_ONLY",
        "scientific_gate": "PASS" if gate else "FAIL", "failure_class": "NA" if gate else fail_class(result.stdout + "\n" + result.stderr),
        "failure_reason": reason, "created_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = out / "fixed_path_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"record_id": row["record_id"], "scientific_status": status, "reason": reason}, sort_keys=True))


if __name__ == "__main__":
    main()
