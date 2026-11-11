#!/usr/bin/env python3
"""Run and audit one endpoint-only self-relaxed MLIP NEB row."""

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


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("ase", "mace-torch", "torch", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def extxyz_atom_count(path: Path) -> int:
    for line in path.read_text().splitlines():
        if line.strip():
            return int(line.strip())
    raise ValueError(f"empty endpoint file: {path}")


def finite_profile(engine: dict, expected_images: int) -> tuple[bool, str]:
    profile = engine.get("profile")
    if not isinstance(profile, list) or len(profile) != expected_images:
        return False, "profile_length_mismatch"
    values = []
    for row in profile:
        try:
            values.extend([float(row["energy_eV"]), float(row["rel_energy_eV"]), float(row["fmax_eV_A"])])
        except (KeyError, TypeError, ValueError):
            return False, "profile_schema_error"
    if not all(math.isfinite(value) for value in values):
        return False, "nonfinite_profile"
    residual = engine.get("final_max_neb_residual_eV_A")
    neb_force = engine.get("final_max_internal_neb_force_eV_A")
    if residual is None or neb_force is None or not all(math.isfinite(float(value)) for value in (residual, neb_force)):
        return False, "missing_or_nonfinite_residual"
    if float(residual) > 0.05 or float(neb_force) > 0.05:
        return False, f"residual_gate_failed:{residual},{neb_force}"
    if not bool(engine.get("optimizer_converged")):
        return False, "optimizer_not_converged"
    return True, ""


def endpoint_displacement(final_images: Path, initial: Path, final: Path, expected_atoms: int) -> tuple[float | None, str | None]:
    try:
        from ase.io import read
        import numpy as np

        images = read(final_images, index=":")
        initial_atoms = read(initial)
        final_atoms = read(final)
        if len(images) != 10:
            return None, f"expected_10_final_images_found_{len(images)}"
        if len(initial_atoms) != expected_atoms or len(final_atoms) != expected_atoms:
            return None, f"input_endpoint_atom_count_mismatch:{len(initial_atoms)},{len(final_atoms)} expected {expected_atoms}"
        if len(images[0]) != len(initial_atoms) or len(images[-1]) != len(final_atoms):
            return None, "endpoint_atom_count_mismatch"
        initial_delta = float(np.max(np.abs(images[0].positions - initial_atoms.positions)))
        final_delta = float(np.max(np.abs(images[-1].positions - final_atoms.positions)))
        return max(initial_delta, final_delta), None
    except Exception as exc:
        return None, f"endpoint_parse_error:{type(exc).__name__}:{exc}"


def failure_class(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("deserial", "attributeerror", "torch.jit", "unpickl", "unknown class")):
        return "MODEL_LOAD_OR_DESERIALIZATION"
    if any(token in lowered for token in ("nan", "inf", "nonfinite", "non-finite")):
        return "NONFINITE"
    if "migrant element" in lowered or "chemical symbol" in lowered:
        return "ELEMENT_IDENTITY"
    if "no such file" in lowered or "not found" in lowered:
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
        / "self"
        / row["system"]
        / row["model_id"]
        / f"attempt_{scheduler_job}_{array_task}"
    )
    out.mkdir(parents=True, exist_ok=True)
    runner = args.root / "code/self/run_mlff_neb.py"
    helper = args.root / "code/self/export_mlff_neb_iteration_history.py"
    provenance_script = args.root / "code/capture_runtime_provenance.py"
    runtime_provenance = out / "runtime_provenance.json"
    freeze = args.root / "ACCEPTANCE_FREEZE_R13.json"
    model_registry = args.root / "MODEL_REGISTRY.tsv"
    registry = registry_row(model_registry, row["model_id"])
    initial = Path(row["initial_endpoint"])
    final = Path(row["final_endpoint"])
    model = Path(row["model_path"])
    observed_model_sha = sha256(model)
    model_gate = observed_model_sha == registry["sha256"]
    endpoint_input_gate = "PASS"
    endpoint_input_gate_reason = ""
    try:
        endpoint_counts = (extxyz_atom_count(initial), extxyz_atom_count(final))
        if endpoint_counts != (int(row["atoms"]), int(row["atoms"])):
            endpoint_input_gate = "FAIL"
            endpoint_input_gate_reason = f"expected_endpoint_atoms_{row['atoms']}_found_{endpoint_counts}"
    except Exception as exc:
        endpoint_input_gate = "FAIL"
        endpoint_input_gate_reason = f"endpoint_input_parse_error:{type(exc).__name__}:{exc}"
    command = [
        sys.executable,
        str(runner),
        "--initial", str(initial),
        "--final", str(final),
        "--n-images", row["n_images"],
        "--out-dir", str(out),
        "--calculator", "mace-model",
        "--model-path", str(model),
        "--device", "cpu",
        "--default-dtype", "float64",
        "--steps", row["steps"],
        "--seed", row["seed"],
        "--fmax", row["fmax_eV_A"],
        "--spring-constant", "0.1",
        "--fire-dt", "0.1",
        "--fire-maxstep", "0.2",
        "--run-role", "production_self_relaxed_endpoint_only",
        "--interpolate-method", "idpp",
        "--source-candidate-manifest", str(freeze),
    ]
    (out / "endpoint_only_command.json").write_text(json.dumps({
        "command": command,
        "endpoint_inputs_only": True,
        "intermediate_inputs_used": [],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "observable": row["observable"],
    }, indent=2) + "\n")
    capture_command = [
        sys.executable, str(provenance_script), "--output", str(runtime_provenance),
        "--binary", sys.executable,
        "--input", f"model={model}",
        "--input", f"initial_endpoint={initial}",
        "--input", f"final_endpoint={final}",
        "--input", f"model_registry={model_registry}",
        "--input", f"job_table={args.job_table}",
        "--input", f"freeze={freeze}",
        "--input", f"runner={runner}",
        "--input", f"wrapper={Path(__file__).resolve()}",
    ]
    capture_result = subprocess.run(capture_command, cwd=out, text=True, capture_output=True, check=False)
    runtime_gate = capture_result.returncode == 0 and runtime_provenance.is_file()
    runtime_sha = sha256(runtime_provenance)
    started = datetime.now(timezone.utc).isoformat()
    if model_gate and endpoint_input_gate == "PASS" and runtime_gate:
        result = subprocess.run(command, cwd=runner.parent, text=True, capture_output=True, check=False)
    else:
        reasons = [endpoint_input_gate_reason, "runtime_provenance_gate_failed" if not runtime_gate else ""]
        if not model_gate:
            reasons.append(f"model_sha256_mismatch:{observed_model_sha}:{registry['sha256']}")
        result = subprocess.CompletedProcess(command, 2, "", ";".join(filter(None, reasons)))
    (out / "runner_stdout.txt").write_text(result.stdout)
    (out / "runner_stderr.txt").write_text(result.stderr)
    engine_manifest = out / "mlff_neb_manifest.json"
    engine = None
    parse_error = None
    if engine_manifest.is_file():
        try:
            engine = json.loads(engine_manifest.read_text())
        except Exception as exc:
            parse_error = f"{type(exc).__name__}:{exc}"
    gate_pass = False
    gate_reason = "runner_failed"
    endpoint_delta = None
    endpoint_reason = None
    if result.returncode == 0 and engine is not None and model_gate:
        profile_pass, profile_reason = finite_profile(engine, int(row["n_images"]))
        final_images = out / "mlff_neb_images.extxyz"
        endpoint_delta, endpoint_reason = endpoint_displacement(final_images, initial, final, int(row["atoms"]))
        endpoint_pass = endpoint_delta is not None and endpoint_delta <= 1.0e-8
        gate_pass = endpoint_input_gate == "PASS" and runtime_gate and profile_pass and endpoint_pass and final_images.is_file() and bool(engine.get("scientific_gate"))
        gate_reason = "" if gate_pass else ";".join(filter(None, [endpoint_input_gate_reason, "runtime_provenance_gate_failed" if not runtime_gate else "", profile_reason, endpoint_reason, "endpoint_tolerance_failed" if not endpoint_pass else ""]))
    else:
        gate_reason = parse_error or failure_class(result.stdout + "\n" + result.stderr)
    scientific_status = "CONVERGED" if gate_pass else "FAILURE_MODE"
    manifest = {
        "schema_version": "foundation_self_relaxed_endpoint_only_v1",
        "campaign_id": "foundation_interlayer_sliding_2x2_20261024",
        "record_id": row["record_id"],
        "observable": row["observable"],
        "system": row["system"],
        "plot_column": row["plot_column"],
        "atoms": int(row["atoms"]),
        "n_images_target": int(row["n_images"]),
        "steps_target": int(row["steps"]),
        "fmax_target_eV_A": float(row["fmax_eV_A"]),
        "seed": int(row["seed"]),
        "initialization": "ASE_IDPP_FROM_ENDPOINTS_ONLY",
        "intermediate_inputs_used": [],
        "endpoint_policy": "FIXED_ENDPOINTS",
        "initial_endpoint": str(initial),
        "initial_endpoint_sha256": sha256(initial),
        "final_endpoint": str(final),
        "final_endpoint_sha256": sha256(final),
        "model_id": row["model_id"],
        "model_label": row["model_label"],
        "model_path": str(model),
        "model_sha256_observed": observed_model_sha,
        "model_sha256_registry": registry["sha256"],
        "model_registry_path": str(model_registry),
        "model_registry_sha256": sha256(model_registry),
        "runner_path": str(runner),
        "runner_sha256": sha256(runner),
        "reference_runner_path": str(args.root / "code/self/run_mlff_neb.reference.py"),
        "reference_runner_sha256": sha256(args.root / "code/self/run_mlff_neb.reference.py"),
        "history_helper_path": str(helper),
        "history_helper_sha256": sha256(helper),
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
        "dft_reference_id": row["dft_reference_id"],
        "dft_reference_status": row["dft_status"],
        "created_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": scheduler_job,
        "slurm_array_task_id": array_task,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "environment_versions": package_versions(),
        "return_code": result.returncode,
        "engine_manifest": str(engine_manifest),
        "engine_manifest_sha256": sha256(engine_manifest),
        "engine": engine,
        "endpoint_max_abs_displacement_A": endpoint_delta,
        "endpoint_input_gate": endpoint_input_gate,
        "endpoint_input_gate_reason": endpoint_input_gate_reason,
        "endpoint_gate": "PASS" if endpoint_delta is not None and endpoint_delta <= 1.0e-8 else "FAIL",
        "model_registry_gate": "PASS" if model_gate else "FAIL",
        "runtime_provenance_gate": "PASS" if runtime_gate else "FAIL",
        "scientific_status": scientific_status,
        "publication_status": "PRIMARY_EVIDENCE" if gate_pass else "DIAGNOSTIC_ONLY",
        "scientific_gate": "PASS" if gate_pass else "FAIL",
        "scientific_gate_reason": gate_reason,
        "failure_class": "NA" if gate_pass else failure_class(result.stdout + "\n" + result.stderr),
        "failure_reason": "" if gate_pass else gate_reason,
    }
    output = out / "endpoint_only_manifest.json"
    manifest["manifest_path"] = str(output)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"record_id": row["record_id"], "scientific_status": scientific_status, "scientific_gate": gate_pass}, indent=2))


if __name__ == "__main__":
    main()
