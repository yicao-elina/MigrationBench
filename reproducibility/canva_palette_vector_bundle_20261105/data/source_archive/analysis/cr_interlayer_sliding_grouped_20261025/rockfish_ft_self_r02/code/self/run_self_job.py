#!/usr/bin/env python3
"""Run and gate one endpoint-only self-relaxed MLIP NEB row."""

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


def endpoint_count(path: Path) -> int:
    for line in path.read_text().splitlines():
        if line.strip():
            return int(line.strip())
    raise ValueError(f"empty endpoint: {path}")


def package_versions() -> dict[str, str | None]:
    out = {}
    for name in ("ase", "mace-torch", "torch", "numpy"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def profile_gate(engine: dict, expected: int) -> tuple[bool, str]:
    profile = engine.get("profile")
    if not isinstance(profile, list) or len(profile) != expected:
        return False, "profile_length_mismatch"
    values = []
    try:
        for item in profile:
            values.extend([float(item["energy_eV"]), float(item["rel_energy_eV"]), float(item["fmax_eV_A"])])
        residual = float(engine["final_max_neb_residual_eV_A"])
        neb_force = float(engine["final_max_internal_neb_force_eV_A"])
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"profile_schema_error:{exc}"
    if not all(math.isfinite(x) for x in values + [residual, neb_force]):
        return False, "nonfinite_profile_or_residual"
    if residual > 0.05 or neb_force > 0.05:
        return False, f"residual_gate_failed:{residual}:{neb_force}"
    if not bool(engine.get("optimizer_converged")):
        return False, "optimizer_not_converged"
    return True, ""


def max_endpoint_delta(images_path: Path, initial: Path, final: Path, atoms: int) -> tuple[float | None, str]:
    try:
        from ase.io import read
        images = read(images_path, index=":")
        start = read(initial)
        end = read(final)
        if len(images) != 10 or len(start) != atoms or len(end) != atoms:
            return None, "endpoint_output_shape_mismatch"
        delta0 = float(abs(images[0].positions - start.positions).max())
        delta1 = float(abs(images[-1].positions - end.positions).max())
        return max(delta0, delta1), ""
    except Exception as exc:
        return None, f"endpoint_output_parse_error:{type(exc).__name__}:{exc}"


def fail_class(text: str) -> str:
    low = text.lower()
    if any(x in low for x in ("deserial", "torch.jit", "unpickl", "load model")):
        return "MODEL_LOAD_OR_DESERIALIZATION"
    if any(x in low for x in ("nan", "inf", "nonfinite", "non-finite")):
        return "NONFINITE"
    if any(x in low for x in ("no such file", "not found", "input")):
        return "INPUT_OR_PATH"
    if "conver" in low or "residual" in low:
        return "OPTIMIZER_OR_SCIENTIFIC_GATE"
    return "RUNNER_EXCEPTION"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-table", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--schema-version", required=True)
    parser.add_argument("--freeze-name", required=True)
    args = parser.parse_args()

    row = read_rows(args.job_table)[args.index]
    scheduler_job = os.environ.get("SLURM_JOB_ID", f"LOCAL_{args.index}")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID", str(args.index))
    out = args.root / "runs" / "self" / row["system"] / row["model_id"] / f"attempt_{scheduler_job}_{array_task}"
    out.mkdir(parents=True, exist_ok=True)
    registry_path = args.root / "MODEL_REGISTRY.tsv"
    freeze = args.root / args.freeze_name
    runner = args.root / "code/self/run_mlff_neb.py"
    helper = args.root / "code/self/export_mlff_neb_iteration_history.py"
    provenance_script = args.root / "code/capture_runtime_provenance.py"
    registry = registry_row(registry_path, row["model_id"])
    model = Path(row["model_path"])
    initial = Path(row["initial_endpoint"])
    final = Path(row["final_endpoint"])
    runtime = out / "runtime_provenance.json"
    engine_manifest = out / "mlff_neb_manifest.json"
    started = datetime.now(timezone.utc).isoformat()

    model_observed = sha256(model)
    model_gate = model_observed == registry["model_sha256"]
    endpoint_gate = "PASS"
    endpoint_reason = ""
    try:
        counts = (endpoint_count(initial), endpoint_count(final))
        if counts != (int(row["atoms"]), int(row["atoms"])):
            endpoint_gate = "FAIL"
            endpoint_reason = f"expected_{row['atoms']}_found_{counts}"
    except Exception as exc:
        endpoint_gate = "FAIL"
        endpoint_reason = f"endpoint_parse_error:{type(exc).__name__}:{exc}"

    capture = [
        sys.executable, str(provenance_script), "--output", str(runtime), "--binary", sys.executable,
        "--input", f"model={model}", "--input", f"initial={initial}", "--input", f"final={final}",
        "--input", f"registry={registry_path}", "--input", f"job_table={args.job_table}",
        "--input", f"freeze={freeze}", "--input", f"runner={runner}", "--input", f"helper={helper}",
        "--input", f"wrapper={Path(__file__).resolve()}",
    ]
    capture_result = subprocess.run(capture, cwd=out, text=True, capture_output=True, check=False)
    runtime_gate = capture_result.returncode == 0 and runtime.is_file()
    command = [
        sys.executable, str(runner), "--initial", str(initial), "--final", str(final),
        "--n-images", row["n_images"], "--out-dir", str(out), "--calculator", "mace-model",
        "--model-path", str(model), "--device", "cpu", "--default-dtype", "float64",
        "--steps", row["steps"], "--seed", row["seed"], "--fmax", row["fmax_eV_A"],
        "--spring-constant", "0.1", "--fire-dt", "0.1", "--fire-maxstep", "0.2",
        "--run-role", "production_self_relaxed_endpoint_only", "--interpolate-method", "idpp",
        "--source-candidate-manifest", str(freeze),
    ]
    if row.get("model_head", ""):
        command += ["--model-head", row["model_head"]]
    (out / "endpoint_only_command.json").write_text(json.dumps({"command": command, "row": row}, indent=2, sort_keys=True) + "\n")

    if endpoint_gate == "PASS" and model_gate and runtime_gate:
        result = subprocess.run(command, cwd=out, text=True, capture_output=True, check=False)
    else:
        reason = ";".join(filter(None, [endpoint_reason, "model_registry_gate_failed" if not model_gate else "", "runtime_provenance_gate_failed" if not runtime_gate else ""]))
        result = subprocess.CompletedProcess(command, 2, "", reason)
    (out / "runner_stdout.txt").write_text(result.stdout)
    (out / "runner_stderr.txt").write_text(result.stderr)

    engine = None
    parse_reason = "missing_engine_manifest"
    if engine_manifest.is_file():
        try:
            engine = json.loads(engine_manifest.read_text())
            parse_reason = ""
        except Exception as exc:
            parse_reason = f"engine_manifest_parse_error:{type(exc).__name__}:{exc}"
    finite_ok, finite_reason = (profile_gate(engine, int(row["n_images"])) if engine else (False, parse_reason))
    images = out / "mlff_neb_images.extxyz"
    endpoint_delta, endpoint_output_reason = (max_endpoint_delta(images, initial, final, int(row["atoms"])) if engine else (None, "missing_images"))
    endpoint_output_gate = endpoint_delta is not None and endpoint_delta <= 1.0e-8
    gate = bool(result.returncode == 0 and engine and model_gate and endpoint_gate == "PASS" and runtime_gate and finite_ok and endpoint_output_gate and images.is_file() and engine.get("scientific_gate"))
    status = "CONVERGED" if gate else "FAILURE_MODE"
    reason = "" if gate else ";".join(filter(None, [endpoint_reason, "model_registry_gate_failed" if not model_gate else "", "runtime_provenance_gate_failed" if not runtime_gate else "", finite_reason, endpoint_output_reason, "endpoint_tolerance_failed" if not endpoint_output_gate else "", "engine_scientific_gate_failed" if engine and not engine.get("scientific_gate") else ""]))
    manifest = {
        "schema_version": args.schema_version, "campaign_id": args.campaign_id,
        "record_id": row["record_id"], "observable": row["observable"], "system": row["system"], "plot_column": row["plot_column"],
        "atoms": int(row["atoms"]), "n_images_target": int(row["n_images"]), "steps_target": int(row["steps"]),
        "fmax_target_eV_A": float(row["fmax_eV_A"]), "spring_constant": 0.1, "seed": int(row["seed"]),
        "initialization": "ASE_IDPP_FROM_ENDPOINTS_ONLY", "endpoint_policy": "FIXED_ENDPOINTS",
        "initial_endpoint": str(initial), "initial_endpoint_sha256": sha256(initial), "final_endpoint": str(final), "final_endpoint_sha256": sha256(final),
        "dft_reference_id": row["dft_reference_id"], "dft_reference_status": row["dft_status"],
        "model_id": row["model_id"], "model_label": row["model_label"], "model_path": str(model), "model_head": row.get("model_head", ""),
        "model_sha256_observed": model_observed, "model_sha256_registry": registry["model_sha256"], "model_registry_path": str(registry_path),
        "model_registry_sha256": sha256(registry_path), "runner_path": str(runner), "runner_sha256": sha256(runner),
        "helper_path": str(helper), "helper_sha256": sha256(helper), "wrapper_path": str(Path(__file__).resolve()), "wrapper_sha256": sha256(Path(__file__).resolve()),
        "acceptance_freeze": str(freeze), "acceptance_freeze_sha256": sha256(freeze), "runtime_provenance_path": str(runtime),
        "runtime_provenance_sha256": sha256(runtime), "environment_sha256": sha256(runtime), "job_table_path": str(args.job_table),
        "job_table_sha256": sha256(args.job_table), "job_id": scheduler_job, "array_task_id": array_task,
        "python_executable": sys.executable, "python_version": sys.version, "environment_versions": package_versions(), "return_code": result.returncode,
        "engine_manifest": str(engine_manifest), "engine_manifest_sha256": sha256(engine_manifest), "engine": engine,
        "endpoint_input_gate": endpoint_gate, "model_registry_gate": "PASS" if model_gate else "FAIL", "runtime_provenance_gate": "PASS" if runtime_gate else "FAIL",
        "profile_gate": "PASS" if finite_ok else "FAIL", "endpoint_output_gate": "PASS" if endpoint_output_gate else "FAIL",
        "endpoint_max_abs_displacement_A": endpoint_delta, "scientific_status": status, "publication_status": "PRIMARY_EVIDENCE" if gate else "DIAGNOSTIC_ONLY",
        "scientific_gate": "PASS" if gate else "FAIL", "failure_class": "NA" if gate else fail_class(result.stdout + "\n" + result.stderr),
        "failure_reason": reason, "created_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = out / "endpoint_only_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"record_id": row["record_id"], "scientific_status": status, "reason": reason}, sort_keys=True))


if __name__ == "__main__":
    main()
