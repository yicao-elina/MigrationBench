#!/usr/bin/env python3
"""Run the existing audited MLFF-NEB engine from two endpoints only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def failure_class(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("deserial", "attributeerror", "torch.jit", "unpickl", "unknown class")):
        return "DESERIALIZATION"
    if any(token in lowered for token in ("nan", "inf", "nonfinite")):
        return "NONFINITE"
    return "RUNNER_EXCEPTION"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-head", default="default")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--path-id", required=True)
    parser.add_argument("--path-class", required=True)
    parser.add_argument("--initialization-method", choices=["IDPP", "MIC_LINEAR"], required=True)
    parser.add_argument("--endpoint-hash-initial", required=True)
    parser.add_argument("--endpoint-hash-final", required=True)
    parser.add_argument("--cell-hash", required=True)
    parser.add_argument("--dft-reference-id", required=True)
    parser.add_argument("--dft-reference-status", required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--neb-seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    model_sha = sha256(args.model_path) if args.model_path.is_file() else ""
    initial_sha = sha256(args.initial)
    final_sha = sha256(args.final)
    runner_sha = sha256(args.runner)
    wrapper_sha = sha256(Path(__file__).resolve())
    command = [
        os.environ.get("PYTHON", "python"), str(args.runner),
        "--initial", str(args.initial), "--final", str(args.final),
        "--n-images", "10", "--out-dir", str(args.out_dir),
        "--calculator", "mace-model", "--model-path", str(args.model_path),
        "--model-head", args.model_head, "--device", args.device,
        "--steps", str(args.steps), "--seed", str(args.neb_seed),
        "--fmax", str(args.fmax), "--spring-constant", "0.1",
        "--fire-dt", "0.1", "--fire-maxstep", "0.2",
        "--migrant-element", "Cr", "--run-role", "production_preconditioner",
        "--interpolate-method", "idpp" if args.initialization_method == "IDPP" else "linear",
        "--source-candidate-manifest", str(args.campaign_manifest),
    ]
    (args.out_dir / "endpoint_only_command.json").write_text(json.dumps({"command": command, "endpoint_inputs_only": True, "intermediate_inputs_used": [], "created_at_utc": created}, indent=2) + "\n")
    result = subprocess.run(command, cwd=args.runner.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    (args.out_dir / "runner_stdout.txt").write_text(result.stdout)
    (args.out_dir / "runner_stderr.txt").write_text(result.stderr)
    base = {
        "schema_version": "migrationbench_endpoint_only_self_neb_v1",
        "record_id": args.record_id,
        "campaign_id": "scratch_600k_migration_canonical_20261101",
        "observable": "SELF_RELAXED_ENDPOINT_ONLY_NEB",
        "initialization_method": args.initialization_method,
        "initialization_source": "ENDPOINTS_ONLY",
        "endpoint_policy": "FIXED_ENDPOINTS",
        "leakage_gate": "PASS",
        "intermediate_inputs_used": [],
        "initial_endpoint_path": str(args.initial),
        "initial_endpoint_sha256": initial_sha,
        "final_endpoint_path": str(args.final),
        "final_endpoint_sha256": final_sha,
        "endpoint_hash_initial": args.endpoint_hash_initial,
        "endpoint_hash_final": args.endpoint_hash_final,
        "cell_hash": args.cell_hash,
        "path_id": args.path_id,
        "path_class": args.path_class,
        "dft_reference_id": args.dft_reference_id,
        "dft_reference_status": args.dft_reference_status,
        "model_id": args.model_id,
        "model_label": args.model_label,
        "model_path": str(args.model_path),
        "model_sha256": model_sha,
        "model_head": args.model_head,
        "neb_seed": args.neb_seed,
        "runner_path": str(args.runner),
        "runner_sha256": runner_sha,
        "wrapper_path": str(Path(__file__).resolve()),
        "wrapper_sha256": wrapper_sha,
        "campaign_manifest": str(args.campaign_manifest),
        "campaign_manifest_sha256": sha256(args.campaign_manifest),
        "created_at_utc": created,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "LOCAL"),
    }
    engine_manifest = args.out_dir / "mlff_neb_manifest.json"
    if result.returncode == 0 and engine_manifest.is_file():
        engine = json.loads(engine_manifest.read_text())
        residual = engine.get("final_max_neb_residual_eV_A")
        neb_force = engine.get("final_max_internal_neb_force_eV_A")
        converged = bool(engine.get("optimizer_converged")) and all(
            value is not None and float(value) <= args.fmax for value in (residual, neb_force)
        )
        base.update(
            {
                "engine_manifest": str(engine_manifest),
                "engine_manifest_sha256": sha256(engine_manifest),
                "engine_outputs": engine.get("outputs", {}),
                "optimizer_converged": bool(engine.get("optimizer_converged")),
                "target_model_convergence": converged,
                "final_max_internal_neb_force_eV_A": neb_force,
                "final_max_neb_residual_eV_A": residual,
                "forward_barrier_eV": engine.get("barrier_proxy_eV"),
                "trajectory_sha256": engine.get("outputs", {}).get("images_sha256"),
                "scientific_status": "CONVERGED" if converged else "NONCONVERGED",
                "publication_status": "PRIMARY_EVIDENCE" if converged else "DIAGNOSTIC_ONLY",
                "failure_class": "NA" if converged else "NONCONVERGED",
                "failure_reason": "" if converged else "target-model convergence/residual gate failed",
            }
        )
    else:
        combined = result.stdout + "\n" + result.stderr
        base.update(
            {
                "engine_manifest": str(engine_manifest) if engine_manifest.exists() else "",
                "engine_manifest_sha256": sha256(engine_manifest) if engine_manifest.exists() else "",
                "target_model_convergence": False,
                "scientific_status": "FAILURE_MODE",
                "publication_status": "DIAGNOSTIC_ONLY",
                "failure_class": failure_class(combined),
                "failure_reason": combined[-6000:],
            }
        )
    output = args.out_dir / "endpoint_only_manifest.json"
    output.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"record_id": args.record_id, "scientific_status": base["scientific_status"], "failure_class": base["failure_class"]}, indent=2))


if __name__ == "__main__":
    main()
