#!/usr/bin/env python3
"""Continue a nonconverged endpoint-only NEB from its last image path.

The extension is additive: it reads the prior endpoint manifest and
``mlff_neb_images.extxyz`` but writes a new attempt directory.  The resulting
manifest keeps the prior status as diagnostic evidence and never upgrades a
nonconverged run to primary evidence without the normal residual gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-images", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    python = os.environ.get("PYTHON", "python")
    command = [
        python,
        str(args.runner),
        "--images", str(args.source_images),
        "--n-images", "10",
        "--out-dir", str(args.out_dir),
        "--calculator", "mace-model",
        "--model-path", source["model_path"],
        "--model-head", source.get("model_head", "default"),
        "--device", args.device,
        "--steps", str(args.steps),
        "--seed", str(source.get("neb_seed", 42)),
        "--fmax", str(args.fmax),
        "--spring-constant", "0.1",
        "--fire-dt", "0.1",
        "--fire-maxstep", "0.2",
        "--migrant-element", "Cr",
        "--run-role", "production_preconditioner",
        "--source-candidate-manifest", str(args.campaign_manifest),
    ]
    (args.out_dir / "extension_command.json").write_text(json.dumps({
        "command": command,
        "parent_manifest": str(args.source_manifest),
        "parent_images": str(args.source_images),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    (args.out_dir / "runner_stdout.txt").write_text(result.stdout)
    (args.out_dir / "runner_stderr.txt").write_text(result.stderr)
    engine_path = args.out_dir / "mlff_neb_manifest.json"
    out = dict(source)
    out.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine_manifest": str(engine_path) if engine_path.is_file() else "",
        "engine_manifest_sha256": sha256(engine_path) if engine_path.is_file() else "",
        "extension_parent_manifest": str(args.source_manifest),
        "extension_parent_images": str(args.source_images),
        "extension_parent_status": source.get("scientific_status"),
        "extension_steps": args.steps,
        "extension_fmax_eV_A": args.fmax,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "LOCAL_EXTENSION"),
    })
    if result.returncode == 0 and engine_path.is_file():
        engine = json.loads(engine_path.read_text())
        residual = engine.get("final_max_neb_residual_eV_A")
        neb_force = engine.get("final_max_internal_neb_force_eV_A")
        converged = bool(engine.get("optimizer_converged")) and all(
            value is not None and float(value) <= args.fmax for value in (residual, neb_force)
        )
        out.update({
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
            "failure_reason": "" if converged else "extension residual gate failed",
        })
    else:
        out.update({
            "target_model_convergence": False,
            "scientific_status": "FAILURE_MODE",
            "publication_status": "DIAGNOSTIC_ONLY",
            "failure_class": "RUNNER_EXCEPTION",
            "failure_reason": (result.stdout + "\n" + result.stderr)[-6000:],
        })
    (args.out_dir / "endpoint_extension_manifest.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"record_id": out.get("record_id"), "scientific_status": out["scientific_status"], "forward_barrier_eV": out.get("forward_barrier_eV")}, indent=2))


if __name__ == "__main__":
    main()
