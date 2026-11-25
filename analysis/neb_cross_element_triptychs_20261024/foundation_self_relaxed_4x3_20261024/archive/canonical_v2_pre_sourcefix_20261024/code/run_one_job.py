#!/usr/bin/env python3
"""Run one endpoint-only, IDPP-initialized Foundation self-relaxed NEB row."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
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
    if any(token in lowered for token in ("deserial", "attributeerror", "torch.jit", "unpickl", "unknown class", "load_fail")):
        return "MODEL_LOAD_OR_DESERIALIZATION"
    if any(token in lowered for token in ("nan", "inf", "nonfinite", "non-finite")):
        return "NONFINITE"
    if "migrant element" in lowered or "chemical symbol" in lowered:
        return "ELEMENT_IDENTITY"
    return "RUNNER_EXCEPTION"


def row_at(path: Path, index: int) -> dict[str, str]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return rows[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-table", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    row = row_at(args.job_table, args.index)
    scheduler_job = os.environ.get("SLURM_JOB_ID", f"LOCAL_{args.index}")
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID", str(args.index))
    out = args.root / "runs" / row["initialization_method"].lower() / row["model_id"] / row["path_id"] / f"attempt_{scheduler_job}_{array_task}"
    out.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    initial = args.root / row["initial_endpoint_file"]
    final = args.root / row["final_endpoint_file"]
    runner = args.root / "code/run_mlff_neb.py"
    campaign_manifest = args.root / "CAMPAIGN_MANIFEST.json"
    wrapper_sha = sha256(Path(__file__).resolve())
    runner_sha = sha256(runner)
    model_path = Path(row["model_path"])

    provenance = args.root / "code/capture_runtime_provenance.py"
    if provenance.is_file():
        capture = [
            sys.executable,
            str(provenance),
            "--output",
            str(out / "runtime_provenance.json"),
            "--binary",
            sys.executable,
            "--input",
            f"initial={initial}",
            "--input",
            f"final={final}",
            "--input",
            f"model={model_path}",
            "--input",
            f"campaign_manifest={campaign_manifest}",
            "--input",
            f"runner={runner}",
            "--input",
            f"wrapper={Path(__file__).resolve()}",
        ]
        subprocess.run(capture, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    command = [
        sys.executable,
        str(runner),
        "--initial",
        str(initial),
        "--final",
        str(final),
        "--n-images",
        row["n_images"],
        "--out-dir",
        str(out),
        "--calculator",
        "mace-model",
        "--model-path",
        row["model_path"],
        "--model-head",
        row["model_head"],
        "--device",
        "cpu",
        "--default-dtype",
        "float64",
        "--steps",
        row["steps_max"],
        "--seed",
        row["neb_seed"],
        "--fmax",
        row["fmax_target_eV_A"],
        "--spring-constant",
        "0.1",
        "--fire-dt",
        "0.1",
        "--fire-maxstep",
        "0.2",
        "--migrant-element",
        row["element"],
        "--run-role",
        "production_preconditioner",
        "--interpolate-method",
        "idpp",
        "--source-candidate-manifest",
        str(campaign_manifest),
    ]
    (out / "endpoint_only_command.json").write_text(
        json.dumps(
            {
                "command": command,
                "endpoint_inputs_only": True,
                "intermediate_inputs_used": [],
                "created_at_utc": created,
                "migrant_element": row["element"],
            },
            indent=2,
        )
        + "\n"
    )
    result = subprocess.run(command, cwd=runner.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    (out / "runner_stdout.txt").write_text(result.stdout)
    (out / "runner_stderr.txt").write_text(result.stderr)

    base = {
        "schema_version": "foundation_self_relaxed_endpoint_only_v1",
        "campaign_id": row["campaign_id"],
        "record_id": row["record_id"],
        "observable": row["observable"],
        "initialization_method": row["initialization_method"],
        "initialization_source": row["initialization_source"],
        "endpoint_policy": "FIXED_ENDPOINTS",
        "leakage_gate": "PASS",
        "intermediate_inputs_used": [],
        "element": row["element"],
        "atoms": int(row["atoms"]),
        "path_id": row["path_id"],
        "path_class": row["path_class"],
        "path_label": row["path_label"],
        "legacy_source_stem": row["legacy_source_stem"],
        "initial_endpoint_path": str(initial),
        "initial_endpoint_sha256": sha256(initial),
        "final_endpoint_path": str(final),
        "final_endpoint_sha256": sha256(final),
        "endpoint_hash_initial_registry": row["initial_endpoint_sha256"],
        "endpoint_hash_final_registry": row["final_endpoint_sha256"],
        "cell_sha256": row["cell_sha256"],
        "n_images_target": int(row["n_images"]),
        "dft_reference_id": row["dft_reference_id"],
        "dft_reference_status": row["dft_reference_status"],
        "model_id": row["model_id"],
        "model_label": row["model_label"],
        "model_family": row["model_family"],
        "model_path": row["model_path"],
        "model_sha256_registry": row["model_sha256"],
        "model_sha256_observed": sha256(model_path) if model_path.is_file() else "",
        "model_head": row["model_head"],
        "neb_seed": int(row["neb_seed"]),
        "runner_path": str(runner),
        "runner_sha256": runner_sha,
        "wrapper_path": str(Path(__file__).resolve()),
        "wrapper_sha256": wrapper_sha,
        "campaign_manifest": str(campaign_manifest),
        "campaign_manifest_sha256": sha256(campaign_manifest),
        "created_at_utc": created,
        "slurm_job_id": scheduler_job,
        "slurm_array_task_id": array_task,
        "return_code": result.returncode,
        "execution_role": row.get("execution_role", "FORMAL_ARRAY"),
        "manifest_role": row.get("manifest_role", "FORMAL_LAUNCH"),
    }
    engine_manifest = out / "mlff_neb_manifest.json"
    if result.returncode == 0 and engine_manifest.is_file():
        try:
            engine = json.loads(engine_manifest.read_text())
        except Exception as exc:
            engine = None
            base["engine_manifest_parse_error"] = f"{type(exc).__name__}: {exc}"
        if engine is not None:
            residual = engine.get("final_max_neb_residual_eV_A")
            neb_force = engine.get("final_max_internal_neb_force_eV_A")
            converged = bool(engine.get("optimizer_converged")) and all(
                value is not None and float(value) <= float(row["fmax_target_eV_A"])
                for value in (residual, neb_force)
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
                "failure_reason": combined[-8000:],
            }
        )
    output = out / "endpoint_only_manifest.json"
    output.write_text(json.dumps(base, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"record_id": row["record_id"], "scientific_status": base["scientific_status"], "failure_class": base["failure_class"]}, indent=2))


if __name__ == "__main__":
    main()

