#!/usr/bin/env python3
"""Dispatch one endpoint-only self-relaxed row from a registry-backed table."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def row_at(path: Path, index: int) -> dict:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-table", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    row = row_at(args.job_table, args.index)
    job = os.environ.get("SLURM_JOB_ID", f"LOCAL_{args.index}")
    out = args.root / "runs" / row["initialization_method"].lower() / row["model_id"] / row["path_id"] / f"attempt_{job}"
    out.mkdir(parents=True, exist_ok=True)
    capture = [
        sys.executable, str(args.root / "code/capture_runtime_provenance.py"),
        "--output", str(out / "runtime_provenance.json"), "--binary", sys.executable,
        "--input", f"initial={args.root / row['initial_endpoint_file']}",
        "--input", f"final={args.root / row['final_endpoint_file']}",
        "--input", f"model={row['model_path']}",
        "--input", f"campaign_manifest={args.root / 'CAMPAIGN_MANIFEST_r01.json'}",
        "--input", f"runner={args.root / 'code/run_mlff_neb.py'}",
        "--input", f"wrapper={args.root / 'code/run_endpoint_only_neb.py'}",
    ]
    subprocess.run(capture, check=False)
    command = [
        sys.executable, str(args.root / "code/run_endpoint_only_neb.py"),
        "--runner", str(args.root / "code/run_mlff_neb.py"),
        "--initial", str(args.root / row["initial_endpoint_file"]),
        "--final", str(args.root / row["final_endpoint_file"]),
        "--model-path", row["model_path"], "--model-id", row["model_id"],
        "--model-label", row["model_label"], "--model-head", row["model_head"],
        "--out-dir", str(out), "--record-id", row["record_id"],
        "--path-id", row["path_id"], "--path-class", row["path_class"],
        "--initialization-method", row["initialization_method"],
        "--endpoint-hash-initial", row["initial_endpoint_sha256"],
        "--endpoint-hash-final", row["final_endpoint_sha256"],
        "--cell-hash", row["cell_sha256"], "--dft-reference-id", row["dft_reference_id"],
        "--dft-reference-status", row["dft_reference_status"],
        "--campaign-manifest", str(args.root / "CAMPAIGN_MANIFEST_r01.json"),
        "--neb-seed", row["neb_seed"], "--device", "cpu",
    ]
    subprocess.run(command, check=False)
    manifest = out / "endpoint_only_manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        data["comparison_campaign"] = {
            "scope_tier": row["scope_tier"],
            "system_id": row["system_id"],
            "atoms": int(row["atoms"]),
            "model_family": row["model_family"],
            "training_seed": row["training_seed"],
            "seed_semantics": row["seed_semantics"],
            "perturbation_seed": row["perturbation_seed"],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", f"LOCAL_{args.index}"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", str(args.index)),
        }
        manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
