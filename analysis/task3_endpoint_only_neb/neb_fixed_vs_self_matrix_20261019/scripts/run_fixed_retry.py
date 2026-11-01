#!/usr/bin/env python3
"""Run selected fixed-path rows into an additive retry directory.

This wrapper preserves the canonical job table and runner while allowing a
targeted GPU retry for checkpoints that cannot be deserialized on a CPU node.
The retry directory is intentionally separate from ``remote_fixed_r01`` so a
retry can never overwrite the original failure evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def _row_at(path: Path, index: int) -> dict[str, str]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job-table", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--retry-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    row = _row_at(args.job_table, args.index)
    job = os.environ.get("SLURM_JOB_ID", f"LOCAL_{args.index}")
    out = args.retry_root / "runs" / "fixed" / row["model_id"] / row["path_id"] / f"attempt_{job}"
    out.mkdir(parents=True, exist_ok=True)
    python = os.environ.get("PYTHON", sys.executable)
    capture = [
        python,
        str(args.root / "code/capture_runtime_provenance.py"),
        "--output",
        str(out / "runtime_provenance.json"),
        "--binary",
        python,
        "--input",
        f"input={args.root / row['fixed_path_file']}",
        "--input",
        f"model={row['model_path']}",
        "--input",
        f"campaign_manifest={args.root / 'CAMPAIGN_MANIFEST_r01.json'}",
        "--input",
        f"runner={args.root / 'code/run_fixed_single_point.py'}",
        "--input",
        f"wrapper={Path(__file__).resolve()}",
    ]
    subprocess.run(capture, check=False)
    command = [
        python,
        str(args.root / "code/run_fixed_single_point.py"),
        "--input",
        str(args.root / row["fixed_path_file"]),
        "--output",
        str(out),
        "--model-id",
        row["model_id"],
        "--model-label",
        row["model_label"],
        "--model-path",
        row["model_path"],
        "--model-head",
        row["model_head"],
        "--device",
        args.device,
        "--record-id",
        row["record_id"],
        "--path-id",
        row["path_id"],
        "--path-class",
        row["path_class"],
        "--endpoint-hash-initial",
        row["initial_endpoint_sha256"],
        "--endpoint-hash-final",
        row["final_endpoint_sha256"],
        "--cell-hash",
        row["cell_sha256"],
        "--dft-reference-id",
        row["dft_reference_id"],
        "--dft-reference-status",
        row["dft_reference_status"],
    ]
    (out / "retry_command.json").write_text(json.dumps({"command": command}, indent=2) + "\n")
    subprocess.run(command, check=False)
    manifest = out / "fixed_single_point_manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        data["comparison_campaign"] = {
            "scope_tier": row["scope_tier"],
            "system_id": row["system_id"],
            "atoms": int(row["atoms"]),
            "model_family": row["model_family"],
            "training_seed": row["training_seed"],
            "seed_semantics": row["seed_semantics"],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", f"LOCAL_{args.index}"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", str(args.index)),
            "retry_directory": str(args.retry_root),
        }
        manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
