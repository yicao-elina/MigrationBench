#!/usr/bin/env python3
"""Submit one bounded campaign array and append its identity to RUN_LOG.tsv."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, read_tsv, write_tsv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["a0", "fixed", "dense"], required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config, root = load_config(args.config)
    if args.stage == "a0":
        table = root / "jobs/a0_fixed_jobs.tsv"
        script = root / "slurm/run_fixed_array.slurm"
        array = "0-0%1" if args.smoke else "0-7%4"
    elif args.stage == "fixed":
        table = root / "jobs/fixed_jobs.tsv"
        script = root / "slurm/run_fixed_array.slurm"
        array = "0-0%1" if args.smoke else "0-29%4"
        gate = root / "manifests/STAGE1_GATE.json"
        if not args.smoke and gate.is_file():
            import json
            data = json.loads(gate.read_text())
            if data.get("a0_status") != "PASS":
                raise SystemExit("fixed submission blocked: A0 gate is not PASS")
    else:
        table = root / "jobs/dense_jobs.tsv"
        script = root / "slurm/run_dense_array.slurm"
        array = "0-0%1" if args.smoke else "0-29%4"
        gate = root / "manifests/STAGE1_GATE.json"
        if not args.smoke and gate.is_file():
            import json
            data = json.loads(gate.read_text())
            if data.get("fixed_status") != "PASS":
                raise SystemExit("dense submission blocked: Tier 1 fixed gate is not PASS")
    if not table.is_file():
        raise SystemExit(f"missing job table: {table}")
    count = len(read_tsv(table))
    if not args.smoke:
        end = count - 1 if args.end is None else args.end
        if args.start < 0 or end < args.start or end >= count:
            raise SystemExit(f"invalid array bounds {args.start}-{end} for {count} rows")
        array = f"{args.start}-{end}%{config['environment']['array_concurrency']}"
    environment = config["environment"]
    sbatch = str(Path(environment["slurm_bin_dir"]) / "sbatch")
    export_values = ",".join([
        "ALL",
        f"CAMPAIGN_ROOT={config['paths']['campaign_root_remote']}",
        f"MACE_PYTHON={environment['python']}",
        f"JOB_TABLE={config['paths']['campaign_root_remote']}/{table.relative_to(root)}",
        f"SLURM_CONF={environment['slurm_conf']}",
        f"SLURM_BIN_DIR={environment['slurm_bin_dir']}",
    ])
    command = [
        sbatch,
        "--parsable",
        "--partition", environment["partition"],
        "--account", environment["account"],
        "--gres", environment["gres"],
        "--cpus-per-task", str(environment["cpus_per_task"]),
        "--mem", environment["memory"],
        "--time", environment["time"],
        "--array", array,
        "--chdir", config["paths"]["campaign_root_remote"],
        "--export", export_values,
        str(script),
    ]
    print(" ".join(command))
    if args.dry_run:
        return
    env = os.environ.copy()
    env["SLURM_CONF"] = environment["slurm_conf"]
    env["PATH"] = f"{environment['slurm_bin_dir']}:{env.get('PATH', '')}"
    result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    job_id = result.stdout.strip()
    run_log = root / "manifests/RUN_LOG.tsv"
    rows = []
    if run_log.is_file():
        with run_log.open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    rows.append({
        "stage": args.stage,
        "job_id": job_id,
        "array": array,
        "expected_rows": "1" if args.smoke else str(count),
        "job_table": str(table),
        "slurm_script": str(script),
        "command": " ".join(command),
    })
    write_tsv(run_log, rows, ["stage", "job_id", "array", "expected_rows", "job_table", "slurm_script", "command"])
    print(f"submitted {args.stage}: {job_id}")


if __name__ == "__main__":
    main()

