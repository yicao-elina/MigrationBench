#!/usr/bin/env python3
"""Submit or print the six-job OMAT-0 training array after local config validation."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--submit", action="store_true", help="actually call sbatch; default is dry-run")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    seeds = cfg["training"]["seeds"]
    array_max = len(cfg["families"]) * len(seeds) - 1
    slurm = cfg["slurm"]
    env = {
        "CAMPAIGN_CONFIG": str(args.config.resolve()),
        "MACE_BOOTSTRAP": cfg["environment"]["bootstrap"],
        "MACE_CONDA_SH": cfg["environment"]["conda_sh"],
        "MACE_CONDA_ENV": cfg["environment"]["conda_env"],
    }
    export = "ALL," + ",".join(f"{k}={v}" for k, v in env.items())
    script = args.config.parent.parent / "slurm" / "train_array.slurm"
    cmd = [
        "sbatch", "--parsable", f"--account={slurm['account']}", f"--qos={slurm['qos']}",
        f"--partition={slurm['partition']}", f"--gres={slurm['gres']}",
        f"--cpus-per-task={slurm['cpus_per_task']}", f"--ntasks-per-node={slurm['ntasks_per_node']}",
        f"--time={slurm['time']}", f"--mem={slurm['mem']}", f"--array=0-{array_max}",
        f"--export={export}", str(script),
    ]
    print(" ".join(cmd))
    if not args.submit:
        print("DRY_RUN: add --submit only after remote preflight passes")
        return
    job_id = subprocess.check_output(cmd, text=True).strip()
    record = {"submitted_at_utc": datetime.now(timezone.utc).isoformat(), "job_id": job_id, "command": cmd, "config": str(args.config.resolve()), "jobs": [{"family": f, "training_seed": s} for f in cfg["families"] for s in seeds]}
    out = Path(cfg["remote_root"]) / "submission_training.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
