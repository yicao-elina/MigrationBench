#!/usr/bin/env python3
"""Capture the immutable formal array task map and terminal scheduler summary."""

from __future__ import annotations

import json
import os
import subprocess
import argparse
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    bookkeeping = json.loads((root / "EXECUTION_BOOKKEEPING_R02.json").read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default=str(bookkeeping["formal_array_job_id"]))
    parser.add_argument("--label", default="FORMAL")
    args = parser.parse_args()
    job_id = args.job_id
    env = os.environ.copy()
    env["SLURM_CONF"] = "/cm/shared/apps/slurm/var/etc/slurm/slurm.conf"
    env["PATH"] = "/cm/shared/apps/slurm/current/bin:" + env.get("PATH", "")
    command = [
        "sacct",
        "-X",
        "-j",
        str(job_id),
        "--parsable2",
        "--noheader",
        "--format=JobIDRaw,State,Elapsed,ExitCode,Start,End,NodeList",
    ]
    result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise SystemExit(result.stderr or "sacct failed")
    suffix = "" if args.label == "FORMAL" else f"_{args.label}"
    output = root / f"data/SCHEDULER_TASK_MAP{suffix}.tsv"
    output.parent.mkdir(exist_ok=True)
    output.write_text("JobIDRaw\tState\tElapsed\tExitCode\tStart\tEnd\tNodeList\n" + result.stdout)
    summary = {
        "schema_version": "scheduler_audit_v1",
        "campaign_id": bookkeeping["campaign_id"],
        "scheduler_job_id": str(job_id),
        "label": args.label,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(command),
        "stderr": result.stderr,
        "task_map": str(output),
    }
    audit_path = root / f"data/SCHEDULER_AUDIT{suffix}.json"
    audit_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

