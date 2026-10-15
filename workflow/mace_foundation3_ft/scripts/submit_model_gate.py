#!/usr/bin/env python3
"""Submit one static model gate, optionally behind production training."""
import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--family", required=True)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--afterok", default="", help="completed production job or array element")
    parser.add_argument("--output-root", type=Path, help="training root for this attempt")
    parser.add_argument("--attempt", default="", help="non-overwriting submission-record suffix")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.attempt and not re.fullmatch(r"[A-Za-z0-9_-]+", args.attempt):
        raise SystemExit("--attempt must contain only letters, numbers, underscore, or hyphen")
    cfg = yaml.safe_load(args.config.read_text())
    if args.family not in cfg["families"]:
        raise SystemExit(f"unknown family: {args.family}")
    family = cfg["families"][args.family]
    if family["runtime"] != "core":
        raise SystemExit("PolarMACE model gates require the separate approved Polar runtime")
    slurm = cfg["slurm"]
    remote_root = Path(cfg["remote_root"])
    output_root = (args.output_root or remote_root).resolve()
    suffix = f"_{args.attempt}" if args.attempt else ""
    record_path = remote_root / f"submission_model_gate_{args.family}{suffix}.json"
    if record_path.exists():
        raise SystemExit(f"refusing to overwrite model-gate submission record: {record_path}")
    export = ",".join(
        [
            "ALL",
            f"CAMPAIGN_CONFIG={args.config.resolve()}",
            f"MODEL_FAMILY={args.family}",
            f"TRAINING_SEED={args.training_seed}",
            f"MODEL_GATE_DEVICE={args.device}",
            f"MODEL_OUTPUT_ROOT={output_root}",
            f"MACE_BOOTSTRAP={cfg['environment']['bootstrap']}",
            f"MACE_CONDA_SH={cfg['environment']['conda_sh']}",
            f"MACE_CONDA_ENV={cfg['environment']['conda_env']}",
        ]
    )
    script = args.config.parent.parent / "slurm" / "model_gate.slurm"
    command = [
        slurm["sbatch"],
        "--parsable",
        f"--account={slurm['account']}",
        f"--qos={slurm['qos']}",
        f"--partition={slurm['partition']}",
        f"--gres={slurm['gres']}",
        f"--cpus-per-task={slurm['cpus_per_task']}",
        f"--ntasks-per-node={slurm['ntasks_per_node']}",
        f"--time={slurm['time']}",
        f"--mem={slurm['mem']}",
        f"--output={output_root / 'logs' / ('model-gate-' + args.family + '-%j.out')}",
        f"--error={output_root / 'logs' / ('model-gate-' + args.family + '-%j.err')}",
        f"--export={export}",
    ]
    if args.afterok:
        command.append(f"--dependency=afterok:{args.afterok}")
    command.append(str(script))
    print(" ".join(command))
    if not args.submit:
        print("DRY_RUN: add --submit only after production dependency review")
        return
    submit_env = dict(os.environ)
    submit_env["SLURM_CONF"] = slurm["slurm_conf"]
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    job_id = subprocess.check_output(command, universal_newlines=True, env=submit_env).strip()
    record = {
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "afterok": args.afterok or None,
        "family": args.family,
        "training_seed": args.training_seed,
        "device": args.device,
        "attempt": args.attempt or "primary",
        "output_root": str(output_root),
        "command": command,
        "config": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
    }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
