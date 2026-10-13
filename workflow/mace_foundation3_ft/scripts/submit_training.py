#!/usr/bin/env python3
"""Submit a filtered smoke or production training array after local validation."""
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--families", required=True, help="comma-separated family IDs")
    parser.add_argument("--runtime", required=True, choices=["core", "polar"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--afterok", help="Slurm job or array element required to pass")
    parser.add_argument("--submit", action="store_true", help="actually call sbatch; default is dry-run")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    if not families or not set(families) <= set(cfg["families"]):
        raise SystemExit("unknown or empty family filter")
    if any(cfg["families"][family]["runtime"] != args.runtime for family in families):
        raise SystemExit("family/runtime filter mismatch")
    seeds = [int(value) for value in cfg["training"]["seeds"]]
    jobs = [{"family": family, "training_seed": seed} for family in families for seed in seeds]
    slurm = cfg["slurm"]
    env = {
        "CAMPAIGN_CONFIG": str(args.config.resolve()),
        "CAMPAIGN_FAMILIES": ",".join(families),
        "MACE_BOOTSTRAP": cfg["environment"]["bootstrap"],
        "MACE_CONDA_SH": cfg["environment"]["conda_sh"],
        "MACE_CONDA_ENV": cfg["environment"]["conda_env"],
        "MACE_POLAR_VENV": cfg["environment"]["polar_venv"],
        "MACE_RUNTIME": args.runtime,
        "CAMPAIGN_OUTPUT_ROOT": str(Path(cfg["remote_root"])),
        "TRAIN_SMOKE": "1" if args.smoke else "0",
    }
    export = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
    script = args.config.parent.parent / "slurm" / "train_array.slurm"
    array_max = len(jobs) - 1
    cmd = [
        slurm["sbatch"], "--parsable", f"--account={slurm['account']}", f"--qos={slurm['qos']}",
        f"--partition={slurm['partition']}", f"--gres={slurm['gres']}",
        f"--cpus-per-task={slurm['cpus_per_task']}", f"--ntasks-per-node={slurm['ntasks_per_node']}",
        f"--time={slurm['time']}", f"--mem={slurm['mem']}", f"--array=0-{array_max}",
        f"--output={Path(cfg['remote_root']) / 'logs' / ('train-%A_%a-' + args.runtime + '.out')}",
        f"--error={Path(cfg['remote_root']) / 'logs' / ('train-%A_%a-' + args.runtime + '.err')}",
        f"--export={export}", str(script),
    ]
    if args.afterok:
        cmd.insert(-1, f"--dependency=afterok:{args.afterok}")
    print(" ".join(cmd))
    if not args.submit:
        print("DRY_RUN: add --submit only after the corresponding smoke/preflight policy passes")
        return
    submit_env = os.environ.copy()
    submit_env["SLURM_CONF"] = slurm["slurm_conf"]
    job_id = subprocess.check_output(cmd, text=True, env=submit_env).strip()
    phase = "smoke" if args.smoke else "production"
    record = {
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "command": cmd,
        "config": str(args.config.resolve()),
        "runtime": args.runtime,
        "phase": phase,
        "afterok": args.afterok,
        "families": families,
        "jobs": jobs,
    }
    family_tag = families[0] if len(families) == 1 else "all"
    out = Path(cfg["remote_root"]) / f"submission_training_{args.runtime}_{phase}_{family_tag}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite submission record: {out}")
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
