#!/usr/bin/env python3
"""Submit MD only for selected families with completed training/model gates."""
import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--families", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    if not families or not set(families) <= set(cfg["families"]):
        raise SystemExit("unknown or empty family filter")
    runtimes = {cfg["families"][family]["runtime"] for family in families}
    if len(runtimes) != 1:
        raise SystemExit("all selected MD families must use one runtime")
    runtime = runtimes.pop()
    seeds = [int(value) for value in cfg["training"]["seeds"]]
    jobs = [(family, seed) for family in families for seed in seeds]
    missing = []
    for family_id, seed in jobs:
        family = cfg["families"][family_id]
        run_root = Path(cfg["remote_root"]) / family["output_dir_template"].format(seed=seed)
        manifest = run_root / "training_manifest.json"
        gate = run_root / "model_gate.json"
        if not manifest.is_file():
            missing.append(str(manifest))
        if not gate.is_file() or not json.loads(gate.read_text()).get("pass"):
            missing.append(str(gate))
    if missing:
        raise SystemExit("MD gate blocked; missing training/model-gate artifacts:\n" + "\n".join(missing))

    slurm = cfg["slurm"]
    export = {
        "CAMPAIGN_CONFIG": str(args.config.resolve()),
        "CAMPAIGN_FAMILIES": ",".join(families),
        "MACE_BOOTSTRAP": cfg["environment"]["bootstrap"],
        "MACE_CONDA_SH": cfg["environment"]["conda_sh"],
        "MACE_CONDA_ENV": cfg["environment"]["conda_env"],
        "MACE_POLAR_VENV": cfg["environment"]["polar_venv"],
        "MACE_RUNTIME": runtime,
    }
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in export.items())
    script = args.config.parent.parent / "slurm" / "md_array.slurm"
    cmd = [
        slurm["sbatch"], "--parsable", f"--account={slurm['account']}", f"--qos={slurm['qos']}",
        f"--partition={slurm['partition']}", f"--gres={slurm['gres']}",
        f"--cpus-per-task={slurm['cpus_per_task']}", f"--ntasks-per-node={slurm['ntasks_per_node']}",
        f"--time={slurm['time']}", f"--mem={slurm['mem']}", f"--array=0-{len(jobs) - 1}",
        f"--output={Path(cfg['remote_root']) / 'logs' / ('md-%A_%a-' + runtime + '.out')}",
        f"--error={Path(cfg['remote_root']) / 'logs' / ('md-%A_%a-' + runtime + '.err')}",
        f"--export={export_arg}", str(script),
    ]
    print(json.dumps({"jobs": jobs, "runtime": runtime, "command": cmd}, indent=2))
    if not args.submit:
        print("DRY_RUN: add --submit only after smoke and model-gate evidence is reviewed")
        return
    submit_env = os.environ.copy()
    submit_env["SLURM_CONF"] = slurm["slurm_conf"]
    job_id = subprocess.check_output(cmd, text=True, env=submit_env).strip()
    record = {
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "command": cmd,
        "config": str(args.config.resolve()),
        "config_sha256": config_sha256,
        "families": families,
        "runtime": runtime,
        "jobs": [{"family": family, "training_seed": seed} for family, seed in jobs],
    }
    out = Path(cfg["remote_root"]) / f"submission_md_{runtime}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite submission record: {out}")
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
