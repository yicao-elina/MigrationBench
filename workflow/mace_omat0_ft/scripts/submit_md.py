#!/usr/bin/env python3
"""Submit the six MD replicas only after all training artifacts exist."""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    seeds = [int(x) for x in cfg["training"]["seeds"]]
    jobs = [(family, seed) for family in cfg["families"] for seed in seeds]
    missing = []
    for family, seed in jobs:
        manifest = Path(cfg["remote_root"]) / "training" / family / ("seed%d" % seed) / "training_manifest.json"
        if not manifest.is_file():
            missing.append(str(manifest))
        gate = manifest.parent / "static_model_gate.json"
        if not gate.is_file() or not json.loads(gate.read_text()).get("pass"):
            missing.append(str(gate))
    if missing:
        raise SystemExit("MD gate blocked; missing training/static-gate artifacts:\n" + "\n".join(missing))
    slurm = cfg["slurm"]
    export = ["ALL", "CAMPAIGN_CONFIG=%s" % args.config.resolve(), "MACE_BOOTSTRAP=%s" % cfg["environment"]["bootstrap"], "MACE_CONDA_SH=%s" % cfg["environment"]["conda_sh"], "MACE_CONDA_ENV=%s" % cfg["environment"]["conda_env"]]
    script = args.config.parent.parent / "slurm" / "md_array.slurm"
    print("MD array plan:", json.dumps(jobs))
    print("MD script:", script)
    sbatch = slurm.get("sbatch", "sbatch")
    cmd = [sbatch, "--parsable", "--account=%s" % slurm["account"], "--qos=%s" % slurm["qos"], "--partition=%s" % slurm["partition"], "--gres=%s" % slurm["gres"], "--cpus-per-task=%s" % slurm["cpus_per_task"], "--ntasks-per-node=%s" % slurm["ntasks_per_node"], "--time=%s" % slurm["time"], "--mem=%s" % slurm["mem"], "--array=0-%d" % (len(jobs) - 1), "--export=ALL,CAMPAIGN_CONFIG=%s,MACE_BOOTSTRAP=%s,MACE_CONDA_SH=%s,MACE_CONDA_ENV=%s" % (args.config.resolve(), cfg["environment"]["bootstrap"], cfg["environment"]["conda_sh"], cfg["environment"]["conda_env"]), str(script)]
    print(" ".join(cmd))
    if not args.submit:
        print("DRY_RUN: training artifact gate passed; add --submit after smoke policy review")
        return
    submit_env = dict(__import__("os").environ)
    submit_env["SLURM_CONF"] = slurm.get("slurm_conf", "/cm/shared/apps/slurm/var/etc/slurm/slurm.conf")
    job_id = subprocess.check_output(cmd, universal_newlines=True, env=submit_env).strip()
    record = {"submitted_at_utc": datetime.now(timezone.utc).isoformat(), "job_id": job_id, "command": cmd, "config": str(args.config.resolve()), "jobs": [{"family": f, "training_seed": s} for f, s in jobs]}
    out = Path(cfg["remote_root"]) / "submission_md.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
