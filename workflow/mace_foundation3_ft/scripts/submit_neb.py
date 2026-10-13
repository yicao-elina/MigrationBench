#!/usr/bin/env python3
"""Plan or submit the NEB array after the endpoint/path gate passes."""
from __future__ import annotations

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
    parser.add_argument("--neb-manifest", required=True, type=Path)
    parser.add_argument("--families", required=True, help="comma-separated accepted model families")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--afterok")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if not config["neb"].get("enabled", False):
        raise SystemExit("NEB submission blocked: set neb.enabled=true only after manifest review")
    manifest = yaml.safe_load(args.neb_manifest.read_text())
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    if not families or not set(families) <= set(config["families"]):
        raise SystemExit("NEB submission blocked: invalid family filter")
    if manifest.get("status") not in {"approved", "ready"}:
        raise SystemExit("NEB submission blocked: manifest status must be approved or ready")
    modes = [mode for mode in config["neb"]["modes"] if manifest["protocol"].get(mode, {}).get("enabled", False)]
    seeds = [int(x) for x in config["training"]["seeds"]]
    jobs = [{"family": family, "training_seed": seed, "mode": mode} for family in families for seed in seeds for mode in modes]
    if not jobs:
        raise SystemExit("NEB submission blocked: no enabled modes")
    if "self_relaxed_mlff_neb" in modes:
        runner = Path(config["neb"].get("runner_script", "src/migrationbench/run_mlff_neb.py"))
        if not runner.is_absolute():
            runner = args.config.resolve().parents[3] / runner
        if not runner.is_file():
            raise SystemExit(
                "NEB submission blocked: configured MLFF-NEB runner is missing: "
                f"{runner}. Deploy the canonical repository runner and its "
                "export_mlff_neb_iteration_history.py dependency first."
            )
    remote_root = Path(config["remote_root"])
    matrix = remote_root / "neb_job_matrix.json"
    if matrix.exists():
        raise SystemExit(f"Refusing to replace existing NEB job matrix: {matrix}")
    matrix.parent.mkdir(parents=True, exist_ok=True)
    matrix.write_text(json.dumps({"created_at_utc": datetime.now(timezone.utc).isoformat(), "config": str(args.config.resolve()), "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(), "neb_manifest": str(args.neb_manifest.resolve()), "neb_manifest_sha256": hashlib.sha256(args.neb_manifest.read_bytes()).hexdigest(), "jobs": jobs}, indent=2, sort_keys=True) + "\n")
    slurm = config["slurm"]
    export = "ALL,CAMPAIGN_CONFIG=%s,NEB_MANIFEST=%s,NEB_FAMILIES=%s,MACE_BOOTSTRAP=%s,MACE_CONDA_SH=%s,MACE_CONDA_ENV=%s" % (args.config.resolve(), args.neb_manifest.resolve(), ",".join(families), config["environment"]["bootstrap"], config["environment"]["conda_sh"], config["environment"]["conda_env"])
    script = args.config.parent.parent / "slurm" / "neb_array.slurm"
    command = [slurm["sbatch"], "--parsable", f"--account={slurm['account']}", f"--qos={slurm['qos']}", f"--partition={slurm['partition']}", f"--gres={slurm['gres']}", f"--cpus-per-task={slurm['cpus_per_task']}", f"--ntasks-per-node={slurm['ntasks_per_node']}", f"--time={slurm['time']}", f"--mem={slurm['mem']}", f"--array=0-{len(jobs)-1}", f"--export={export}"]
    if args.afterok:
        command.append(f"--dependency=afterok:{args.afterok}")
    command.append(str(script))
    print(" ".join(command))
    if not args.submit:
        print("DRY_RUN: no NEB job submitted")
        return
    env = dict(os.environ)
    env["SLURM_CONF"] = slurm.get("slurm_conf", "/cm/shared/apps/slurm/var/etc/slurm/slurm.conf")
    job_id = subprocess.check_output(command, text=True, env=env).strip()
    record = {"submitted_at_utc": datetime.now(timezone.utc).isoformat(), "job_id": job_id, "command": command, "jobs": jobs, "neb_manifest": str(args.neb_manifest.resolve())}
    submission_record = remote_root / ("submission_neb_%s.json" % "_".join(families))
    if submission_record.exists():
        raise SystemExit(f"Refusing to replace existing NEB submission record: {submission_record}")
    submission_record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
