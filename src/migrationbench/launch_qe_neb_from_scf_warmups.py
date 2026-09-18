#!/usr/bin/env python3
"""Gate, assemble, and optionally submit a production QE NEB from image SCFs."""

import argparse
import fcntl
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from expand_qe_scf_warmup_batch import atomic_json, find_remote_job, run, sha256_file
from prepare_qe_neb_from_scf_warmups import select_warmup_jobs, walltime_seconds


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args()
    defaults = {}
    config_hash = None
    if pre_args.config:
        defaults = json.loads(pre_args.config.read_text()).get("arguments", {})
        config_hash = sha256_file(pre_args.config)

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre_parser])
    parser.set_defaults(**defaults)
    parser.add_argument("--base-neb-input")
    parser.add_argument("--warmup-status")
    parser.add_argument("--out-dir")
    parser.add_argument("--launch-manifest")
    parser.add_argument("--active-qe-jobs")
    parser.add_argument("--path-id")
    parser.add_argument("--branch-id")
    parser.add_argument("--job-name")
    parser.add_argument("--num-images", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--remote-input-dir")
    parser.add_argument("--remote-code-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_pipeline")
    parser.add_argument("--remote-run-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_runs")
    parser.add_argument("--remote-wrapper", default="/scratch16/pclancy3/yi/revision1_migrationbench_pipeline/scripts/migrationbench/submit_slurm_qe_neb_warmstarted.sh")
    parser.add_argument("--assembler", default=str(Path(__file__).with_name("prepare_qe_neb_from_scf_warmups.py")))
    parser.add_argument("--ssh-host", default="rockfish")
    parser.add_argument("--walltime", default="48:00:00")
    parser.add_argument("--max-seconds", type=int, default=171000)
    parser.add_argument("--ntasks", type=int, default=4)
    parser.add_argument("--memory", default="180G")
    parser.add_argument("--kpoint-pools", type=int, default=4)
    parser.add_argument("--production-degauss-Ry", type=float, default=0.005)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    required = [
        "base_neb_input", "warmup_status", "out_dir", "launch_manifest", "active_qe_jobs",
        "path_id", "branch_id", "job_name", "num_images", "remote_input_dir",
    ]
    missing = [name for name in required if getattr(args, name) in (None, "")]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    for name in ("base_neb_input", "warmup_status", "out_dir", "launch_manifest", "active_qe_jobs", "assembler"):
        setattr(args, name, Path(getattr(args, name)))
    args.config_sha256 = config_hash
    return args


def gate(status, path_id, num_images):
    jobs = select_warmup_jobs(status, path_id, num_images)
    rejected = {
        index: row.get("classification")
        for index, row in jobs.items()
        if row.get("classification") != "accepted_scf_warmstart"
    }
    if rejected:
        raise RuntimeError(f"production NEB warmup gate failed: {rejected}")
    return jobs


def ensure_assembled(args, selected):
    neb_input = args.out_dir / "neb.in"
    warmup_manifest = args.out_dir / "warmstarted_neb_manifest.json"
    if neb_input.is_file() and warmup_manifest.is_file():
        manifest = json.loads(warmup_manifest.read_text())
        expected_jobs = {index: str(row["job_id"]) for index, row in selected.items()}
        observed_jobs = {int(row["image_index_qe"]): str(row["warmup_job_id"]) for row in manifest["warmup_sources"]}
        checks = {
            "path_id": manifest.get("path_id") == args.path_id,
            "seed": manifest.get("seed") == args.seed,
            "base_hash": manifest.get("base_neb_input_sha256") == sha256_file(args.base_neb_input),
            "input_hash": manifest.get("generated_neb_input_sha256") == sha256_file(neb_input),
            "selected_jobs": observed_jobs == expected_jobs,
        }
        if not all(checks.values()):
            raise RuntimeError(f"existing assembled NEB conflicts with current accepted warmups: {checks}")
        return neb_input, warmup_manifest
    if neb_input.exists() or warmup_manifest.exists():
        raise RuntimeError(f"partial assembled NEB artifacts in {args.out_dir}")

    command = [
        sys.executable, str(args.assembler),
        "--base-neb-input", str(args.base_neb_input),
        "--warmup-status", str(args.warmup_status),
        "--out-dir", str(args.out_dir),
        "--path-id", args.path_id,
        "--num-images", str(args.num_images),
        "--seed", str(args.seed),
        "--walltime", args.walltime,
        "--max-seconds", str(args.max_seconds),
        "--ntasks", str(args.ntasks),
        "--memory", args.memory,
        "--kpoint-pools", str(args.kpoint_pools),
        "--production-degauss-Ry", str(args.production_degauss_Ry),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError(f"assembler failed\n{result.stderr}")
    return neb_input, warmup_manifest


def main():
    args = parse_args()
    if args.max_seconds >= walltime_seconds(args.walltime):
        raise ValueError("max_seconds must be below the Slurm walltime")
    status = json.loads(args.warmup_status.read_text())
    selected = gate(status, args.path_id, args.num_images)

    lock_path = args.active_qe_jobs.with_name(args.active_qe_jobs.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        neb_input, warmup_manifest = ensure_assembled(args, selected)
        launch = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_role": "production_qe_neb_launch_from_accepted_image_scf_charge_densities",
            "path_id": args.path_id,
            "branch_id": args.branch_id,
            "job_name": args.job_name,
            "seed": args.seed,
            "canonical_config": str(args.config.resolve()) if args.config else None,
            "canonical_config_sha256": args.config_sha256,
            "neb_input": str(neb_input.resolve()),
            "neb_input_sha256": sha256_file(neb_input),
            "warmstarted_neb_manifest": str(warmup_manifest.resolve()),
            "warmstarted_neb_manifest_sha256": sha256_file(warmup_manifest),
            "selected_warmup_jobs": [
                {"image_index_qe": index, "job_id": str(row["job_id"]), "attempt": row.get("attempt", 1)}
                for index, row in sorted(selected.items())
            ],
            "remote_input_dir": args.remote_input_dir,
            "remote_run_root": args.remote_run_root,
            "resources": {
                "walltime": args.walltime, "max_seconds": args.max_seconds, "ntasks": args.ntasks,
                "memory": args.memory, "kpoint_pools": args.kpoint_pools,
            },
            "label_policy": {"warmup_energies_are_labels": False, "production_reconvergence_required": True},
        }
        if args.launch_manifest.is_file():
            existing = json.loads(args.launch_manifest.read_text())
            immutable_keys = ["path_id", "branch_id", "job_name", "seed", "neb_input_sha256", "warmstarted_neb_manifest_sha256", "selected_warmup_jobs", "resources"]
            checks = {key: existing.get(key) == launch.get(key) for key in immutable_keys}
            if not all(checks.values()):
                raise RuntimeError(f"existing launch manifest conflicts with requested production NEB: {checks}")
            launch = existing
        else:
            atomic_json(args.launch_manifest, launch)

        if args.submit:
            registry = json.loads(args.active_qe_jobs.read_text())
            registered = next((row for row in registry["jobs"] if row["job_name"] == args.job_name), None)
            job_id = str(registered["job_id"]) if registered else find_remote_job(args.ssh_host, args.job_name)
            submission_state = "already_registered" if registered else "recovered_from_rockfish" if job_id else "submitted"
            if not job_id:
                run(["ssh", args.ssh_host, "mkdir", "-p", args.remote_input_dir])
                run(["rsync", "-a", str(args.out_dir) + "/", f"{args.ssh_host}:{args.remote_input_dir}/"])
                run(["rsync", "-a", str(args.launch_manifest), f"{args.ssh_host}:{args.remote_input_dir}/"])
                export = ",".join([
                    "ALL",
                    f"MIGRATIONBENCH_NEB_INPUT={args.remote_input_dir}/neb.in",
                    f"MIGRATIONBENCH_NEB_MANIFEST={args.remote_input_dir}/warmstarted_neb_manifest.json",
                    f"MIGRATIONBENCH_NEB_LAUNCH_MANIFEST={args.remote_input_dir}/production_launch_manifest.json",
                    f"MIGRATIONBENCH_CODE_ROOT={args.remote_code_root}",
                    f"MIGRATIONBENCH_RUN_ROOT={args.remote_run_root}",
                    f"MIGRATIONBENCH_KPOINT_POOLS={args.kpoint_pools}",
                ])
                output = run(["ssh", args.ssh_host, "sbatch", "--parsable", "--job-name", args.job_name, "--export", export, args.remote_wrapper])
                job_id = output.split(";", 1)[0].strip()
                if not re.fullmatch(r"\d+", job_id):
                    raise RuntimeError(f"could not parse Slurm job id from: {output!r}")
            if not registered:
                registry["jobs"].append({
                    "branch_id": args.branch_id,
                    "job_id": job_id,
                    "job_name": args.job_name,
                    "path_id": args.path_id,
                    "remote_run_dir": f"{args.remote_run_root.rstrip('/')}/{args.job_name}_{job_id}",
                })
                registry["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                atomic_json(args.active_qe_jobs, registry)
            if str(launch.get("submission", {}).get("job_id", "")) != job_id:
                launch["submission"] = {"state": submission_state, "job_id": job_id}
                atomic_json(args.launch_manifest, launch)
                run(["rsync", "-a", str(args.launch_manifest), f"{args.ssh_host}:{args.remote_input_dir}/"])

    print(json.dumps(launch, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
