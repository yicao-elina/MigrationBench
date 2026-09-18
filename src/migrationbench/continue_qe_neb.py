#!/usr/bin/env python3
"""Prepare and optionally submit one validated, provenance-complete QE NEB continuation."""

import argparse
import fcntl
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from expand_qe_scf_warmup_batch import atomic_json, find_remote_job, run, sha256_file
from prepare_qe_neb_from_scf_warmups import walltime_seconds
from prepare_qe_scf_warmup import set_namelist_value


def select_parent(status, job_id):
    matches = [row for row in status["jobs"] if str(row["job_id"]) == str(job_id)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one parent row for job {job_id}, found {len(matches)}")
    row = matches[0]
    if row.get("classification") != "clean_max_seconds_restartable":
        raise RuntimeError(f"parent {job_id} is not restartable: {row.get('classification')}")
    restart = row.get("restart_artifacts", {})
    if not restart.get("restart_ready") or restart.get("latest_path_iteration") is None:
        raise RuntimeError(f"parent {job_id} lacks a complete QE restart state")
    return row


def derive_name(parent_name, attempt, seed):
    suffix = f"_r{attempt}_s{seed}"
    if re.search(r"_r\d+_s\d+$", parent_name):
        return re.sub(r"_r\d+_s\d+$", suffix, parent_name)
    return parent_name + suffix


def prepare(args, parent):
    local_dir = Path(parent["local_dir"])
    source_input = local_dir / "neb.in"
    source_output = local_dir / "neb.out"
    latest_iteration = int(parent["restart_artifacts"]["latest_path_iteration"])
    prefix = parent.get("input_limits", {}).get("prefix") or "sb2te3"
    source_path = local_dir / "out" / f"{prefix}.path{latest_iteration}"
    for required in (source_input, source_output, source_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "neb.in"
    manifest_path = args.out_dir / "qe_restart_manifest.json"
    resources = {
        "walltime": args.walltime,
        "max_seconds": args.max_seconds,
        "safety_seconds": walltime_seconds(args.walltime) - args.max_seconds,
        "ntasks": args.ntasks,
        "memory": args.memory,
        "kpoint_pools": args.kpoint_pools,
    }
    job_name = derive_name(parent["job_name"], args.attempt, parent.get("seed", args.seed))
    branch_id = derive_name(parent["branch_id"], args.attempt, parent.get("seed", args.seed))
    if output.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        checks = {
            "parent_job": manifest.get("parent", {}).get("job_id") == str(parent["job_id"]),
            "attempt": manifest.get("attempt") == args.attempt,
            "input_hash": manifest.get("generated_input_sha256") == sha256_file(output),
            "parent_input_hash": manifest.get("parent", {}).get("neb_input_sha256") == sha256_file(source_input),
            "parent_output_hash": manifest.get("parent", {}).get("neb_output_sha256") == sha256_file(source_output),
            "latest_path_hash": manifest.get("parent", {}).get("latest_path_sha256") == sha256_file(source_path),
            "resources": manifest.get("resources") == resources,
        }
        if not all(checks.values()):
            raise RuntimeError(f"existing QE NEB continuation conflicts with request: {checks}")
        return manifest, manifest_path
    if output.exists() or manifest_path.exists():
        raise RuntimeError(f"partial QE NEB continuation artifacts in {args.out_dir}")

    text = source_input.read_text()
    text = set_namelist_value(text, "PATH", "restart_mode", "'restart'")
    text = set_namelist_value(text, "CONTROL", "max_seconds", str(args.max_seconds))
    output.write_text(text)
    parsed = parent.get("parsed", {})
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "validated_qe_neb_restart_segment",
        "path_id": parent["path_id"],
        "branch_id": branch_id,
        "job_name": job_name,
        "seed": parent.get("seed", args.seed),
        "attempt": args.attempt,
        "parent": {
            "job_id": str(parent["job_id"]),
            "branch_id": parent["branch_id"],
            "job_name": parent["job_name"],
            "remote_run_dir": parent["remote_run_dir"],
            "classification": parent["classification"],
            "status_file": str(args.qe_status.resolve()),
            "status_file_sha256": sha256_file(args.qe_status),
            "neb_input": str(source_input.resolve()),
            "neb_input_sha256": sha256_file(source_input),
            "neb_output": str(source_output.resolve()),
            "neb_output_sha256": sha256_file(source_output),
            "latest_path_iteration": latest_iteration,
            "qe_prefix": prefix,
            "latest_path": str(source_path.resolve()),
            "latest_path_sha256": sha256_file(source_path),
            "restart_artifacts": parent["restart_artifacts"],
            "last_complete_iteration": parsed.get("last_complete_iteration"),
            "activation_forward_eV": parsed.get("activation_forward_eV"),
            "max_movable_error_eV_A": parsed.get("max_image_error_movable_eV_A"),
            "barrier_drift_last_three_eV": parsed.get("barrier_drift_last_three_eV"),
        },
        "generated_input": str(output.resolve()),
        "generated_input_sha256": sha256_file(output),
        "resources": resources,
        "restart_policy": {
            "restart_mode": "restart",
            "copy_scope": "parent_out_directory_only",
            "copy_verification": "complete_relative_path_sha256_inventory_before_neb",
            "parent_mutated": False,
        },
        "acceptance": {
            "max_movable_error_eV_A": 0.03,
            "max_barrier_drift_last_three_eV": 0.02,
            "final_check_max_movable_error_eV_A": 0.02,
            "complete_final_iteration": True,
            "endpoint_gate_required": True,
        },
    }
    atomic_json(manifest_path, manifest)
    return manifest, manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qe-status", type=Path, required=True)
    parser.add_argument("--active-qe-jobs", type=Path, required=True)
    parser.add_argument("--parent-job-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--remote-input-dir", required=True)
    parser.add_argument("--remote-code-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_pipeline")
    parser.add_argument("--remote-run-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_runs")
    parser.add_argument("--remote-wrapper", default="/scratch16/pclancy3/yi/revision1_migrationbench_pipeline/scripts/migrationbench/submit_slurm_qe_neb_restart_verified.sh")
    parser.add_argument("--ssh-host", default="rockfish")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--walltime", required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--ntasks", type=int, required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--kpoint-pools", type=int, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    if args.attempt < 2:
        raise ValueError("continuation attempt must be at least 2")
    if args.max_seconds >= walltime_seconds(args.walltime):
        raise ValueError("max_seconds must be below Slurm walltime")
    status = json.loads(args.qe_status.read_text())
    parent = select_parent(status, args.parent_job_id)

    lock_path = args.active_qe_jobs.with_name(args.active_qe_jobs.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest, manifest_path = prepare(args, parent)
        if args.submit:
            registry = json.loads(args.active_qe_jobs.read_text())
            registered = next((row for row in registry["jobs"] if row["job_name"] == manifest["job_name"]), None)
            job_id = str(registered["job_id"]) if registered else find_remote_job(args.ssh_host, manifest["job_name"])
            submission_state = "already_registered" if registered else "recovered_from_rockfish" if job_id else "submitted"
            if not job_id:
                run(["ssh", args.ssh_host, "mkdir", "-p", args.remote_input_dir])
                run(["rsync", "-a", str(args.out_dir) + "/", f"{args.ssh_host}:{args.remote_input_dir}/"])
                export = ",".join([
                    "ALL",
                    f"MIGRATIONBENCH_NEB_INPUT={args.remote_input_dir}/neb.in",
                    f"MIGRATIONBENCH_NEB_RESTART_MANIFEST={args.remote_input_dir}/qe_restart_manifest.json",
                    f"MIGRATIONBENCH_PARENT_RUN_DIR={parent['remote_run_dir']}",
                    f"MIGRATIONBENCH_CODE_ROOT={args.remote_code_root}",
                    f"MIGRATIONBENCH_RUN_ROOT={args.remote_run_root}",
                    f"MIGRATIONBENCH_KPOINT_POOLS={args.kpoint_pools}",
                ])
                output_text = run([
                    "ssh", args.ssh_host, "sbatch", "--parsable", "--job-name", manifest["job_name"],
                    "--time", args.walltime, "--ntasks-per-node", str(args.ntasks), "--mem", args.memory,
                    "--export", export, args.remote_wrapper,
                ])
                job_id = output_text.split(";", 1)[0].strip()
                if not re.fullmatch(r"\d+", job_id):
                    raise RuntimeError(f"could not parse Slurm job id from: {output_text!r}")
            if not registered:
                registry["jobs"] = [row for row in registry["jobs"] if str(row["job_id"]) != str(parent["job_id"])]
                registry["jobs"].append({
                    "branch_id": manifest["branch_id"],
                    "job_id": job_id,
                    "job_name": manifest["job_name"],
                    "path_id": manifest["path_id"],
                    "seed": manifest["seed"],
                    "attempt": manifest["attempt"],
                    "parent_job_id": str(parent["job_id"]),
                    "remote_run_dir": f"{args.remote_run_root.rstrip('/')}/{manifest['job_name']}_{job_id}",
                })
                registry["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                atomic_json(args.active_qe_jobs, registry)
            if str(manifest.get("submission", {}).get("job_id", "")) != job_id:
                manifest["submission"] = {"state": submission_state, "job_id": job_id}
                atomic_json(manifest_path, manifest)
                run(["rsync", "-a", str(manifest_path), f"{args.ssh_host}:{args.remote_input_dir}/"])

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
