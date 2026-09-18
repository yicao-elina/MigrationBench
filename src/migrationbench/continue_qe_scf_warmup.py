#!/usr/bin/env python3
"""Prepare and optionally submit one provenance-preserving QE SCF continuation."""

import argparse
import fcntl
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from expand_qe_scf_warmup_batch import atomic_json, find_remote_job, run, sha256_file
from prepare_qe_scf_warmup import set_namelist_value, walltime_seconds


def selected_parent(status, job_id):
    matches = [row for row in status["jobs"] if str(row["job_id"]) == str(job_id)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one parent row for job {job_id}, found {len(matches)}")
    row = matches[0]
    if row.get("classification") != "clean_timeout_needs_restart":
        raise RuntimeError(f"parent {job_id} is not restartable: {row.get('classification')}")
    if row.get("queue_state") in {"RUNNING", "PENDING"}:
        raise RuntimeError(f"parent {job_id} is still active")
    restart = row.get("restart_artifacts", {})
    if not restart.get("charge_density") or not restart.get("xml"):
        raise RuntimeError(f"parent {job_id} lacks charge/XML restart artifacts")
    return row


def prepare(args, parent):
    parent_input = Path(parent["local_dir"]) / "scf.in"
    if not parent_input.is_file():
        raise FileNotFoundError(parent_input)
    parent_manifest = Path(parent["local_dir"]) / "scf_warmup_manifest.json"
    output = args.out_dir / "scf.in"
    manifest_path = args.out_dir / "scf_continuation_manifest.json"
    expected_resources = {
        "walltime": args.walltime,
        "max_seconds": args.max_seconds,
        "ntasks": args.ntasks,
        "memory": args.memory,
        "kpoint_pools": args.kpoint_pools,
    }
    if output.is_file() and manifest_path.is_file():
        payload = json.loads(manifest_path.read_text())
        checks = {
            "parent": payload.get("parent", {}).get("job_id") == str(parent["job_id"]),
            "attempt": payload.get("attempt") == args.attempt,
            "source_hash": payload.get("parent", {}).get("source_input_sha256") == sha256_file(parent_input),
            "input_hash": payload.get("generated_input_sha256") == sha256_file(output),
            "resources": payload.get("resources") == expected_resources,
        }
        if not all(checks.values()):
            raise RuntimeError(f"existing continuation conflicts with requested config: {checks}")
        return payload, manifest_path
    if output.exists() or manifest_path.exists():
        raise RuntimeError(f"partial continuation artifacts in {args.out_dir}")

    text = parent_input.read_text()
    updates = {
        "CONTROL": {
            "calculation": "'scf'",
            "restart_mode": "'from_scratch'",
            "outdir": "'./out/'",
            "max_seconds": str(args.max_seconds),
            "disk_io": "'low'",
        },
        "ELECTRONS": {
            "startingpot": "'file'",
            "startingwfc": "'atomic+random'",
        },
    }
    for section, values in updates.items():
        for key, value in values.items():
            text = set_namelist_value(text, section, key, value)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_job_name = re.sub(r"_r\d+$", "", parent["job_name"])
    job_name = f"{base_job_name}_r{args.attempt}"
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "standalone_image_scf_charge_density_continuation",
        "path_id": parent["path_id"],
        "image_index_qe": parent["image_index_qe"],
        "seed": parent["seed"],
        "attempt": args.attempt,
        "job_name": job_name,
        "parent": {
            "job_id": str(parent["job_id"]),
            "job_name": parent["job_name"],
            "remote_run_dir": parent["remote_run_dir"],
            "source_save_dir": parent["remote_run_dir"].rstrip("/") + "/out/sb2te3.save",
            "source_input": str(parent_input.resolve()),
            "source_input_sha256": sha256_file(parent_input),
            "source_manifest": str(parent_manifest.resolve()) if parent_manifest.is_file() else None,
            "source_manifest_sha256": sha256_file(parent_manifest) if parent_manifest.is_file() else None,
            "status_file": str(args.warmup_status.resolve()),
            "status_file_sha256": sha256_file(args.warmup_status),
            "classification": parent["classification"],
        },
        "generated_input": str(output.resolve()),
        "resources": expected_resources,
        "electronic_restart": {
            "restart_mode": "from_scratch",
            "startingpot": "file",
            "startingwfc": "atomic+random",
            "copied_required": ["charge-density.dat", "data-file-schema.xml"],
            "copy_excluded": ["wfc*.dat"],
            "reason": "preserve_parent_run_and_reuse_charge_density_without_copying_large_wavefunctions",
        },
    }
    output.write_text(text)
    payload["generated_input_sha256"] = sha256_file(output)
    atomic_json(manifest_path, payload)
    return payload, manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-status", type=Path, required=True)
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--parent-job-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--remote-input-dir", required=True)
    parser.add_argument("--remote-code-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_pipeline")
    parser.add_argument("--remote-run-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_runs")
    parser.add_argument("--remote-wrapper", default="/scratch16/pclancy3/yi/revision1_migrationbench_pipeline/scripts/migrationbench/submit_slurm_qe_scf_warmup_restart.sh")
    parser.add_argument("--ssh-host", default="rockfish")
    parser.add_argument("--walltime", default="24:00:00")
    parser.add_argument("--max-seconds", type=int, default=84600)
    parser.add_argument("--ntasks", type=int, default=4)
    parser.add_argument("--memory", default="180G")
    parser.add_argument("--kpoint-pools", type=int, default=4)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    if args.attempt < 2:
        raise ValueError("continuation attempt must be at least 2")
    if args.max_seconds >= walltime_seconds(args.walltime):
        raise ValueError("max_seconds must be below the Slurm walltime")
    status = json.loads(args.warmup_status.read_text())
    parent = selected_parent(status, args.parent_job_id)
    lock_path = args.jobs_file.with_name(args.jobs_file.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        payload, manifest_path = prepare(args, parent)

        if args.submit:
            job_name = payload["job_name"]
            registry = json.loads(args.jobs_file.read_text())
            registered = next((row for row in registry["jobs"] if row["job_name"] == job_name), None)
            job_id = str(registered["job_id"]) if registered else find_remote_job(args.ssh_host, job_name)
            submission_state = "already_registered" if registered else "recovered_from_rockfish" if job_id else "submitted"
            if not job_id:
                run(["ssh", args.ssh_host, "mkdir", "-p", args.remote_input_dir])
                run(["rsync", "-a", str(args.out_dir) + "/", f"{args.ssh_host}:{args.remote_input_dir}/"])
                export = ",".join([
                    "ALL",
                    f"MIGRATIONBENCH_SCF_INPUT={args.remote_input_dir}/scf.in",
                    f"MIGRATIONBENCH_SCF_MANIFEST={args.remote_input_dir}/scf_continuation_manifest.json",
                    f"MIGRATIONBENCH_CODE_ROOT={args.remote_code_root}",
                    f"MIGRATIONBENCH_RUN_ROOT={args.remote_run_root}",
                    f"MIGRATIONBENCH_KPOINT_POOLS={args.kpoint_pools}",
                ])
                submission_output = run(["ssh", args.ssh_host, "sbatch", "--parsable", "--job-name", job_name, "--export", export, args.remote_wrapper])
                job_id = submission_output.split(";", 1)[0].strip()
                if not re.fullmatch(r"\d+", job_id):
                    raise RuntimeError(f"could not parse Slurm job id from: {submission_output!r}")

            if not registered:
                registry["jobs"].append({
                    "job_id": job_id,
                    "job_name": job_name,
                    "path_id": parent["path_id"],
                    "image_index_qe": parent["image_index_qe"],
                    "seed": parent["seed"],
                    "attempt": args.attempt,
                    "parent_job_id": str(parent["job_id"]),
                    "remote_run_dir": f"{args.remote_run_root.rstrip('/')}/{job_name}_{job_id}",
                })
                atomic_json(args.jobs_file, registry)
            existing_submission = payload.get("submission", {})
            if str(existing_submission.get("job_id", "")) != job_id:
                payload["submission"] = {"state": submission_state, "job_id": job_id}
                atomic_json(manifest_path, payload)
            run(["rsync", "-a", str(manifest_path), f"{args.ssh_host}:{args.remote_input_dir}/"])

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
