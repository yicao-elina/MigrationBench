#!/usr/bin/env python3
"""Plan or submit candidate-manifest MACE NEB jobs with production/smoke gates."""

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eligible_candidates(manifest, mode, branches, steps):
    selected = [
        row for row in manifest["candidates"]
        if row["geometry_gate"]["status"] == "pass"
        and row.get("duplicate_of") is None
        and (not branches or row["branch_id"] in branches)
    ]
    if not selected:
        raise RuntimeError("No selected unique geometry-gate-passing candidates")
    if mode == "production":
        if manifest.get("endpoint_status") != "accepted_local_minima":
            raise RuntimeError("Production launch requires accepted_local_minima endpoints")
        if not all(row.get("production_eligible_before_mace") for row in selected):
            raise RuntimeError("One or more selected candidates are not production eligible")
    elif mode == "smoke":
        if steps > 2 or len(selected) > 1:
            raise RuntimeError("Smoke mode permits one candidate and at most two optimizer steps")
    elif mode == "diagnostic":
        if steps > 500:
            raise RuntimeError("Diagnostic mode permits at most 500 optimizer steps")
    else:
        raise ValueError(mode)
    return selected


def run(command, attempts=4):
    for attempt in range(attempts):
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode == 0:
            return result.stdout.strip()
        transient = result.returncode == 255 or "Connection reset" in result.stderr
        if not transient or attempt == attempts - 1:
            raise RuntimeError(f"Command failed: {' '.join(command)}\n{result.stderr}")
        time.sleep(5 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=["production", "diagnostic", "smoke"], required=True)
    parser.add_argument("--branch", action="append", default=[])
    parser.add_argument("--calculator", choices=["mace-foundation", "mace-model"], default="mace-model")
    parser.add_argument("--model-path")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--spring", type=float, default=0.1)
    parser.add_argument("--migrant-tether-k", type=float, default=0.0)
    parser.add_argument("--host-tether-k", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--walltime", default="04:00:00")
    parser.add_argument("--memory", default="48G")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--ssh-host", default="rockfish")
    parser.add_argument("--remote-code-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_pipeline")
    parser.add_argument("--remote-input-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_inputs/nonlinear_mace")
    parser.add_argument("--remote-run-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_runs")
    args = parser.parse_args()
    if args.migrant_tether_k < 0 or args.host_tether_k < 0:
        raise ValueError("Reference-tether force constants must be non-negative")
    if not args.model_path:
        raise ValueError(
            "--model-path is required for reproducible MACE runs, including foundation models"
        )
    model_path_sha256 = None
    model_path = Path(args.model_path).expanduser()
    if model_path.is_file():
        model_path_sha256 = sha256_file(model_path)
    elif args.submit:
        model_path_sha256 = run(
            ["ssh", args.ssh_host, "sha256sum", args.model_path]
        ).split()[0]
    if not model_path_sha256:
        raise ValueError("Unable to establish the explicit model checkpoint SHA-256")
    manifest_path = args.candidate_manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    selected = eligible_candidates(manifest, args.mode, set(args.branch), args.steps)
    jobs = []
    model_token = hashlib.sha256(f"{args.calculator}:{args.model_path}".encode()).hexdigest()[:4]
    for row in selected:
        token = hashlib.sha256(row["branch_id"].encode()).hexdigest()[:6]
        mode_token = {"smoke": "sm", "diagnostic": "dg", "production": "pr"}[args.mode]
        tether_token = ""
        if args.migrant_tether_k > 0 or args.host_tether_k > 0:
            protocol = f"{args.migrant_tether_k:.12g}:{args.host_tether_k:.12g}"
            tether_token = "_tr" + hashlib.sha256(protocol.encode()).hexdigest()[:4]
        job_name = f"mb_nlm{token}_{mode_token}{model_token}{tether_token}_s{args.seed}"
        local_images = manifest_path.parent / row["images"]
        remote_dir = f"{args.remote_input_root}/{job_name}"
        remote_images = f"{remote_dir}/{local_images.name}"
        remote_manifest = f"{remote_dir}/nonlinear_candidate_manifest.json"
        job = {
            "branch_id": row["branch_id"],
            "job_name": job_name,
            "mode": args.mode,
            "seed": args.seed,
            "migrant_element": row.get("migrant_element") or manifest.get("migrant_element") or "Cr",
            "steps": args.steps,
            "fmax_target_eV_A": args.fmax,
            "spring_constant_eV_A2": args.spring,
            "migrant_tether_k_eV_A2": args.migrant_tether_k,
            "host_tether_k_eV_A2": args.host_tether_k,
            "calculator": args.calculator,
            "model_path": args.model_path,
            "model_path_sha256": model_path_sha256,
            "local_images": str(local_images),
            "local_images_sha256": sha256_file(local_images),
            "remote_images": remote_images,
            "remote_candidate_manifest": remote_manifest,
            "remote_input_dir": remote_dir,
            "resources": {"walltime": args.walltime, "memory": args.memory, "cpus": args.cpus},
        }
        if args.submit:
            run(["ssh", args.ssh_host, "mkdir", "-p", remote_dir])
            run(["scp", str(local_images), f"{args.ssh_host}:{remote_images}"])
            run(["scp", str(manifest_path), f"{args.ssh_host}:{remote_manifest}"])
            snapshot_dir = f"{remote_dir}/submitted_code"
            snapshot_files = [
                "scripts/migrationbench/run_mlff_neb.py",
                "scripts/migrationbench/submit_slurm_mlff_neb_cpu.sh",
                "scripts/migrationbench/export_mlff_neb_iteration_history.py",
                "scripts/migrationbench/capture_runtime_provenance.py",
            ]
            copy_lines = "\n".join(
                f"cp {shlex.quote(args.remote_code_root + '/' + source)} {shlex.quote(snapshot_dir + '/' + Path(source).name)}"
                for source in snapshot_files
            )
            snapshot_output = run([
                "ssh", args.ssh_host,
                "set -e; mkdir -p {directory}; {copies}; cd {directory}; "
                "sha256sum * > SHA256SUMS; cat SHA256SUMS".format(
                    directory=shlex.quote(snapshot_dir),
                    copies=copy_lines.replace("\n", "; "),
                ),
            ])
            snapshot_hashes = {
                fields[1]: fields[0]
                for line in snapshot_output.splitlines()
                if len(fields := line.split()) == 2 and fields[1] != "SHA256SUMS"
            }
            if len(snapshot_hashes) != len(snapshot_files):
                raise RuntimeError(f"Incomplete submitted-code snapshot in {snapshot_dir}")
            job["submitted_code_snapshot"] = {
                "remote_directory": snapshot_dir,
                "sha256": snapshot_hashes,
            }
            export_values = {
                "MIGRATIONBENCH_IMAGES": remote_images,
                "MIGRATIONBENCH_PATH_ID": row["branch_id"],
                "MIGRATIONBENCH_CALCULATOR": args.calculator,
                "MIGRATIONBENCH_MODEL_PATH": args.model_path or "",
                "MIGRATIONBENCH_STEPS": str(args.steps),
                "MIGRATIONBENCH_FMAX": str(args.fmax),
                "MIGRATIONBENCH_SPRING": str(args.spring),
                "MIGRATIONBENCH_MIGRANT_ELEMENT": job["migrant_element"],
                "MIGRATIONBENCH_MIGRANT_TETHER_K": str(args.migrant_tether_k),
                "MIGRATIONBENCH_HOST_TETHER_K": str(args.host_tether_k),
                "MIGRATIONBENCH_SEED": str(args.seed),
                "MIGRATIONBENCH_RUN_ROLE": {
                    "smoke": "unverified_smoke",
                    "diagnostic": "diagnostic_preconditioner",
                    "production": "production_preconditioner",
                }[args.mode],
                "MIGRATIONBENCH_SOURCE_CANDIDATE_MANIFEST": remote_manifest,
                "MIGRATIONBENCH_SUBMITTED_CODE_DIR": snapshot_dir,
                "MIGRATIONBENCH_CODE_ROOT": args.remote_code_root,
                "MIGRATIONBENCH_RUN_ROOT": args.remote_run_root,
            }
            exports = "ALL," + ",".join(f"{key}={value}" for key, value in export_values.items())
            sbatch = " ".join(shlex.quote(value) for value in (
                "sbatch", f"--job-name={job_name}", f"--time={args.walltime}",
                f"--mem={args.memory}", f"--cpus-per-task={args.cpus}", f"--export={exports}",
                f"{args.remote_code_root}/scripts/migrationbench/submit_slurm_mlff_neb_cpu.sh",
            ))
            body = f"""set -e
receipt={shlex.quote(remote_dir + '/submission.receipt')}
if test -s \"$receipt\"; then cat \"$receipt\"; exit 0; fi
job_id=$({sbatch} | awk '{{print $NF}}')
tmp=\"$receipt.tmp.$$\"; printf '%s|submitted_new_job\\n' \"$job_id\" > \"$tmp\"; mv \"$tmp\" \"$receipt\"; cat \"$receipt\""""
            receipt = run([
                "ssh", args.ssh_host,
                f"cd {shlex.quote(args.remote_code_root)} && flock -x {shlex.quote(remote_dir + '/submission.lock')} bash -lc {shlex.quote(body)}",
            ])
            job_id, submission_status = receipt.splitlines()[-1].split("|", 1)
            job.update({
                "job_id": job_id,
                "submission_status": submission_status,
                "remote_run_dir": f"{args.remote_run_root}/{job_name}_{job_id}",
            })
        jobs.append(job)
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_manifest": str(manifest_path),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "mode": args.mode,
        "model_path": args.model_path,
        "model_path_sha256": model_path_sha256,
        "submitted": args.submit,
        "jobs": jobs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"jobs": len(jobs), "submitted": args.submit, "output": str(args.out.resolve())}, indent=2))


if __name__ == "__main__":
    main()
