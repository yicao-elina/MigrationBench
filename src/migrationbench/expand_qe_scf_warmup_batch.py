#!/usr/bin/env python3
"""Gate, prepare, and optionally submit the remaining standalone QE image SCFs."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_image_list(value):
    images = []
    for field in value.split(","):
        if "-" in field:
            start, end = (int(item) for item in field.split("-", 1))
            images.extend(range(start, end + 1))
        else:
            images.append(int(field))
    if not images or len(images) != len(set(images)) or min(images) < 1:
        raise ValueError("images must be unique positive QE image indices")
    return images


def accepted_images(status, path_id):
    return {
        int(row["image_index_qe"])
        for row in status["jobs"]
        if row["path_id"] == path_id and row.get("classification") == "accepted_scf_warmstart"
    }


def run(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{result.stderr}")
    return result.stdout.strip()


def path_job_slug(path_id):
    slug = path_id.lower().replace("_neb_", "n")
    return re.sub(r"[^a-z0-9]+", "", slug)


def find_remote_job(host, job_name):
    # Avoid shell metacharacters because OpenSSH joins remote argv into a shell command.
    active = run(["ssh", host, "squeue", "-h", "-n", job_name, "-o", "%A:%j"])
    for line in active.splitlines():
        job_id, _, observed_name = line.partition(":")
        if observed_name == job_name and re.fullmatch(r"\d+", job_id):
            return job_id
    history = run([
        "ssh", host, "sacct", "-X", "-S", "now-2days", "-n", "-P",
        "--name", job_name, "--format=JobIDRaw,JobName,State",
    ])
    matches = []
    for line in history.splitlines():
        fields = line.split("|")
        if len(fields) >= 2 and fields[1] == job_name and re.fullmatch(r"\d+", fields[0]):
            matches.append(fields[0])
    return max(matches, key=int) if matches else None


def ensure_prepared(args, image, out_dir):
    source = args.source_pw_dir / f"pw_{image}.in"
    generated = out_dir / "scf.in"
    manifest_path = out_dir / "scf_warmup_manifest.json"
    if generated.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        expected_resources = {
            "walltime": args.walltime,
            "max_seconds": args.max_seconds,
            "ntasks": args.ntasks,
            "memory": args.memory,
            "kpoint_pools": args.kpoint_pools,
        }
        checks = {
            "path_id": manifest.get("path_id") == args.path_id,
            "image_index": manifest.get("image_index_qe") == image,
            "seed": manifest.get("seed") == args.seed,
            "source_hash": manifest.get("source_pw_sha256") == sha256_file(source),
            "input_hash": manifest.get("generated_input_sha256") == sha256_file(generated),
            "resources": manifest.get("resources") == expected_resources,
        }
        if not all(checks.values()):
            raise RuntimeError(f"existing prepared image {image} conflicts with requested config: {checks}")
        return source
    if generated.exists() or manifest_path.exists():
        raise RuntimeError(f"partial prepared artifacts for image {image}; inspect {out_dir}")
    return prepare_image(args, image, out_dir)


def prepare_image(args, image, out_dir):
    source = args.source_pw_dir / f"pw_{image}.in"
    command = [
        sys.executable,
        str(args.generator),
        "--source-pw", str(source),
        "--out-dir", str(out_dir),
        "--path-id", args.path_id,
        "--image-index", str(image),
        "--seed", str(args.seed),
        "--walltime", args.walltime,
        "--max-seconds", str(args.max_seconds),
        "--ntasks", str(args.ntasks),
        "--memory", args.memory,
        "--kpoint-pools", str(args.kpoint_pools),
        "--degauss-Ry", str(args.degauss_Ry),
        "--mixing-beta", str(args.mixing_beta),
        "--mixing-mode", args.mixing_mode,
        "--mixing-ndim", str(args.mixing_ndim),
        "--electron-maxstep", str(args.electron_maxstep),
        "--conv-thr", args.conv_thr,
    ]
    run(command)
    return source


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path)
    pre_args, _ = pre_parser.parse_known_args()
    defaults = {}
    if pre_args.config:
        defaults = json.loads(pre_args.config.read_text()).get("arguments", {})

    parser = argparse.ArgumentParser(description=__doc__, parents=[pre_parser])
    parser.set_defaults(**defaults)
    parser.add_argument("--warmup-status", type=Path)
    parser.add_argument("--jobs-file", type=Path)
    parser.add_argument("--source-pw-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--batch-manifest", type=Path)
    parser.add_argument("--path-id")
    parser.add_argument("--pilot-images", default="1-2")
    parser.add_argument("--images", default="3-10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ssh-host", default="rockfish")
    parser.add_argument("--remote-input-root")
    parser.add_argument("--remote-code-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_pipeline")
    parser.add_argument("--remote-run-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_runs")
    parser.add_argument(
        "--remote-wrapper",
        default="/scratch16/pclancy3/yi/revision1_migrationbench_pipeline/scripts/migrationbench/submit_slurm_qe_scf_warmup.sh",
    )
    parser.add_argument(
        "--generator",
        type=Path,
        default=Path(__file__).with_name("prepare_qe_scf_warmup.py"),
    )
    parser.add_argument("--walltime", default="24:00:00")
    parser.add_argument("--max-seconds", type=int, default=84600)
    parser.add_argument("--ntasks", type=int, default=4)
    parser.add_argument("--memory", default="180G")
    parser.add_argument("--kpoint-pools", type=int, default=4)
    parser.add_argument("--degauss-Ry", type=float, default=0.01)
    parser.add_argument("--mixing-beta", type=float, default=0.2)
    parser.add_argument("--mixing-mode", default="local-TF")
    parser.add_argument("--mixing-ndim", type=int, default=8)
    parser.add_argument("--electron-maxstep", type=int, default=800)
    parser.add_argument("--conv-thr", default="1.0d-6")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    required = ["warmup_status", "jobs_file", "source_pw_dir", "output_root", "batch_manifest", "path_id", "remote_input_root"]
    missing = [name for name in required if getattr(args, name) in (None, "")]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    for name in ("warmup_status", "jobs_file", "source_pw_dir", "output_root", "batch_manifest", "generator"):
        setattr(args, name, Path(getattr(args, name)))

    pilot_images = set(parse_image_list(args.pilot_images))
    target_images = parse_image_list(args.images)
    status = json.loads(args.warmup_status.read_text())
    accepted = accepted_images(status, args.path_id)
    missing_pilots = sorted(pilot_images - accepted)
    if missing_pilots:
        raise RuntimeError(f"pilot gate failed; unaccepted images: {missing_pilots}")

    source_paths = [args.source_pw_dir / f"pw_{image}.in" for image in target_images]
    missing_sources = [str(path) for path in source_paths if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"missing source PW inputs: {missing_sources}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.jobs_file.with_name(args.jobs_file.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = json.loads(args.jobs_file.read_text())
        existing = {
            int(row["image_index_qe"]): row
            for row in registry["jobs"]
            if row["path_id"] == args.path_id
        }
        manifest = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "path_id": args.path_id,
            "seed": args.seed,
            "pilot_images": sorted(pilot_images),
            "accepted_pilot_images": sorted(pilot_images & accepted),
            "target_images": target_images,
            "submit_requested": args.submit,
            "resources": {
                "walltime": args.walltime,
                "max_seconds": args.max_seconds,
                "ntasks": args.ntasks,
                "memory": args.memory,
                "kpoint_pools": args.kpoint_pools,
            },
            "images": [],
        }
        if args.batch_manifest.exists():
            previous = json.loads(args.batch_manifest.read_text())
            previous_rows = {int(row["image_index_qe"]): row for row in previous.get("images", [])}
        else:
            previous_rows = {}

        for image in target_images:
            out_dir = (args.output_root / f"image_{image}_s{args.seed}").resolve()
            source = ensure_prepared(args, image, out_dir)
            remote_dir = f"{args.remote_input_root.rstrip('/')}/image_{image}_s{args.seed}"
            job_name = f"mb_scf{path_job_slug(args.path_id)}i{image}_s{args.seed}"
            row = {
                "image_index_qe": image,
                "source_pw_input": str(source.resolve()),
                "source_pw_sha256": sha256_file(source),
                "generated_input": str(out_dir / "scf.in"),
                "generated_input_sha256": sha256_file(out_dir / "scf.in"),
                "generated_manifest": str(out_dir / "scf_warmup_manifest.json"),
                "generated_manifest_sha256": sha256_file(out_dir / "scf_warmup_manifest.json"),
                "remote_input_dir": remote_dir,
                "job_name": job_name,
                "submission_state": "prepared",
            }
            if image in existing:
                row.update({"submission_state": "already_registered", "job_id": existing[image]["job_id"]})
            elif previous_rows.get(image, {}).get("job_id"):
                recovered = previous_rows[image]
                job_id = str(recovered["job_id"])
                row.update({"submission_state": "recovered_from_manifest", "job_id": job_id})
                registry["jobs"].append({
                    "job_id": job_id,
                    "job_name": job_name,
                    "path_id": args.path_id,
                    "image_index_qe": image,
                    "seed": args.seed,
                    "remote_run_dir": f"{args.remote_run_root.rstrip('/')}/{job_name}_{job_id}",
                })
                atomic_json(args.jobs_file, registry)
            elif args.submit:
                recovered_job_id = find_remote_job(args.ssh_host, job_name)
                if recovered_job_id:
                    job_id = recovered_job_id
                    row.update({"submission_state": "recovered_from_rockfish", "job_id": job_id})
                else:
                    run(["ssh", args.ssh_host, "mkdir", "-p", remote_dir])
                    run(["rsync", "-a", str(out_dir) + "/", f"{args.ssh_host}:{remote_dir}/"])
                    export = ",".join([
                        "ALL",
                        f"MIGRATIONBENCH_SCF_INPUT={remote_dir}/scf.in",
                        f"MIGRATIONBENCH_SCF_MANIFEST={remote_dir}/scf_warmup_manifest.json",
                        f"MIGRATIONBENCH_CODE_ROOT={args.remote_code_root}",
                        f"MIGRATIONBENCH_RUN_ROOT={args.remote_run_root}",
                        f"MIGRATIONBENCH_KPOINT_POOLS={args.kpoint_pools}",
                    ])
                    output = run([
                        "ssh", args.ssh_host, "sbatch", "--parsable", "--job-name", job_name,
                        "--export", export, args.remote_wrapper,
                    ])
                    job_id = output.split(";", 1)[0].strip()
                    if not re.fullmatch(r"\d+", job_id):
                        raise RuntimeError(f"could not parse Slurm job id from: {output!r}")
                    row.update({"submission_state": "submitted", "job_id": job_id})
                registry["jobs"].append({
                    "job_id": job_id,
                    "job_name": job_name,
                    "path_id": args.path_id,
                    "image_index_qe": image,
                    "seed": args.seed,
                    "remote_run_dir": f"{args.remote_run_root.rstrip('/')}/{job_name}_{job_id}",
                })
                atomic_json(args.jobs_file, registry)
            manifest["images"].append(row)
            atomic_json(args.batch_manifest, manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
