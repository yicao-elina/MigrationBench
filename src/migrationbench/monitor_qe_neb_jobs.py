#!/usr/bin/env python3
"""Monitor QE NEB jobs, sync compact outputs, parse neb.out, and classify state.

This script is intentionally conservative: it does not cancel or submit jobs.
It creates an auditable JSON status file that downstream docs, dashboards, and
heartbeat automations can consume.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from parse_qe_neb_output import parse_neb_out


MAX_SECONDS_RE = re.compile(r"(?im)^\s*max_seconds\s*=\s*([-+0-9.EedD]+)")
NUM_IMAGES_RE = re.compile(r"(?im)^\s*num_of_images\s*=\s*(\d+)")
PREFIX_RE = re.compile(r"(?im)^\s*prefix\s*=\s*['\"]?([^,'\"\s/]+)")


def run(cmd, check=True):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError("command failed: {}\nSTDOUT:\n{}\nSTDERR:\n{}".format(" ".join(cmd), proc.stdout, proc.stderr))
    return proc


def parse_job_spec(spec):
    parts = spec.split("|", 4)
    if len(parts) != 5:
        raise ValueError("job spec must be job_id|path_id|branch_id|job_name|remote_run_dir")
    return {"job_id": parts[0], "path_id": parts[1], "branch_id": parts[2], "job_name": parts[3], "remote_run_dir": parts[4]}


def squeue_state(job_id, ssh_host):
    cmd = ["ssh", ssh_host, "squeue -j {} -h -o '%T|%M|%R'".format(job_id)]
    proc = run(cmd, check=False)
    text = proc.stdout.strip()
    if not text:
        return {"queue_state": "not_in_squeue", "elapsed": None, "reason": None}
    first = text.splitlines()[0].split("|", 2)
    while len(first) < 3:
        first.append(None)
    return {"queue_state": first[0], "elapsed": first[1], "reason": first[2]}


def sacct_state(job_id, ssh_host):
    fmt = "JobID,JobName%40,State,Elapsed,MaxRSS,ReqMem"
    cmd = ["ssh", ssh_host, "sacct -j {} --format={} -P".format(job_id, fmt)]
    proc = run(cmd, check=False)
    rows = []
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) >= 2:
        header = lines[0].split("|")
        for line in lines[1:]:
            rows.append(dict(zip(header, line.split("|"))))
    main = next((row for row in rows if row.get("JobID") == job_id), rows[0] if rows else {})
    return {"sacct_main": main, "sacct_rows": rows}


def sync_run(job, local_cluster_root, ssh_host):
    dest = local_cluster_root / Path(job["remote_run_dir"]).name
    dest.mkdir(parents=True, exist_ok=True)
    remote = "{}:{}/".format(ssh_host, job["remote_run_dir"].rstrip("/"))
    run(
        [
            "rsync",
            "-a",
            "--include",
            "neb.out",
            "--include",
            "*.in",
            "--include",
            "*.json",
            "--include",
            "out/",
            "--include",
            "out/*.path[0-9]*",
            "--include",
            "out/*_*/",
            "--include",
            "out/*_*/PW.out",
            "--include",
            "*/",
            "--exclude",
            "*",
            remote,
            str(dest) + "/",
        ],
        check=False,
    )
    return dest


def parse_input_limits(path):
    if not path.is_file():
        return {"max_seconds": None, "num_of_images": None}
    text = path.read_text(errors="replace")
    max_match = MAX_SECONDS_RE.search(text)
    image_match = NUM_IMAGES_RE.search(text)
    return {
        "max_seconds": float(max_match.group(1).replace("d", "e").replace("D", "E")) if max_match else None,
        "num_of_images": int(image_match.group(1)) if image_match else None,
        "prefix": prefix_match.group(1) if (prefix_match := PREFIX_RE.search(text)) else "sb2te3",
    }


def restart_inventory(remote_dir, expected_images, prefix, ssh_host):
    out = shlex.quote(remote_dir.rstrip("/") + "/out")
    quoted_prefix = shlex.quote(prefix or "sb2te3")
    command = (
        f"out={out}; prefix={quoted_prefix}; "
        "path_count=$(find \"$out\" -maxdepth 1 -type f -name \"${prefix}.path[0-9]*\" -size +0c 2>/dev/null | wc -l); "
        "latest=$(find \"$out\" -maxdepth 1 -type f -name \"${prefix}.path[0-9]*\" -size +0c 2>/dev/null | sed 's/.*path//' | sort -n | tail -1); "
        "image_dirs=0; complete=0; "
        "for d in \"$out\"/\"${prefix}_\"*; do [ -d \"$d\" ] || continue; image_dirs=$((image_dirs+1)); "
        "[ -s \"$d/${prefix}.save/charge-density.dat\" ] && [ -s \"$d/${prefix}.save/data-file-schema.xml\" ] && complete=$((complete+1)); done; "
        "printf '%s|%s|%s|%s' \"$path_count\" \"${latest:-}\" \"$image_dirs\" \"$complete\""
    )
    proc = run(["ssh", ssh_host, command], check=False)
    fields = proc.stdout.strip().split("|")
    if len(fields) != 4:
        return {"path_files": 0, "latest_path_iteration": None, "image_save_dirs": 0, "complete_image_save_dirs": 0, "restart_ready": False}
    path_count, latest, image_dirs, complete = fields
    ready = int(path_count) > 0 and expected_images is not None and int(complete) >= expected_images
    return {
        "path_files": int(path_count),
        "latest_path_iteration": int(latest) if latest else None,
        "image_save_dirs": int(image_dirs),
        "complete_image_save_dirs": int(complete),
        "restart_ready": ready,
    }


def classify(parsed, queue_state, accounting_state="", input_limits=None, restart=None):
    input_limits = input_limits or {}
    restart = restart or {}
    if parsed.get("has_overflow"):
        return "nonphysical_or_overflow"
    if parsed.get("converged_by_default_gate"):
        return "converged"
    if queue_state == "RUNNING":
        return "running_unconverged" if parsed.get("n_complete_iterations_parsed", 0) > 0 else "running_no_table_yet"
    if queue_state == "PENDING":
        return "pending"
    max_seconds = input_limits.get("max_seconds")
    tcpu = parsed.get("last_tcpu_seconds")
    reached_time_guard = bool(max_seconds and tcpu is not None and tcpu >= 0.90 * max_seconds)
    terminal = queue_state == "not_in_squeue" or accounting_state.split("+", 1)[0] not in {"", "RUNNING", "PENDING"}
    if terminal and parsed.get("job_done") and reached_time_guard and restart.get("restart_ready"):
        return "clean_max_seconds_restartable"
    if terminal:
        return "terminal_needs_review"
    return "unknown_active_state"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", action="append", default=[], help="job_id|path_id|branch_id|job_name|remote_run_dir")
    ap.add_argument("--jobs-file", help="JSON file with a jobs array; each job has job_id, path_id, branch_id, job_name, remote_run_dir")
    ap.add_argument("--ssh-host", default="rockfish")
    ap.add_argument("--local-cluster-root", default="./cluster")
    ap.add_argument("--output", default="registry/runtime_state/qe_neb_job_status.json")
    args = ap.parse_args()

    local_root = Path(args.local_cluster_root)
    job_specs = list(args.job or [])
    jobs = [parse_job_spec(spec) for spec in job_specs]
    if args.jobs_file:
        loaded = json.loads(Path(args.jobs_file).read_text())
        jobs.extend(loaded.get("jobs", []))
    if not jobs:
        raise ValueError("provide at least one --job or --jobs-file")

    results = []
    for job in jobs:
        q = squeue_state(job["job_id"], args.ssh_host)
        acct = sacct_state(job["job_id"], args.ssh_host)
        local_dir = sync_run(job, local_root, args.ssh_host)
        neb_out = local_dir / "neb.out"
        if neb_out.exists():
            parsed = parse_neb_out(neb_out)
        else:
            parsed = {"neb_out": str(neb_out), "missing": True, "has_overflow": False, "n_complete_iterations_parsed": 0}
        result = dict(job)
        result.update(q)
        result.update(acct)
        result["local_dir"] = str(local_dir)
        result["parsed"] = parsed
        limits = parse_input_limits(local_dir / "neb.in")
        inventory = restart_inventory(
            job["remote_run_dir"], limits.get("num_of_images"), limits.get("prefix"), args.ssh_host
        )
        result["input_limits"] = limits
        result["restart_artifacts"] = inventory
        result["classification"] = classify(
            parsed,
            q.get("queue_state"),
            acct.get("sacct_main", {}).get("State", ""),
            limits,
            inventory,
        )
        results.append(result)

    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
