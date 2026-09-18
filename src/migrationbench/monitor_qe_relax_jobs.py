#!/usr/bin/env python3
"""Sync, parse, and classify standalone QE endpoint-relax jobs."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from analyze_neb_path_topology import read_qe_image
from parse_qe_relax_output import parse_relax_out, write_extxyz


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_lineage(local_dir, cluster_root, seen=None):
    seen = set() if seen is None else seen
    manifest_path = local_dir / "endpoint_relax_manifest.json"
    output_path = local_dir / "relax.out"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    identity = str(manifest.get("submission", {}).get("job_id") or local_dir.name)
    if identity in seen:
        raise RuntimeError("QE relax lineage cycle at {}".format(identity))
    seen.add(identity)
    segments = []
    parent = manifest.get("parent", {})
    if parent.get("remote_run_dir"):
        parent_dir = cluster_root / Path(parent["remote_run_dir"]).name
        if (parent_dir / "relax.out").exists():
            segments.extend(collect_lineage(parent_dir, cluster_root, seen))
    input_path = local_dir / "relax.in"
    parsed = parse_relax_out(output_path, input_path) if output_path.exists() else {"n_ionic_steps": 0, "total_scf_iterations_seen": 0}
    segments.append({
        "segment_index": len(segments) + 1,
        "local_dir": str(local_dir),
        "relax_input_sha256": sha256_file(local_dir / "relax.in") if (local_dir / "relax.in").exists() else None,
        "relax_output_sha256": sha256_file(output_path) if output_path.exists() else None,
        "ionic_steps": int(parsed.get("n_ionic_steps", 0)),
        "scf_iterations": int(parsed.get("total_scf_iterations_seen", 0)),
        "job_done": bool(parsed.get("job_done")),
        "bfgs_converged": bool(parsed.get("bfgs_converged")),
        "max_seconds_reached": bool(parsed.get("max_seconds_reached")),
    })
    return segments


def sync_lineage_parents(local_dir, cluster_root, host, seen=None):
    """Refresh every declared parent before cumulative lineage parsing."""
    seen = set() if seen is None else seen
    manifest_path = local_dir / "endpoint_relax_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    parent_remote = manifest.get("parent", {}).get("remote_run_dir")
    if not parent_remote or parent_remote in seen:
        return
    seen.add(parent_remote)
    parent_dir = cluster_root / Path(parent_remote).name
    parent_dir.mkdir(parents=True, exist_ok=True)
    remote = "{}:{}/".format(host, parent_remote.rstrip("/"))
    run(
        [
            "rsync", "-a", "--include", "relax.out", "--include", "relax.in",
            "--include", "*.json", "--include", "slurm-*.out", "--include",
            "slurm-*.err", "--exclude", "*", remote, str(parent_dir) + "/",
        ],
        check=False,
    )
    sync_lineage_parents(parent_dir, cluster_root, host, seen)


def run(command, check=True):
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and process.returncode:
        raise RuntimeError("command failed: {}\n{}".format(" ".join(command), process.stderr))
    return process


def queue_state(job_id, host):
    process = run(["ssh", host, "squeue -j {} -h -o '%T|%M|%R'".format(job_id)], check=False)
    if not process.stdout.strip():
        return {"queue_state": "not_in_squeue", "elapsed": None, "reason": None}
    fields = process.stdout.strip().splitlines()[0].split("|", 2)
    return {"queue_state": fields[0], "elapsed": fields[1], "reason": fields[2] if len(fields) > 2 else None}


def accounting_state(job_id, host):
    process = run(
        ["ssh", host, "sacct -X -S now-7days -j {} --format=JobID,JobName%40,State,Elapsed,MaxRSS,ReqMem -P".format(job_id)],
        check=False,
    )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    rows = []
    if len(lines) > 1:
        header = lines[0].split("|")
        rows = [dict(zip(header, line.split("|"))) for line in lines[1:]]
    main = next((row for row in rows if row.get("JobID") == str(job_id)), rows[0] if rows else {})
    return {"sacct_main": main, "sacct_rows": rows}


def scontrol_state(job_id, host):
    process = run(["ssh", host, "scontrol show job {} -o".format(job_id)], check=False)
    fields = {}
    for token in process.stdout.strip().split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return {
        "queue_state": fields.get("JobState"),
        "elapsed": fields.get("RunTime"),
        "reason": fields.get("Reason"),
        "scontrol": fields,
    }


def classify(parsed, queue, accounting_state=""):
    if parsed.get("job_done") and parsed.get("bfgs_converged") and parsed.get("latest_geometry_has_evaluated_forces"):
        force = parsed.get("final_max_atom_force_eV_A")
        return "accepted_local_minimum" if force is not None and force <= 0.05 else "relax_done_force_gate_failed"
    terminal = accounting_state.split("+", 1)[0] in {"COMPLETED", "TIMEOUT", "FAILED", "CANCELLED"}
    if (
        terminal
        and parsed.get("max_seconds_reached")
        and parsed.get("n_ionic_steps", 0) > 0
        and parsed.get("final_force_ionic_step") is not None
        and parsed.get("scf_not_converged_count", 0) == 0
    ):
        return "clean_max_seconds_restartable"
    if (
        terminal
        and parsed.get("job_done")
        and parsed.get("max_ionic_steps_reached")
        and parsed.get("n_ionic_steps", 0) > 0
        and parsed.get("final_force_ionic_step") is not None
        and parsed.get("scf_not_converged_count", 0) == 0
    ):
        return "clean_max_ionic_steps_restartable"
    if parsed.get("n_ionic_steps", 0) > 0:
        return "ionic_relax_in_progress" if queue == "RUNNING" else "partial_relax_needs_review_or_restart"
    if queue == "RUNNING":
        return "first_scf_in_progress"
    if queue == "PENDING":
        return "pending"
    return "terminal_without_ionic_step"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-file", type=Path, action="append", required=True)
    parser.add_argument("--ssh-host", default="rockfish")
    parser.add_argument("--local-cluster-root", type=Path, default=Path("./cluster"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--offline-from-status", type=Path,
        help="Reparse already-synced local outputs using scheduler fields from this status file",
    )
    args = parser.parse_args()
    jobs = []
    for jobs_file in args.jobs_file:
        jobs.extend(json.loads(jobs_file.read_text())["jobs"])
    prior = {}
    if args.offline_from_status:
        prior = {
            str(row["job_id"]): row
            for row in json.loads(args.offline_from_status.read_text()).get("jobs", [])
        }
    results = []
    for job in jobs:
        if args.offline_from_status:
            previous = prior.get(str(job["job_id"]))
            if previous is None:
                raise RuntimeError(f"job {job['job_id']} missing from offline status")
            queue = {
                "queue_state": previous.get("queue_state", "not_in_squeue"),
                "elapsed": previous.get("elapsed"),
                "reason": previous.get("reason"),
            }
            accounting = {
                "sacct_main": previous.get("sacct_main", {}),
                "sacct_rows": previous.get("sacct_rows", []),
            }
        else:
            queue = queue_state(job["job_id"], args.ssh_host)
            accounting = accounting_state(job["job_id"], args.ssh_host)
        accounting_main_state = accounting.get("sacct_main", {}).get("State", "").split("+", 1)[0]
        if queue["queue_state"] == "not_in_squeue" and accounting_main_state in {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}:
            queue["queue_state"] = accounting_main_state
            queue["elapsed"] = accounting.get("sacct_main", {}).get("Elapsed")
            queue["reason"] = "recovered_from_sacct"
        elif queue["queue_state"] == "not_in_squeue" and not accounting.get("sacct_main"):
            fallback = scontrol_state(job["job_id"], args.ssh_host)
            if fallback.get("queue_state"):
                queue = fallback
        local_dir = args.local_cluster_root / Path(job["remote_run_dir"]).name
        local_dir.mkdir(parents=True, exist_ok=True)
        remote = "{}:{}/".format(args.ssh_host, job["remote_run_dir"].rstrip("/"))
        if not args.offline_from_status:
            run(
                ["rsync", "-a", "--include", "relax.out", "--include", "relax.in", "--include", "*.json", "--include", "slurm-*.out", "--include", "slurm-*.err", "--exclude", "*", remote, str(local_dir) + "/"],
                check=False,
            )
            sync_lineage_parents(local_dir, args.local_cluster_root.resolve(), args.ssh_host)
        relax_out = local_dir / "relax.out"
        relax_input = local_dir / "relax.in"
        parsed = parse_relax_out(relax_out, relax_input) if relax_out.exists() else {"n_ionic_steps": 0, "missing": True}
        if parsed.get("steps"):
            relax_input = local_dir / "relax.in"
            input_cell = read_qe_image(relax_input)["cell"] if relax_input.exists() else None
            if input_cell:
                for step in parsed["steps"]:
                    if step.get("cell_A") is None:
                        step["cell_A"] = input_cell
            write_extxyz(local_dir / "relax_iterations.extxyz", parsed["steps"])
        lineage = collect_lineage(local_dir, args.local_cluster_root.resolve())
        lineage_payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "segments": lineage,
            "total_ionic_steps": sum(row["ionic_steps"] for row in lineage),
            "total_scf_iterations": sum(row["scf_iterations"] for row in lineage),
            "segment_count": len(lineage),
        }
        (local_dir / "relax_lineage.json").write_text(json.dumps(lineage_payload, indent=2) + "\n")
        parsed["lineage_total_ionic_steps"] = lineage_payload["total_ionic_steps"]
        parsed["lineage_total_scf_iterations"] = lineage_payload["total_scf_iterations"]
        parsed["lineage_segment_count"] = lineage_payload["segment_count"]
        compact_parsed = {key: value for key, value in parsed.items() if key != "steps"}
        result = dict(job)
        result.update(queue)
        result.update(accounting)
        result["local_dir"] = str(local_dir)
        result["parsed"] = compact_parsed
        result["classification"] = classify(parsed, queue["queue_state"], accounting_main_state)
        results.append(result)
    payload = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "jobs": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
