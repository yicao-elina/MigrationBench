#!/usr/bin/env python3
"""Compare a non-label QE SCF scaling probe with its 2-rank baseline prefix."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def elapsed_seconds(value):
    if not value:
        return None
    days = 0
    if "-" in value:
        day, value = value.split("-", 1)
        days = int(day)
    fields = [int(item) for item in value.split(":")]
    if len(fields) == 3:
        hours, minutes, seconds = fields
    elif len(fields) == 2:
        hours, minutes, seconds = 0, *fields
    else:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def iteration_rate(iterations, seconds):
    if not iterations or not seconds:
        return None
    return iterations * 3600.0 / seconds


def analyze(baseline, probe, speedup_threshold=1.5):
    baseline_elapsed = elapsed_seconds(
        (baseline.get("sacct_main") or baseline.get("accounting") or {}).get("Elapsed")
        or baseline.get("elapsed")
    )
    probe_elapsed = elapsed_seconds(
        (probe.get("accounting") or probe.get("sacct_main") or {}).get("Elapsed")
        or probe.get("elapsed")
    )
    baseline_iterations = int(
        baseline.get("parsed", {}).get("total_scf_iterations_seen")
        or baseline.get("parsed", {}).get("scf_iteration_records")
        or 0
    )
    probe_iterations = int(
        probe.get("parsed", {}).get("scf_iteration_records")
        or probe.get("parsed", {}).get("total_scf_iterations_seen")
        or 0
    )
    baseline_rate = iteration_rate(baseline_iterations, baseline_elapsed)
    probe_rate = iteration_rate(probe_iterations, probe_elapsed)
    rate_ratio = (
        probe_rate / baseline_rate
        if baseline_rate and probe_rate is not None else None
    )
    probe_classification = probe.get("classification")
    probe_clean = probe_classification in {
        "accepted_scf_warmstart",
        "scf_running",
        "clean_timeout_needs_restart",
    }
    sufficient = (
        probe_clean
        and rate_ratio is not None
        and rate_ratio >= speedup_threshold
        and probe_iterations >= 2
    )
    return {
        "baseline_job_id": str(baseline["job_id"]),
        "probe_job_id": str(probe["job_id"]),
        "baseline_ntasks": 2,
        "probe_ntasks": int(probe.get("ntasks", 4)),
        "baseline_elapsed_seconds": baseline_elapsed,
        "probe_elapsed_seconds": probe_elapsed,
        "baseline_scf_iterations": baseline_iterations,
        "probe_scf_iterations": probe_iterations,
        "baseline_iterations_per_hour": baseline_rate,
        "probe_iterations_per_hour": probe_rate,
        "walltime_iteration_rate_ratio": rate_ratio,
        "baseline_cpu_hours": (
            baseline_elapsed * 2 / 3600.0 if baseline_elapsed else None
        ),
        "probe_cpu_hours": (
            probe_elapsed * int(probe.get("ntasks", 4)) / 3600.0
            if probe_elapsed else None
        ),
        "probe_classification": probe_classification,
        "probe_scf_converged": bool(
            probe.get("parsed", {}).get("scf_converged")
        ),
        "speedup_threshold": speedup_threshold,
        "recommend_4r4p_for_future_clean_continuations": sufficient,
        "current_scientific_jobs_unchanged": True,
        "label_eligibility": False,
    }


def find_job(payload, job_id):
    matches = [row for row in payload["jobs"] if str(row["job_id"]) == str(job_id)]
    if len(matches) != 1:
        raise ValueError(f"Expected one status row for job {job_id}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-status", type=Path, required=True)
    parser.add_argument("--baseline-job-id", required=True)
    parser.add_argument("--probe-status", type=Path, required=True)
    parser.add_argument("--probe-job-id", required=True)
    parser.add_argument("--speedup-threshold", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_path = args.baseline_status.resolve()
    probe_path = args.probe_status.resolve()
    result = analyze(
        find_job(json.loads(baseline_path.read_text()), args.baseline_job_id),
        find_job(json.loads(probe_path.read_text()), args.probe_job_id),
        args.speedup_threshold,
    )
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "qe_scf_cpu_scaling_diagnostic_not_label",
        "baseline_status": str(baseline_path),
        "baseline_status_sha256": sha256_file(baseline_path),
        "probe_status": str(probe_path),
        "probe_status_sha256": sha256_file(probe_path),
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
