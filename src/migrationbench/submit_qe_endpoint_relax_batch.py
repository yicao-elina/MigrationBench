#!/usr/bin/env python3
"""Upload and submit a prepared endpoint-relax batch to Rockfish."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def run(command, attempts=4):
    last = None
    for attempt in range(attempts):
        try:
            return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()
        except subprocess.CalledProcessError as error:
            last = error
            transient = error.returncode == 255 or "Connection reset" in (error.stderr or "")
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_manifest", type=Path)
    parser.add_argument("--remote-code-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_pipeline")
    parser.add_argument("--remote-input-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_inputs/endpoint_relax")
    parser.add_argument("--remote-run-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_runs")
    args = parser.parse_args()
    payload = json.loads(args.batch_manifest.read_text())
    submitted = []
    output = args.batch_manifest.with_name("submitted_endpoint_relax_jobs.json")
    for job in payload["jobs"]:
        existing = run(
            [
                "ssh",
                "rockfish",
                "squeue",
                "-h",
                "-n",
                job["job_name"],
                "-o",
                "%A",
            ]
        )
        if existing:
            job_id = existing.splitlines()[0].strip()
            submitted.append(
                {
                    "branch_id": job["branch_id"],
                    "path_id": job["source_path_id"],
                    "image_index_qe": job["source_image_index_qe"],
                    "job_name": job["job_name"],
                    "job_id": job_id,
                    "remote_run_dir": f"{args.remote_run_root}/{job['job_name']}_{job_id}",
                    "submission_status": "adopted_existing_active_job",
                }
            )
            output.write_text(json.dumps({"jobs": submitted}, indent=2) + "\n")
            continue
        remote_dir = f"{args.remote_input_root}/{job['branch_id']}"
        run(["ssh", "rockfish", "mkdir", "-p", remote_dir])
        run(["scp", job["relax_input"], f"rockfish:{remote_dir}/relax.in"])
        run(["scp", job["manifest_path"], f"rockfish:{remote_dir}/endpoint_relax_manifest.json"])
        resources = job["resources"]
        export = (
            f"ALL,MIGRATIONBENCH_RELAX_INPUT={remote_dir}/relax.in,"
            f"MIGRATIONBENCH_RELAX_MANIFEST={remote_dir}/endpoint_relax_manifest.json,"
            f"MIGRATIONBENCH_CODE_ROOT={args.remote_code_root},"
            f"MIGRATIONBENCH_RUN_ROOT={args.remote_run_root}"
        )
        stdout = run(
            [
                "ssh",
                "rockfish",
                "sbatch",
                f"--job-name={job['job_name']}",
                f"--time={resources['walltime']}",
                f"--ntasks-per-node={resources['ntasks']}",
                f"--mem={resources['memory']}",
                f"--export={export}",
                f"{args.remote_code_root}/scripts/migrationbench/submit_slurm_qe_relax.sh",
            ]
        )
        job_id = stdout.split()[-1]
        submitted.append(
            {
                "branch_id": job["branch_id"],
                "path_id": job["source_path_id"],
                "image_index_qe": job["source_image_index_qe"],
                "job_name": job["job_name"],
                "job_id": job_id,
                "remote_run_dir": f"{args.remote_run_root}/{job['job_name']}_{job_id}",
                "submission_status": "submitted_new_job",
            }
        )
        output.write_text(json.dumps({"jobs": submitted}, indent=2) + "\n")
    print(json.dumps({"submitted": submitted, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
