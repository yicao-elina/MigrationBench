#!/usr/bin/env python3
"""Upload and submit a prepared QE protocol-sensitivity SCF batch."""

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path


def parse_receipt(text):
    lines = text.splitlines()
    fields = lines[-1].split("|", 1) if lines else []
    if len(fields) != 2 or not fields[0].isdigit() or not fields[1]:
        raise RuntimeError(f"Invalid Slurm submission receipt: {text!r}")
    return fields


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
    parser.add_argument("batch_manifest", type=Path)
    parser.add_argument("--remote-code-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_pipeline")
    parser.add_argument("--remote-input-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_inputs/protocol_sensitivity")
    parser.add_argument("--remote-run-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_runs")
    args = parser.parse_args()
    payload = json.loads(args.batch_manifest.read_text())
    submitted = []
    output = args.batch_manifest.with_name("submitted_scf_sensitivity_jobs.json")
    for job in payload["jobs"]:
        remote_dir = f"{args.remote_input_root}/{job['branch_id']}"
        run(["ssh", "rockfish", "mkdir", "-p", remote_dir])
        run(["scp", job["generated_input"], f"rockfish:{remote_dir}/scf.in"])
        run(["scp", job["manifest_path"], f"rockfish:{remote_dir}/scf_warmup_manifest.json"])
        resources = job["resources"]
        export = (
            f"ALL,MIGRATIONBENCH_SCF_INPUT={remote_dir}/scf.in,"
            f"MIGRATIONBENCH_SCF_MANIFEST={remote_dir}/scf_warmup_manifest.json,"
            f"MIGRATIONBENCH_CODE_ROOT={args.remote_code_root},"
            f"MIGRATIONBENCH_KPOINT_POOLS={resources['kpoint_pools']},"
            f"MIGRATIONBENCH_RUN_ROOT={args.remote_run_root}"
        )
        sbatch = " ".join(
            shlex.quote(value)
            for value in (
                "sbatch", f"--job-name={job['job_name']}", f"--time={resources['walltime']}",
                f"--ntasks-per-node={resources['ntasks']}", f"--mem={resources['memory']}",
                f"--export={export}",
                f"{args.remote_code_root}/scripts/migrationbench/submit_slurm_qe_scf_warmup.sh",
            )
        )
        body = f"""set -eo pipefail
receipt={shlex.quote(remote_dir + '/submission.receipt')}
if test -s \"$receipt\"; then
  existing=$(cat \"$receipt\")
  case \"$existing\" in [0-9]*\\|*) printf '%s\\n' \"$existing\"; exit 0;; esac
  mv \"$receipt\" \"$receipt.invalid.$(date +%s)\"
fi
job_id=$(squeue -h -n {shlex.quote(job['job_name'])} -o %A | head -n 1)
if test -n \"$job_id\"; then status=adopted_existing_active_job
else
  submission=$({sbatch})
  job_id=$(printf '%s\\n' \"$submission\" | awk 'NF {{print $NF}}' | tail -n 1)
  test -n \"$job_id\"
  status=submitted_new_job
fi
tmp=\"$receipt.tmp.$$\"
printf '%s|%s\\n' \"$job_id\" \"$status\" > \"$tmp\"
mv \"$tmp\" \"$receipt\"
cat \"$receipt\""""
        lock = shlex.quote(remote_dir + "/submission.lock")
        receipt = run(["ssh", "rockfish", f"flock -x {lock} bash -lc {shlex.quote(body)}"])
        job_id, status = parse_receipt(receipt)
        submitted.append({
            "branch_id": job["branch_id"], "path_id": job["path_id"],
            "image_index_qe": job["image_index_qe"], "profile_id": job["profile"]["id"],
            "job_name": job["job_name"], "job_id": job_id,
            "remote_run_dir": f"{args.remote_run_root}/{job['job_name']}_{job_id}",
            "submission_status": status,
        })
        output.write_text(json.dumps({"jobs": submitted}, indent=2) + "\n")
    print(json.dumps({"submitted": submitted, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
