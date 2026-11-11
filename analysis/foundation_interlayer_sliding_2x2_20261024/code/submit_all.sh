#!/bin/bash -l
# Submit the complete frozen 12-row production campaign after pre-run review.
set -euo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export PATH=/cm/shared/apps/slurm/current/bin:$PATH
ROOT=/scratch16/pclancy3/yi/foundation_interlayer_sliding_2x2_20261024
GATE="${ROOT}/manifests/PRE_RUN_REVIEW_GATE.json"
test -s "${ROOT}/ACCEPTANCE_FREEZE_R13.json"
test -s "${GATE}"
grep -q '"data_status": "PASS"' "${GATE}"
grep -q '"visual_status": "PASS"' "${GATE}"
mkdir -p "${ROOT}/logs" "${ROOT}/runs/fixed" "${ROOT}/runs/self" "${ROOT}/manifests"
fixed_job_id="$(sbatch --parsable "${ROOT}/code/run_fixed_array.slurm")"
self_job_id="$(sbatch --parsable "${ROOT}/code/run_self_array.slurm")"
{
  printf 'submitted_at_utc\tfixed_array_job_id\tself_array_job_id\n'
  printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${fixed_job_id}" "${self_job_id}"
} > "${ROOT}/manifests/SUBMISSION_RECORD.tsv"
printf 'fixed_array_job_id=%s\nself_array_job_id=%s\n' "${fixed_job_id}" "${self_job_id}"
