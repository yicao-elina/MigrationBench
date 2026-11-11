#!/bin/bash -l
#SBATCH --job-name=slide_fixed
#SBATCH --output=slurm-fixed-%A_%a.out
#SBATCH --error=slurm-fixed-%A_%a.err

set -euo pipefail
export SLURM_CONF="${SLURM_CONF:?SLURM_CONF must be supplied by submit_array.py}"
export PATH="${SLURM_BIN_DIR:?SLURM_BIN_DIR must be supplied by submit_array.py}:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?CAMPAIGN_ROOT must be supplied by submit_array.py}"
MACE_PYTHON="${MACE_PYTHON:?MACE_PYTHON must be supplied by submit_array.py}"
JOB_TABLE="${JOB_TABLE:?JOB_TABLE must be supplied by submit_array.py}"

exec "${MACE_PYTHON}" "${CAMPAIGN_ROOT}/scripts/eval_fixed_path.py" \
  --config "${CAMPAIGN_ROOT}/config/campaign.yaml" \
  --job-table "${JOB_TABLE}" \
  --index "${SLURM_ARRAY_TASK_ID}"

