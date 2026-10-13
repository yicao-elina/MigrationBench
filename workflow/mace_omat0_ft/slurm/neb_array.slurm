#!/bin/bash -l
#SBATCH --job-name=omat0-neb
#SBATCH --output=logs/omat0-neb_%A_%a.out
#SBATCH --error=logs/omat0-neb_%A_%a.err
set -euo pipefail

source "${MACE_BOOTSTRAP:?MACE_BOOTSTRAP is required}"
source "${MACE_CONDA_SH:?MACE_CONDA_SH is required}"
conda activate "${MACE_CONDA_ENV:?MACE_CONDA_ENV is required}"

python3 "$(dirname "${CAMPAIGN_CONFIG}")/../scripts/run_neb_task.py" \
  --config "${CAMPAIGN_CONFIG}" \
  --neb-manifest "${NEB_MANIFEST}" \
  --index "${SLURM_ARRAY_TASK_ID}"
