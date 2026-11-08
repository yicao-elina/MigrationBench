#!/bin/bash -l
#SBATCH --job-name=foundation3-ft-md
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --signal=B:USR1@180
#SBATCH --requeue

set -euo pipefail
: "${CAMPAIGN_CONFIG:?CAMPAIGN_CONFIG is required}"
: "${MACE_BOOTSTRAP:?MACE_BOOTSTRAP is required}"
: "${MACE_CONDA_SH:?MACE_CONDA_SH is required}"
: "${MACE_CONDA_ENV:?MACE_CONDA_ENV is required}"
: "${MACE_RUNTIME:?MACE_RUNTIME is required}"
export SLURM_CONF="${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/slurm/slurm.conf}"
source "$MACE_BOOTSTRAP"
source /data/apps/helpers/singularity.sh 2>/dev/null || true
if [[ "$MACE_RUNTIME" == "polar" ]]; then
  : "${MACE_POLAR_VENV:?MACE_POLAR_VENV is required for PolarMACE}"
  source "$MACE_POLAR_VENV/bin/activate"
else
  source "$MACE_CONDA_SH"
  conda activate "$MACE_CONDA_ENV"
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python3 "$(dirname "$CAMPAIGN_CONFIG")/../scripts/run_md_replica.py" --config "$CAMPAIGN_CONFIG" --index "$SLURM_ARRAY_TASK_ID"
