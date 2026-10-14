#!/bin/bash -l
#SBATCH --job-name=foundation3-model-gate
#SBATCH --signal=B:TERM@60

set -euo pipefail
: "${CAMPAIGN_CONFIG:?CAMPAIGN_CONFIG is required}"
: "${MODEL_FAMILY:?MODEL_FAMILY is required}"
: "${TRAINING_SEED:?TRAINING_SEED is required}"
: "${MODEL_OUTPUT_ROOT:?MODEL_OUTPUT_ROOT is required}"
: "${MACE_BOOTSTRAP:?MACE_BOOTSTRAP is required}"
: "${MACE_CONDA_SH:?MACE_CONDA_SH is required}"
: "${MACE_CONDA_ENV:?MACE_CONDA_ENV is required}"
export SLURM_CONF="${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/slurm/slurm.conf}"
source "$MACE_BOOTSTRAP"
source /data/apps/helpers/singularity.sh 2>/dev/null || true
set +u
source "$MACE_CONDA_SH"
conda activate "$MACE_CONDA_ENV"
set -u
python3 "$(dirname "$CAMPAIGN_CONFIG")/../scripts/static_model_gate.py" \
  --config "$CAMPAIGN_CONFIG" \
  --family "$MODEL_FAMILY" \
  --training-seed "$TRAINING_SEED" \
  --output-root "$MODEL_OUTPUT_ROOT" \
  --device "${MODEL_GATE_DEVICE:-cuda}"
