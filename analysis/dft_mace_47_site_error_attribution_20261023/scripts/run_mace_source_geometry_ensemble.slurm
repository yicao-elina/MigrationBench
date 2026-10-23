#!/bin/bash
#SBATCH --job-name=mb47-mace-static
#SBATCH --account=pclancy3_gpu
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-11%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
source /etc/profile.d/modules.sh

RUN_ROOT=/scratch16/pclancy3/yi/dft_mace_47_source_geometry_ensemble_20261023
MACE_ENV=/data/pclancy3/yi/.conda/envs/mace
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/results"

"$MACE_ENV/bin/python" "$RUN_ROOT/scripts/run_mace_source_geometry_ensemble.py" \
  --registry "$RUN_ROOT/MODEL_REGISTRY.tsv" \
  --manifest "$RUN_ROOT/MACE_SOURCE_GEOMETRY_INPUT_MANIFEST.tsv" \
  --inputs-root "$RUN_ROOT/inputs" \
  --output-root "$RUN_ROOT/results" \
  --model-index "$SLURM_ARRAY_TASK_ID" \
  --device cuda
