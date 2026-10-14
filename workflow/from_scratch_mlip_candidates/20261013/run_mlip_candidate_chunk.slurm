#!/bin/bash -l
#SBATCH --job-name=mb-mlip-cand
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-31%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
source /etc/profile.d/modules.sh
RUN_ROOT="${RUN_ROOT:-/scratch16/pclancy3/yi/mace_candidate_eval_20261013}"
RESULT_ROOT="${RESULT_ROOT:-$RUN_ROOT/results_production}"
MACE_ENV="${MACE_ENV:-/data/pclancy3/yi/.conda/envs/mace}"
export PATH="$MACE_ENV/bin:$PATH"
export PYTHONPATH="/data/pclancy3/yi/.python_pkgs:/data/pclancy3/yi/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$RESULT_ROOT" "$RUN_ROOT/logs"
"$MACE_ENV/bin/python" "$RUN_ROOT/run_mlip_candidate_chunk.py" \
  --manifest "$RUN_ROOT/candidate_manifest_639.csv" \
  --models "$RUN_ROOT/models_rockfish.tsv" \
  --chunk-index "${SLURM_ARRAY_TASK_ID}" --chunk-size 20 \
  --out "$RESULT_ROOT" --device cuda
