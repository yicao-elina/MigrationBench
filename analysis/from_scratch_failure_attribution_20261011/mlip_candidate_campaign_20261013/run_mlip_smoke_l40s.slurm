#!/bin/bash -l
#SBATCH --job-name=mb-mlip-smoke-l40s
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
source /etc/profile.d/modules.sh
RUN_ROOT="${RUN_ROOT:-/scratch16/pclancy3/yi/mace_candidate_eval_20261013}"
MACE_ENV="${MACE_ENV:-/data/pclancy3/yi/.conda/envs/mace}"
export PATH="$MACE_ENV/bin:$PATH"
export PYTHONPATH="/data/pclancy3/yi/.python_pkgs:/data/pclancy3/yi/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$RUN_ROOT/smoke_results_l40s" "$RUN_ROOT/logs"
"$MACE_ENV/bin/python" "$RUN_ROOT/run_mlip_candidate_chunk.py" \
  --manifest "$RUN_ROOT/candidate_manifest_639.csv" \
  --models "$RUN_ROOT/models_rockfish.tsv" \
  --chunk-index 0 --chunk-size 1 --out "$RUN_ROOT/smoke_results_l40s" --device cuda
