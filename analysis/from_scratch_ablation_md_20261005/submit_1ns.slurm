#!/bin/bash -l
#SBATCH --job-name=ablate-md1ns
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --signal=USR1@120
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --array=0-2%2
set -euo pipefail

export MACE_ENV=/data/pclancy3/yi/.conda/envs/mace
export PATH="$MACE_ENV/bin:$PATH"
export PYTHONPATH=/data/pclancy3/yi/.python_pkgs:/data/pclancy3/yi/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

REGISTRY="$CAMPAIGN_ROOT/registry/selected_1ns.tsv"
ROW=$(awk -v idx="$SLURM_ARRAY_TASK_ID" 'NR == idx + 2 {print}' "$REGISTRY")
IFS=$'\t' read -r FAMILY TRAIN_SEED MODEL MODEL_SHA REASON <<< "$ROW"
RUN_NAME="1ns__${FAMILY}__seed${TRAIN_SEED}__mdseed20261005__r00"
OUT="$CAMPAIGN_ROOT/runs/$RUN_NAME"

exec "$MACE_ENV/bin/python" "$CAMPAIGN_ROOT/code/run_ablation_md.py" \
  --input "$CAMPAIGN_ROOT/inputs/common_initial.xyz" \
  --model "$MODEL" \
  --output "$OUT" \
  --family "$FAMILY" \
  --train-seed "$TRAIN_SEED" \
  --md-seed 20261005 \
  --steps 1000000 \
  --write-interval 200 \
  --temperature 600 \
  --friction 0.01 \
  --timestep-fs 1.0 \
  --device cuda
