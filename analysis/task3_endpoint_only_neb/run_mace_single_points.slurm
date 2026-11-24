#!/bin/bash
#SBATCH --job-name=t3-path-sp
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -euo pipefail
ROOT="${TASK3_ROOT:?set TASK3_ROOT}"
PYTHON="${MACE_PYTHON:-/data/pclancy3/yi/.conda/envs/mace/bin/python}"
MODELS="${MODELS:-$ROOT/models.tsv}"
IFS=$'\t' read -r MODEL_ID MODEL_LABEL MODEL_PATH <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID+2))p" "$MODELS")"
OUT_ROOT="${RESULT_ROOT:-$ROOT/results}"
OUT="$OUT_ROOT/$MODEL_ID"
mkdir -p "$OUT"
INPUT_DIR="${INPUT_DIR:-$ROOT/inputs}"
exec "$PYTHON" "$ROOT/run_mace_single_points.py" --input-dir "$INPUT_DIR" --model-id "$MODEL_ID" --model-label "$MODEL_LABEL" --model-path "$MODEL_PATH" --output-dir "$OUT" --device cpu
