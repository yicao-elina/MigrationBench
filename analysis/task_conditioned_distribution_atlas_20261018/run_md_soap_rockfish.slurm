#!/bin/bash
#SBATCH --job-name=mb_soap_pca
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch16/pclancy3/yi/task_conditioned_distribution_atlas_20261018/md_soap_%A_%a.out
#SBATCH --error=/scratch16/pclancy3/yi/task_conditioned_distribution_atlas_20261018/md_soap_%A_%a.err
#SBATCH --array=0-3%4

set -euo pipefail

PY=/data/pclancy3/yi/.conda/envs/mace/bin/python
SCRIPT=/scratch16/pclancy3/yi/task_conditioned_distribution_atlas_20261018_compute_soap_pca.py
PCA=/scratch16/pclancy3/yi/task_conditioned_distribution_atlas_20261018/grouped/soap_pca_fit.npz
OUT=/scratch16/pclancy3/yi/task_conditioned_distribution_atlas_20261018/md
ROOT=/scratch16/pclancy3/yi/mace_four_model_MD_continuation_20260930

mkdir -p "$OUT"

case "$SLURM_ARRAY_TASK_ID" in
  0) family="Foundation"; model="Foundation-OMAT"; seed=""; input="$ROOT/foundation-omat/segment_from_000008000.extxyz" ;;
  1) family="Scratch"; model="From-scratch"; seed=""; input="$ROOT/1-from-scratch/segment_from_000200000.extxyz" ;;
  2) family="FT-600K"; model="FT-600K"; seed=""; input="$ROOT/2-naive-fine-tuning/segment_from_000080000.extxyz" ;;
  3) family="FT-MultiT"; model="FT-MultiT"; seed=""; input="$ROOT/3-multi_T-fine-tuning/segment_from_000200000.extxyz" ;;
esac

safe=$(echo "$family" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_')
exec "$PY" "$SCRIPT" --mode md --input "$input" --pca "$PCA" --output "$OUT/${safe}_md_soap.csv" --family "$family" --model-id "$model" --training-seed "$seed" --stride 1 --guard-short-contact 1.2
