#!/bin/bash
#SBATCH --job-name=mb_geomcov15_s48
#SBATCH --time=00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=parallel
#SBATCH -A pclancy3
#SBATCH --mem=8G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail
source /data/apps/go.sh
module restore || true
source ~/.bashrc
conda activate mace
set -u

CODE_ROOT="${MIGRATIONBENCH_CODE_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_pipeline}"
RUN_ROOT="${MIGRATIONBENCH_RUN_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_smoke/runs}"
SOURCE_RUN="${MIGRATIONBENCH_SOURCE_RUN:-/scratch16/pclancy3/yi/revision1_migrationbench_smoke/runs/mb_cleargraph15_s48_30850985/clearance_graph_1-5_s48}"
OUT="${RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}/geometry_coverage"
mkdir -p "${OUT}"
cd "${CODE_ROOT}"
python scripts/migrationbench/score_geometry_candidate_coverage.py \
  --images "${SOURCE_RUN}/1-5_clearance_graph_13.extxyz" \
  --path-id 1-5_clearance_graph_s48 --system-id Sb2Te3Cr_61 \
  --selection-config configs/historical_trajectory_portfolio.json \
  --portfolio-manifest data_processed/representative_trajectory_portfolio/portfolio_manifest.json \
  --descriptors data_processed/representative_trajectory_portfolio/trajectory_descriptors.json \
  --out-dir "${OUT}"
