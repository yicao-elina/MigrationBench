#!/bin/bash
#SBATCH --job-name=mb_cleargraph15_s48
#SBATCH --time=00:20:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=parallel
#SBATCH -A pclancy3
#SBATCH --mem=16G
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
OUT="${RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}/clearance_graph_1-5_s48"
mkdir -p "${OUT}"
cd "${CODE_ROOT}"
python scripts/migrationbench/reconstruct_migrant_clearance_graph.py \
  --images data_raw/historical_neb/legacy_61/1-5/sb2te3.xyz \
  --cell-template data_raw/historical_neb/legacy_61/1-5/pw_1.in \
  --output "${OUT}/1-5_clearance_graph_13.extxyz" \
  --manifest "${OUT}/1-5_clearance_graph_13.manifest.json" \
  --n-images 13 --clearance-A 1.8 --max-step-A 2.0 \
  --max-offset-A 2.5 --radial-step-A 0.25 --directions 128 \
  --max-candidates 500 --reference-weight 1.0 --step-weight 1.0 --seed 48
