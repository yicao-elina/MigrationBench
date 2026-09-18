#!/bin/bash
#SBATCH --job-name=mb_mlhist_smoke_s42
#SBATCH --time=00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=parallel
#SBATCH -A pclancy3
#SBATCH --mem=4G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail

source /data/apps/go.sh
module restore || true
source ~/.bashrc
conda activate mace
set -u

: "${MIGRATIONBENCH_MLFF_TRAJECTORY:?set source MLFF trajectory}"
: "${MIGRATIONBENCH_N_IMAGES:?set image count}"
RUN_ROOT="${MIGRATIONBENCH_RUN_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_smoke/runs}"
JOB_RUN_DIR="${RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${JOB_RUN_DIR}"

echo "host=$(hostname)"
echo "seed=42"
echo "source_trajectory=${MIGRATIONBENCH_MLFF_TRAJECTORY}"
echo "job_run_dir=${JOB_RUN_DIR}"

python scripts/migrationbench/export_mlff_neb_iteration_history.py \
  --trajectory "${MIGRATIONBENCH_MLFF_TRAJECTORY}" \
  --n-images "${MIGRATIONBENCH_N_IMAGES}" \
  --spring-constant "${MIGRATIONBENCH_SPRING:-0.1}" \
  --method improvedtangent \
  --out-dir "${JOB_RUN_DIR}"
