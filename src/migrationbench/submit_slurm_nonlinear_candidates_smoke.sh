#!/bin/bash
#SBATCH --job-name=mb_nlpath_smoke_s42
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

: "${MIGRATIONBENCH_CANDIDATE_MANIFEST:?set candidate manifest}"
RUN_ROOT="${MIGRATIONBENCH_RUN_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_runs}"
JOB_RUN_DIR="${RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${JOB_RUN_DIR}"

echo "host=$(hostname)"
echo "seed=42"
echo "manifest=${MIGRATIONBENCH_CANDIDATE_MANIFEST}"
echo "job_run_dir=${JOB_RUN_DIR}"
python --version

python scripts/migrationbench/validate_nonlinear_candidate_manifest.py \
  --manifest "${MIGRATIONBENCH_CANDIDATE_MANIFEST}" \
  --output "${JOB_RUN_DIR}/candidate_validation.json"

cp "${MIGRATIONBENCH_CANDIDATE_MANIFEST}" "${JOB_RUN_DIR}/nonlinear_candidate_manifest.json"
