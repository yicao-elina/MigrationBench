#!/bin/bash
#SBATCH --job-name=mb_epaccept_sm_s42
#SBATCH --time=00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
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

SEED="${MIGRATIONBENCH_SEED:-42}"
CODE_ROOT="${MIGRATIONBENCH_CODE_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_pipeline}"
RUN_ROOT="${MIGRATIONBENCH_RUN_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_smoke/runs}"
JOB_RUN_DIR="${RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${JOB_RUN_DIR}"
cd "${CODE_ROOT}"

python --version
python scripts/migrationbench/smoke_endpoint_acceptance_state_machine.py \
  --seed "${SEED}" \
  --output "${JOB_RUN_DIR}/endpoint_acceptance_smoke.json"

python scripts/migrationbench/build_endpoint_acceptance_batch.py \
  --first-status "${JOB_RUN_DIR}/fixture/first_status.json" \
  --repeat-status "${JOB_RUN_DIR}/fixture/repeat_status.json" \
  --policy "${CODE_ROOT}/configs/representative_path_selection.json" \
  --out-dir "${JOB_RUN_DIR}/batch_acceptance"

python scripts/migrationbench/assign_endpoint_basins.py \
  --acceptance "${JOB_RUN_DIR}/batch_acceptance/endpoint_acceptance_batch.json" \
  --policy "${CODE_ROOT}/configs/representative_path_selection.json" \
  --out-dir "${JOB_RUN_DIR}/basin_assignments"
