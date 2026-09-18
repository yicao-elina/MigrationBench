#!/bin/bash
#SBATCH --job-name=mb_trajrep3_sm_s42
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

SEED="${MIGRATIONBENCH_SEED:-42}"
CODE_ROOT="${MIGRATIONBENCH_CODE_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_pipeline}"
RUN_ROOT="${MIGRATIONBENCH_RUN_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_smoke/runs}"
JOB_RUN_DIR="${RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${JOB_RUN_DIR}"
cd "${CODE_ROOT}"

python --version
python scripts/migrationbench/select_representative_trajectories.py \
  --config "${CODE_ROOT}/configs/historical_trajectory_portfolio.json" \
  --out-dir "${JOB_RUN_DIR}/representative_trajectory_portfolio" \
  --seed "${SEED}"

python - "${JOB_RUN_DIR}/representative_trajectory_portfolio/portfolio_manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
checks = manifest["quality_checks"]
assert manifest["seed"] == 42
assert checks["candidate_count"] == 12
assert checks["cross_system_pairs_created"] == 0
assert checks["coverage_monotonic_all"] is True
assert checks["coverage_target_reached_all"] is True
assert checks["valid_center_target_reached_all"] is False
print("representative trajectory smoke gates passed")
PY
