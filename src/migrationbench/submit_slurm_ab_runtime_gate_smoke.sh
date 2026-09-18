#!/bin/bash
#SBATCH --job-name=mb_abprov_s42
#SBATCH --time=00:05:00
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

test "${SEED}" = "42"
python -m pytest -q \
  tests/test_representative_path_pipeline.py::RepresentativePathPipelineTest::test_ab_comparison_never_finalizes_basin_or_speedup_from_running_prefix \
  tests/test_representative_path_pipeline.py::RepresentativePathPipelineTest::test_ab_speedup_runtime_gate_covers_every_lineage_segment \
  | tee "${JOB_RUN_DIR}/pytest.out"
