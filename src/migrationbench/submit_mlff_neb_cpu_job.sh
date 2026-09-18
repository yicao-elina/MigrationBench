#!/bin/bash
# Submit a parameterized CPU MLFF NEB job on Rockfish.

set -euo pipefail

echo "Deprecated unsafe launcher: use launch_mlff_candidate_batch.py with a candidate manifest." >&2
exit 64

if [ "$#" -lt 3 ]; then
  echo "usage: $0 PATH_ID IMAGES MODEL_PATH [STEPS] [SEED] [TIME] [MEM] [CPUS]" >&2
  exit 2
fi

PATH_ID="$1"
IMAGES="$2"
MODEL_PATH="$3"
STEPS="${4:-500}"
SEED="${5:-42}"
TIME_LIMIT="${6:-04:00:00}"
MEM="${7:-48G}"
CPUS="${8:-8}"
JOB_PATH_ID="$(printf '%s' "${PATH_ID}" | tr -c 'A-Za-z0-9' '_')"
JOB_NAME="mb_mlff_cpu_${JOB_PATH_ID}_s${SEED}"

ssh rockfish "cd \$HOME/revision1_pipeline_20260909 && \
  MIGRATIONBENCH_PATH_ID='${PATH_ID}' \
  MIGRATIONBENCH_IMAGES='${IMAGES}' \
  MIGRATIONBENCH_MODEL_PATH='${MODEL_PATH}' \
  MIGRATIONBENCH_STEPS='${STEPS}' \
  MIGRATIONBENCH_SEED='${SEED}' \
  MIGRATIONBENCH_RUN_ROOT=/scratch16/pclancy3/yi/revision1_migrationbench_runs \
  sbatch --job-name='${JOB_NAME}' --time='${TIME_LIMIT}' --mem='${MEM}' --cpus-per-task='${CPUS}' scripts/migrationbench/submit_slurm_mlff_neb_cpu.sh"
