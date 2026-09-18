#!/bin/bash
# Continue MigrationBench 1-7 from a previous restartable QE run directory.

set -euo pipefail

echo "Deprecated historical recipe: use ../continue_qe_neb.py after monitor validation." >&2
exit 64

if [ "$#" -lt 2 ]; then
  echo "usage: $0 PREVIOUS_REMOTE_RUN_DIR ROUND_NUMBER [WALLTIME]" >&2
  echo "example: $0 /scratch16/pclancy3/yi/revision1_migrationbench_runs/mb_qe17_mace_r1_s42_307XXXXX 2 24:00:00" >&2
  exit 2
fi

PREVIOUS_RUN_DIR="$1"
ROUND="$2"
WALLTIME="${3:-24:00:00}"
INPUT_ROOT="/scratch16/pclancy3/yi/revision1_migrationbench_inputs/mb_qe17_mace_r1_s42"
JOB_NAME="mb_qe17_mace_r${ROUND}_s42"

ssh rockfish "cd \$HOME/revision1_pipeline_20260909 && \
  bash scripts/migrationbench/submit_qe_neb_restart_job.sh \
  '${JOB_NAME}' \
  '${INPUT_ROOT}/neb.in' \
  '${PREVIOUS_RUN_DIR}' \
  2 \
  '${WALLTIME}' \
  160G \
  1800 \
  restart"
