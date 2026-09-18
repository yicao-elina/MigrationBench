#!/bin/bash
# Submit a restartable QE neb.x job on Rockfish.

set -euo pipefail

echo "Deprecated unsafe restart chain: use ../continue_qe_neb.py after the monitor restart gate." >&2
exit 64

if [ "$#" -lt 2 ]; then
  echo "usage: $0 JOB_NAME LOCAL_OR_REMOTE_NEB_IN [PARENT_RUN_DIR|-] [NTASKS] [TIME] [MEM] [SAFETY_SECONDS] [RESTART_MODE]" >&2
  exit 2
fi

JOB_NAME="$1"
NEB_IN="$2"
PARENT_RUN_DIR="${3:--}"
NTASKS="${4:-2}"
TIME_LIMIT="${5:-24:00:00}"
MEM="${6:-160G}"
SAFETY_SECONDS="${7:-1800}"
RESTART_MODE="${8:-from_scratch}"
REMOTE_INPUT_ROOT="/scratch16/pclancy3/yi/revision1_migrationbench_inputs/${JOB_NAME}"

time_to_seconds() {
  local t="$1"
  local h m s
  IFS=: read -r h m s <<<"${t}"
  if [ -z "${s:-}" ]; then
    echo "TIME must be HH:MM:SS" >&2
    exit 2
  fi
  echo $((10#$h * 3600 + 10#$m * 60 + 10#$s))
}

TOTAL_SECONDS="$(time_to_seconds "${TIME_LIMIT}")"
MAX_SECONDS="$((TOTAL_SECONDS - SAFETY_SECONDS))"
if [ "${MAX_SECONDS}" -le 600 ]; then
  echo "Computed max_seconds=${MAX_SECONDS}; choose a longer walltime or smaller safety margin." >&2
  exit 2
fi

if [[ "${NEB_IN}" == /scratch16/* || "${NEB_IN}" == /data/* || "${NEB_IN}" == /home/* ]]; then
  REMOTE_NEB_IN="${NEB_IN}"
else
  ssh rockfish "mkdir -p '${REMOTE_INPUT_ROOT}'"
  scp "${NEB_IN}" "rockfish:${REMOTE_INPUT_ROOT}/neb.in" >/dev/null
  REMOTE_NEB_IN="${REMOTE_INPUT_ROOT}/neb.in"
fi

PARENT_ENV=""
if [ "${PARENT_RUN_DIR}" != "-" ]; then
  PARENT_ENV="MIGRATIONBENCH_PARENT_RUN_DIR='${PARENT_RUN_DIR}'"
fi

ssh rockfish "cd \$HOME/revision1_pipeline_20260909 && \
  MIGRATIONBENCH_NEB_INPUT='${REMOTE_NEB_IN}' \
  ${PARENT_ENV} \
  MIGRATIONBENCH_RESTART_MODE='${RESTART_MODE}' \
  MIGRATIONBENCH_QE_MAX_SECONDS='${MAX_SECONDS}' \
  MIGRATIONBENCH_WALLTIME='${TIME_LIMIT}' \
  MIGRATIONBENCH_SAFETY_SECONDS='${SAFETY_SECONDS}' \
  MIGRATIONBENCH_CHAIN_ID='${JOB_NAME}' \
  MIGRATIONBENCH_RUN_ROOT=/scratch16/pclancy3/yi/revision1_migrationbench_runs \
  sbatch --job-name='${JOB_NAME}' --time='${TIME_LIMIT}' --ntasks-per-node='${NTASKS}' --mem='${MEM}' scripts/migrationbench/submit_slurm_qe_neb_restart.sh"
