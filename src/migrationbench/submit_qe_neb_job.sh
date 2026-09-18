#!/bin/bash
# Submit a QE neb.x job on Rockfish from a prepared neb.in file.

set -euo pipefail

echo "Deprecated unsafe launcher: use continue_qe_neb.py or launch_qe_neb_from_scf_warmups.py with a manifest." >&2
exit 64

if [ "$#" -lt 2 ]; then
  echo "usage: $0 JOB_NAME LOCAL_OR_REMOTE_NEB_IN [NTASKS_PER_NODE] [TIME] [MEM]" >&2
  exit 2
fi

JOB_NAME="$1"
NEB_IN="$2"
NTASKS="${3:-2}"
TIME_LIMIT="${4:-24:00:00}"
MEM="${5:-160G}"
REMOTE_INPUT_ROOT="/scratch16/pclancy3/yi/revision1_migrationbench_inputs/${JOB_NAME}"

if [[ "${NEB_IN}" == /scratch16/* || "${NEB_IN}" == /data/* || "${NEB_IN}" == /home/* ]]; then
  REMOTE_NEB_IN="${NEB_IN}"
else
  ssh rockfish "mkdir -p '${REMOTE_INPUT_ROOT}'"
  scp "${NEB_IN}" "rockfish:${REMOTE_INPUT_ROOT}/neb.in" >/dev/null
  REMOTE_NEB_IN="${REMOTE_INPUT_ROOT}/neb.in"
fi

ssh rockfish "cd \$HOME/revision1_pipeline_20260909 && \
  MIGRATIONBENCH_NEB_INPUT='${REMOTE_NEB_IN}' \
  MIGRATIONBENCH_RUN_ROOT=/scratch16/pclancy3/yi/revision1_migrationbench_runs \
  sbatch --job-name='${JOB_NAME}' --time='${TIME_LIMIT}' --ntasks-per-node='${NTASKS}' --mem='${MEM}' scripts/migrationbench/submit_slurm_qe_neb.sh"
