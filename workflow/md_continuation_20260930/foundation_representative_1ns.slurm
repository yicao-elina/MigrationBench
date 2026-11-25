#!/bin/bash -l
#SBATCH --job-name=foundation-rep-1ns
#SBATCH --account=pclancy3_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --output=/scratch16/pclancy3/yi/mace_foundation_representatives_1ns_20261006/logs/%A_%a.out
#SBATCH --error=/scratch16/pclancy3/yi/mace_foundation_representatives_1ns_20261006/logs/%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=4

: "${REP_ROOT:?REP_ROOT is required}"
: "${REP_SOURCE:?REP_SOURCE is required}"
: "${REP_RUN:?REP_RUN is required}"
: "${REP_LABEL:?REP_LABEL is required}"
: "${REP_TRAIN:?REP_TRAIN is required}"
: "${REP_MODEL:?REP_MODEL is required}"
: "${REP_INPUT:?REP_INPUT is required}"
: "${REP_SCRIPT:?REP_SCRIPT is required}"

PY=/data/pclancy3/yi/.conda/envs/mace/bin/python
mkdir -p "$REP_ROOT/logs" "$REP_RUN"

set +e
child=0
forward_signal() {
  if [ "$child" -gt 0 ]; then
    kill -USR1 "$child" 2>/dev/null || true
  fi
}
trap forward_signal USR1 TERM
"$PY" "$REP_SCRIPT" \
  --source-dir "$REP_SOURCE" \
  --input "$REP_INPUT" \
  --model "$REP_MODEL" \
  --model-label "$REP_LABEL" \
  --training-seed "$REP_TRAIN" \
  --md-seed 1100 \
  --run-dir "$REP_RUN" \
  --target-step 1000000 \
  --temperature-K 600 \
  --device cuda &
child=$!
wait "$child"
rc=$?
trap - USR1 TERM
set -e

if [ "$rc" -eq 99 ]; then
  next=$(sbatch --dependency="afterany:${SLURM_JOB_ID}" \
    --export="ALL,REP_ROOT=${REP_ROOT},REP_SOURCE=${REP_SOURCE},REP_RUN=${REP_RUN},REP_LABEL=${REP_LABEL},REP_TRAIN=${REP_TRAIN},REP_MODEL=${REP_MODEL},REP_INPUT=${REP_INPUT},REP_SCRIPT=${REP_SCRIPT}" \
    "$0" | awk '{print $4}')
  echo "walltime/stop checkpoint saved; dependent continuation ${next}" >&2
  exit 0
elif [ "$rc" -ne 0 ]; then
  echo "scientific or provenance failure; refusing automatic continuation (rc=$rc)" >&2
  exit "$rc"
fi
