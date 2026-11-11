#!/bin/bash -l
#SBATCH --job-name=omat0-1ns-retry
#SBATCH --account=pclancy3_gpu
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/scratch16/pclancy3/yi/mace_foundation_omat_true_1ns_20261013/logs/omat0_retry_%A.out
#SBATCH --error=/scratch16/pclancy3/yi/mace_foundation_omat_true_1ns_20261013/logs/omat0_retry_%A.err

set -euo pipefail
export OMP_NUM_THREADS=4

: "${REP_ROOT:?REP_ROOT is required}"
: "${REP_SOURCE:?REP_SOURCE is required}"
: "${REP_RUN:?REP_RUN is required}"
: "${REP_CHECKPOINT_SOURCE:?REP_CHECKPOINT_SOURCE is required}"
: "${REP_LABEL:?REP_LABEL is required}"
: "${REP_TRAIN:?REP_TRAIN is required}"
: "${REP_MODEL:?REP_MODEL is required}"
: "${REP_INPUT:?REP_INPUT is required}"
: "${REP_SCRIPT:?REP_SCRIPT is required}"

PY=/data/pclancy3/yi/.conda/envs/mace/bin/python
SBATCH=/cm/shared/apps/slurm/23.02.2/bin/sbatch
mkdir -p "$REP_ROOT/logs" "$REP_RUN"

# Seed a new, isolated continuation directory from the last committed checkpoint.
# The historical directory and its uncheckpointed tail are never modified.
if [ ! -e "$REP_RUN/status.json" ]; then
  test -f "$REP_CHECKPOINT_SOURCE/status.json"
  test -f "$REP_CHECKPOINT_SOURCE/latest_restart.extxyz"
  cp -p "$REP_CHECKPOINT_SOURCE/status.json" "$REP_RUN/status.json"
  cp -p "$REP_CHECKPOINT_SOURCE/latest_restart.extxyz" "$REP_RUN/latest_restart.extxyz"
fi

set +e
child=0
stop_requested=0

forward_usr1() {
  stop_requested=1
  if [ "$child" -gt 0 ]; then
    kill -USR1 "$child" 2>/dev/null || true
  fi
}

# With --signal=B, Slurm signals this batch shell only. Forward the signal to
# the Python child so it can checkpoint and return its continuation code.
trap forward_usr1 USR1

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

# A trapped signal can interrupt wait(2) with 128+SIGUSR1 while the child is
# still alive. Keep waiting until the child really exits.
while :; do
  wait "$child"
  rc=$?
  if kill -0 "$child" 2>/dev/null; then
    continue
  fi
  break
done

trap - USR1
set -e

if [ "$rc" -eq 99 ] || [ "$rc" -eq 138 ]; then
  next=$("$SBATCH" --dependency="afterany:${SLURM_JOB_ID}" \
    --export="ALL,REP_ROOT=${REP_ROOT},REP_SOURCE=${REP_SOURCE},REP_RUN=${REP_RUN},REP_CHECKPOINT_SOURCE=${REP_CHECKPOINT_SOURCE},REP_LABEL=${REP_LABEL},REP_TRAIN=${REP_TRAIN},REP_MODEL=${REP_MODEL},REP_INPUT=${REP_INPUT},REP_SCRIPT=${REP_SCRIPT}" \
    "$0" | awk '{print $4}')
  echo "walltime/signal checkpoint saved; dependent continuation ${next}" >&2
  exit 0
elif [ "$rc" -ne 0 ]; then
  echo "scientific or provenance failure; refusing automatic continuation (rc=$rc)" >&2
  exit "$rc"
fi
