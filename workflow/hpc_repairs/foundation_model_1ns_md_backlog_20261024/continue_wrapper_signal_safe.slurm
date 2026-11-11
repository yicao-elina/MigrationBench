#!/bin/bash -l
# Signal-safe, non-overwriting 1 ns continuation wrapper for Foundation MD.
# Model/run-specific values are supplied through REP_* environment variables.
#SBATCH --job-name=foundation-1ns-safe
#SBATCH --account=pclancy3_gpu
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/scratch16/pclancy3/yi/foundation_md_backlog_20261024/logs/%x-%A.out
#SBATCH --error=/scratch16/pclancy3/yi/foundation_md_backlog_20261024/logs/%x-%A.err

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
REP_MD_SEED="${REP_MD_SEED:-1100}"
REP_VEL_SEED="${REP_VEL_SEED:-}"
PY="${REP_PYTHON:-/data/pclancy3/yi/.conda/envs/mace/bin/python}"
REP_PYTHONPATH="${REP_PYTHONPATH:-}"
if [ -n "$REP_PYTHONPATH" ]; then
  export PYTHONPATH="$REP_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
fi
CONT_HEAD_ARGS=()
if [ -n "${REP_HEAD:-}" ]; then
  CONT_HEAD_ARGS=(--head "$REP_HEAD")
fi
CONT_VELOCITY_ARGS=()
if [ -n "$REP_VEL_SEED" ]; then
  CONT_VELOCITY_ARGS=(--velocity-seed "$REP_VEL_SEED")
fi

SBATCH=/cm/shared/apps/slurm/23.02.2/bin/sbatch
mkdir -p "$REP_ROOT/logs" "$REP_RUN"
test -x "$PY"

# Copy only the validated checkpoint into an isolated continuation directory.
# Historical smoke/continuation directories are never modified.
if [ ! -e "$REP_RUN/status.json" ]; then
  test -s "$REP_CHECKPOINT_SOURCE/status.json"
  test -s "$REP_CHECKPOINT_SOURCE/latest_restart.extxyz"
  cp -p "$REP_CHECKPOINT_SOURCE/status.json" "$REP_RUN/status.json"
  cp -p "$REP_CHECKPOINT_SOURCE/latest_restart.extxyz" "$REP_RUN/latest_restart.extxyz"
fi

set +e
child=0

forward_usr1() {
  if [ "$child" -gt 0 ]; then
    kill -USR1 "$child" 2>/dev/null || true
  fi
}

# Slurm's B:USR1 targets this batch shell. Forward it to the Python driver so
# the driver writes a checkpoint and returns its continuation code.
trap forward_usr1 USR1

"$PY" "$REP_SCRIPT" \
  --source-dir "$REP_SOURCE" \
  --input "$REP_INPUT" \
  --model "$REP_MODEL" \
  --model-label "$REP_LABEL" \
  --training-seed "$REP_TRAIN" \
  --md-seed "$REP_MD_SEED" \
  --run-dir "$REP_RUN" \
  --target-step 1000000 \
  --temperature-K 600 \
  --device cuda \
  "${CONT_HEAD_ARGS[@]}" \
  "${CONT_VELOCITY_ARGS[@]}" &
child=$!

# wait(2) can return 128+SIGUSR1 while the child is still alive. Do not
# interpret that transient value as a scientific failure.
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
    --export="ALL,REP_ROOT=${REP_ROOT},REP_SOURCE=${REP_SOURCE},REP_RUN=${REP_RUN},REP_CHECKPOINT_SOURCE=${REP_CHECKPOINT_SOURCE},REP_LABEL=${REP_LABEL},REP_TRAIN=${REP_TRAIN},REP_MD_SEED=${REP_MD_SEED},REP_VEL_SEED=${REP_VEL_SEED},REP_FRESH_VELOCITY=${REP_FRESH_VELOCITY:-0},REP_PYTHON=${PY},REP_PYTHONPATH=${REP_PYTHONPATH},REP_HEAD=${REP_HEAD:-},REP_MODEL=${REP_MODEL},REP_INPUT=${REP_INPUT},REP_SCRIPT=${REP_SCRIPT}" \
    "$0" | awk '{print $4}')
  echo "walltime/signal checkpoint saved; dependent continuation ${next}" >&2
  exit 0
elif [ "$rc" -ne 0 ]; then
  echo "scientific or provenance failure; refusing automatic continuation (rc=$rc)" >&2
  exit "$rc"
fi
