#!/bin/bash -l
# 20 ps smoke gate followed by an afterok signal-safe 1 ns continuation.
# Model-specific values are exported at submission time.
#SBATCH --job-name=foundation-smoke-safe
#SBATCH --account=pclancy3_gpu
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch16/pclancy3/yi/foundation_md_backlog_20261024/logs/%x-%A.out
#SBATCH --error=/scratch16/pclancy3/yi/foundation_md_backlog_20261024/logs/%x-%A.err

set -euo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
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
: "${SMOKE_SCRIPT:?SMOKE_SCRIPT is required}"
: "${CONT_WRAPPER:?CONT_WRAPPER is required}"
REP_MD_SEED="${REP_MD_SEED:-1100}"
REP_START_STEP="${REP_START_STEP:-200000}"
REP_VEL_SEED="${REP_VEL_SEED:-}"
PY="${REP_PYTHON:-/data/pclancy3/yi/.conda/envs/mace/bin/python}"
REP_PYTHONPATH="${REP_PYTHONPATH:-}"
if [ -n "$REP_PYTHONPATH" ]; then
  export PYTHONPATH="$REP_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
fi
SMOKE_HEAD_ARGS=()
if [ -n "${REP_HEAD:-}" ]; then
  SMOKE_HEAD_ARGS=(--head "$REP_HEAD")
fi
SMOKE_VELOCITY_ARGS=()
if [ -n "$REP_VEL_SEED" ]; then
  SMOKE_VELOCITY_ARGS+=(--velocity-seed "$REP_VEL_SEED")
fi
SMOKE_INIT_ARGS=()
if [ "${REP_FRESH_VELOCITY:-0}" = "1" ]; then
  SMOKE_INIT_ARGS+=(--fresh-velocity)
fi

mkdir -p "$REP_ROOT/logs"
test -x "$PY"
test -s "$SMOKE_SCRIPT"
test -s "$REP_MODEL"
test -s "$REP_INPUT"
test ! -e "$REP_SOURCE" || { echo "refusing to overwrite smoke directory: $REP_SOURCE" >&2; exit 43; }

{
  echo "campaign_id=foundation_model_1ns_md_backlog_20261024"
  echo "model_label=$REP_LABEL"
  echo "training_seed=$REP_TRAIN"
  echo "md_velocity_seed=$REP_MD_SEED"
  echo "velocity_seed=$REP_VEL_SEED"
  echo "start_step=$REP_START_STEP"
  echo "fresh_velocity=${REP_FRESH_VELOCITY:-0}"
  echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
  echo "host=$(hostname)"
  echo "python=$PY"
  echo "pythonpath=$REP_PYTHONPATH"
  sha256sum "$REP_MODEL" "$REP_INPUT" "$SMOKE_SCRIPT" "$CONT_WRAPPER"
  if [ -n "$REP_PYTHONPATH" ]; then
    sha256sum "$REP_PYTHONPATH/mace/modules/blocks.py"
  fi
} | tee "$REP_ROOT/preflight_smoke.txt"

"$PY" "$SMOKE_SCRIPT" \
  --input "$REP_INPUT" \
  --model "$REP_MODEL" \
  --model-label "$REP_LABEL" \
  --training-seed "$REP_TRAIN" \
  --md-seed "$REP_MD_SEED" \
  --output "$REP_SOURCE" \
  --steps 20000 \
  --start-step "$REP_START_STEP" \
  --temperature-K 600 \
  --device cuda \
  "${SMOKE_HEAD_ARGS[@]}" \
  "${SMOKE_VELOCITY_ARGS[@]}" \
  "${SMOKE_INIT_ARGS[@]}"

"$PY" - "$REP_SOURCE/status.json" <<'PY'
import json
import sys

status = json.load(open(sys.argv[1]))
if status.get("smoke_pass") is not True or status.get("reason") != "target_steps":
    raise SystemExit(f"smoke gate failed: {status}")
PY

echo "smoke gate passed; releasing signal-safe 1 ns continuation"
next=$(sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" \
  --export="ALL,REP_ROOT=${REP_ROOT},REP_SOURCE=${REP_SOURCE},REP_RUN=${REP_RUN},REP_CHECKPOINT_SOURCE=${REP_CHECKPOINT_SOURCE},REP_LABEL=${REP_LABEL},REP_TRAIN=${REP_TRAIN},REP_MD_SEED=${REP_MD_SEED},REP_VEL_SEED=${REP_VEL_SEED},REP_FRESH_VELOCITY=${REP_FRESH_VELOCITY:-0},REP_PYTHON=${PY},REP_PYTHONPATH=${REP_PYTHONPATH},REP_HEAD=${REP_HEAD:-},REP_MODEL=${REP_MODEL},REP_INPUT=${REP_INPUT},REP_SCRIPT=${REP_SCRIPT}" \
  "$CONT_WRAPPER")
echo "continuation_job_id=${next}"
