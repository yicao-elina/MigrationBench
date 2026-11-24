#!/bin/bash -l
# Exact-model 20 ps gate, followed only after success by a signal-safe 1 ns continuation.
# This script is additive and refuses to overwrite any existing model run directory.
#SBATCH --job-name=ftA2-smoke
#SBATCH --account=pclancy3_gpu
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch16/pclancy3/yi/grouped_A2_ft_1ns_md_20261025/logs/%x-%A_%a.out
#SBATCH --error=/scratch16/pclancy3/yi/grouped_A2_ft_1ns_md_20261025/logs/%x-%A_%a.err

set -euo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

ROOT=/scratch16/pclancy3/yi/grouped_A2_ft_1ns_md_20261025
REGISTRY="$ROOT/MODEL_REGISTRY.tsv"
PY=/data/pclancy3/yi/.conda/envs/mace/bin/python
SMOKE_SCRIPT=/scratch16/pclancy3/yi/foundation_md_backlog_20261024/seedreplica_smoke_head.py
CONT_SCRIPT=/scratch16/pclancy3/yi/foundation_md_backlog_20261024/seedreplica_continue_head.py
CONT_WRAPPER="$ROOT/grouped_A2_continue_signal_safe.slurm"
SBATCH=/cm/shared/apps/slurm/23.02.2/bin/sbatch

test -x "$PY"
test -s "$REGISTRY"
test -s "$SMOKE_SCRIPT"
test -s "$CONT_SCRIPT"
test -s "$CONT_WRAPPER"

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$REGISTRY")
test -n "$line"
IFS=$'\t' read -r INDEX MODEL_ID FAMILY TRAINING_SEED MODEL_PATH MODEL_SHA256 INPUT_PATH INPUT_SHA256 START_STEP TARGET_STEP TEMP DT FRICTION MD_SEED SMOKE_DIR RUN_DIR <<< "$line"

test "$INDEX" = "$SLURM_ARRAY_TASK_ID"
test -s "$MODEL_PATH"
test -s "$INPUT_PATH"
test ! -e "$SMOKE_DIR"
test ! -e "$RUN_DIR"

actual_model_sha=$(sha256sum "$MODEL_PATH" | awk '{print $1}')
actual_input_sha=$(sha256sum "$INPUT_PATH" | awk '{print $1}')
test "$actual_model_sha" = "$MODEL_SHA256"
test "$actual_input_sha" = "$INPUT_SHA256"

mkdir -p "$ROOT/logs" "$ROOT/smoke" "$ROOT/runs"
{
  echo "campaign_id=ft_grouped_A2_1ns_md_20261025"
  echo "array_task_id=$SLURM_ARRAY_TASK_ID"
  echo "model_id=$MODEL_ID"
  echo "family=$FAMILY"
  echo "training_seed=$TRAINING_SEED"
  echo "start_step=$START_STEP target_step=$TARGET_STEP"
  echo "temperature_K=$TEMP timestep_fs=$DT friction=$FRICTION md_seed=$MD_SEED"
  echo "host=$(hostname) job=${SLURM_JOB_ID:-unknown} python=$PY"
  echo "smoke_script=$SMOKE_SCRIPT"
  echo "continuation_script=$CONT_SCRIPT"
  sha256sum "$MODEL_PATH" "$INPUT_PATH" "$SMOKE_SCRIPT" "$CONT_SCRIPT" "$CONT_WRAPPER"
} | tee "$ROOT/logs/preflight-${MODEL_ID}-${SLURM_JOB_ID:-local}.txt"

"$PY" "$SMOKE_SCRIPT" \
  --input "$INPUT_PATH" \
  --model "$MODEL_PATH" \
  --model-label "$MODEL_ID" \
  --training-seed "$TRAINING_SEED" \
  --md-seed "$MD_SEED" \
  --output "$SMOKE_DIR" \
  --steps 20000 \
  --start-step "$START_STEP" \
  --temperature-K "$TEMP" \
  --device cuda

"$PY" - "$SMOKE_DIR/status.json" "$SMOKE_DIR/run_manifest.json" "$MODEL_SHA256" "$INPUT_SHA256" "$START_STEP" "$TARGET_STEP" <<'PY'
import json
import sys

status = json.load(open(sys.argv[1]))
manifest = json.load(open(sys.argv[2]))
model_sha, input_sha = sys.argv[3:5]
start, target = map(int, sys.argv[5:7])
expected = start + 20000
if status.get("smoke_pass") is not True or status.get("reason") != "target_steps":
    raise SystemExit(f"smoke gate failed: {status}")
if int(status.get("last_step", -1)) != expected:
    raise SystemExit(f"smoke ended at unexpected step: {status}")
if manifest.get("model_sha256") != model_sha or manifest.get("input_sha256") != input_sha:
    raise SystemExit(f"smoke provenance mismatch: {manifest}")
if manifest.get("natoms") != 2050 or manifest.get("pbc") != [True, True, True]:
    raise SystemExit(f"smoke structure gate failed: {manifest}")
if target <= expected:
    raise SystemExit("target must exceed smoke endpoint")
PY

export_list="ALL,REP_ROOT=$ROOT,REP_SOURCE=$SMOKE_DIR,REP_RUN=$RUN_DIR,REP_LABEL=$MODEL_ID,REP_TRAIN=$TRAINING_SEED,REP_MD_SEED=$MD_SEED,REP_MODEL=$MODEL_PATH,REP_INPUT=$INPUT_PATH,REP_TARGET=$TARGET_STEP,REP_TEMP=$TEMP,REP_SCRIPT=$CONT_SCRIPT,REP_PYTHON=$PY"
next=$("$SBATCH" --parsable --dependency="afterok:${SLURM_JOB_ID}" --export="$export_list" "$CONT_WRAPPER")
echo "smoke_gate=PASS model_id=$MODEL_ID smoke_job=$SLURM_JOB_ID continuation_job=$next"
