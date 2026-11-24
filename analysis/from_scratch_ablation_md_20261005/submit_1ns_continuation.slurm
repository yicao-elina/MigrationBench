#!/bin/bash -l
# Additive wall-time continuation for the selected 1 ns from-scratch runs.
# This is submitted only after the source segment is terminal.  It never
# overwrites r00 or a previously-created continuation segment.
#SBATCH --job-name=ablate-md1ns-cont
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --signal=USR1@120
#SBATCH --array=0-2%2
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail

export MACE_ENV=/data/pclancy3/yi/.conda/envs/mace
export PATH="$MACE_ENV/bin:$PATH"
export PYTHONPATH=/data/pclancy3/yi/.python_pkgs:/data/pclancy3/yi/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

ROOT="${CAMPAIGN_ROOT:-/scratch16/pclancy3/yi/mace_from_scratch_ablation_MD_20261005}"
PY="$MACE_ENV/bin/python"
REGISTRY="$ROOT/registry/selected_1ns.tsv"
CONTINUE="$ROOT/code/continue_ablation_md.py"

ROW=$(awk -v idx="$SLURM_ARRAY_TASK_ID" 'NR == idx + 2 {print}' "$REGISTRY")
IFS=$'\t' read -r FAMILY TRAIN_SEED MODEL MODEL_SHA REASON <<< "$ROW"
BASE="1ns__${FAMILY}__seed${TRAIN_SEED}__mdseed20261005"
SOURCE="$ROOT/runs/${BASE}__r00"

if [ ! -f "$SOURCE/status.json" ] || [ ! -f "$SOURCE/checkpoint.extxyz" ]; then
  echo "source segment is incomplete: $SOURCE" >&2
  exit 42
fi

read -r SOURCE_STATUS SOURCE_STEP TARGET_STEP < <("$PY" - "$SOURCE/status.json" <<'PYCODE'
import json, sys
p = json.load(open(sys.argv[1]))
print(p.get("status", "missing"), int(p.get("last_step", -1)), int(p.get("target_step", 1000000)))
PYCODE
)

case "$SOURCE_STATUS" in
  complete)
    echo "source already complete at step $SOURCE_STEP"
    exit 0
    ;;
  failed)
    echo "source has a scientific failure at step $SOURCE_STEP; refusing continuation" >&2
    exit 42
    ;;
esac

# Resume the first incomplete continuation segment if one exists; otherwise
# allocate the next segment name.  A segment's original start step is retained
# in its manifest so a restart does not create a second file with a new origin.
OUT=""
SEGMENT_START=""
for idx in $(seq 1 99); do
  candidate="$ROOT/runs/${BASE}__r$(printf '%02d' "$idx")"
  if [ -f "$candidate/status.json" ]; then
    state=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", "missing"))' "$candidate/status.json")
    if [ "$state" != "complete" ] && [ "$state" != "failed" ] && [ "$state" != "target_steps" ]; then
      OUT="$candidate"
      SEGMENT_START=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("start_step", -1))' "$candidate/run_manifest.json")
      break
    fi
  elif [ -z "$OUT" ]; then
    OUT="$candidate"
    SEGMENT_START="$SOURCE_STEP"
    break
  fi
done

if [ -z "$OUT" ] || [ -z "$SEGMENT_START" ] || [ "$SEGMENT_START" -lt 0 ]; then
  echo "could not allocate/resume an additive continuation segment" >&2
  exit 42
fi

set +e
child=0
forward_signal() {
  if [ "$child" -gt 0 ]; then
    kill -USR1 "$child" 2>/dev/null || true
  fi
}
trap forward_signal USR1 TERM
"$PY" "$CONTINUE" \
  --checkpoint "$SOURCE/checkpoint.extxyz" \
  --model "$MODEL" \
  --output "$OUT" \
  --family "$FAMILY" \
  --train-seed "$TRAIN_SEED" \
  --md-seed 20261005 \
  --start-step "$SEGMENT_START" \
  --target-step "$TARGET_STEP" \
  --write-interval 200 \
  --temperature 600 \
  --friction 0.01 \
  --timestep-fs 1.0 \
  --device cuda &
child=$!
wait "$child"
rc=$?
trap - USR1 TERM
set -e

if [ "$rc" -eq 99 ]; then
  next=$(sbatch --dependency="afterany:${SLURM_JOB_ID}" --array="${SLURM_ARRAY_TASK_ID}" "$0" | awk '{print $4}')
  echo "checkpointed continuation; submitted dependent segment $next" >&2
  exit 0
elif [ "$rc" -ne 0 ]; then
  echo "scientific/runtime failure; refusing automatic continuation (rc=$rc)" >&2
  exit "$rc"
fi

echo "1 ns target reached for $BASE at $TARGET_STEP steps"
