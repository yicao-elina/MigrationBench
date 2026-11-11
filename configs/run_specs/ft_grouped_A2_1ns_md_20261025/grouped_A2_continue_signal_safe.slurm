#!/bin/bash -l
# Signal-safe, non-overwriting continuation for one exact grouped A2 model.
# The driver writes a checkpoint and returns 99 on B:USR1; this wrapper submits
# a dependent continuation only for that checkpoint/walltime path.
#SBATCH --job-name=ftA2-1ns
#SBATCH --account=pclancy3_gpu
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/scratch16/pclancy3/yi/grouped_A2_ft_1ns_md_20261025/logs/%x-%A.out
#SBATCH --error=/scratch16/pclancy3/yi/grouped_A2_ft_1ns_md_20261025/logs/%x-%A.err

set -euo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

: "${REP_ROOT:?REP_ROOT is required}"
: "${REP_SOURCE:?REP_SOURCE is required}"
: "${REP_RUN:?REP_RUN is required}"
: "${REP_LABEL:?REP_LABEL is required}"
: "${REP_TRAIN:?REP_TRAIN is required}"
: "${REP_MD_SEED:?REP_MD_SEED is required}"
: "${REP_MODEL:?REP_MODEL is required}"
: "${REP_INPUT:?REP_INPUT is required}"
: "${REP_TARGET:?REP_TARGET is required}"
: "${REP_TEMP:?REP_TEMP is required}"
: "${REP_SCRIPT:?REP_SCRIPT is required}"

PY="${REP_PYTHON:-/data/pclancy3/yi/.conda/envs/mace/bin/python}"
SBATCH=/cm/shared/apps/slurm/23.02.2/bin/sbatch
mkdir -p "$REP_ROOT/logs" "$REP_RUN"
test -x "$PY"
test -s "$REP_SCRIPT"
test -s "$REP_MODEL"
test -s "$REP_INPUT"
test -s "$REP_SOURCE/status.json"
test -s "$REP_SOURCE/latest_restart.extxyz"

set +e
child=0
forward_usr1() {
  if [ "$child" -gt 0 ]; then
    kill -USR1 "$child" 2>/dev/null || true
  fi
}
trap forward_usr1 USR1

"$PY" "$REP_SCRIPT" \
  --source-dir "$REP_SOURCE" \
  --input "$REP_INPUT" \
  --model "$REP_MODEL" \
  --model-label "$REP_LABEL" \
  --training-seed "$REP_TRAIN" \
  --md-seed "$REP_MD_SEED" \
  --run-dir "$REP_RUN" \
  --target-step "$REP_TARGET" \
  --temperature-K "$REP_TEMP" \
  --device cuda &
child=$!

# wait(2) can return 128+SIGUSR1 while the child is still alive. Only act once
# the child has actually exited.
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
  export_list="ALL,REP_ROOT=$REP_ROOT,REP_SOURCE=$REP_SOURCE,REP_RUN=$REP_RUN,REP_LABEL=$REP_LABEL,REP_TRAIN=$REP_TRAIN,REP_MD_SEED=$REP_MD_SEED,REP_MODEL=$REP_MODEL,REP_INPUT=$REP_INPUT,REP_TARGET=$REP_TARGET,REP_TEMP=$REP_TEMP,REP_SCRIPT=$REP_SCRIPT,REP_PYTHON=$PY"
  next=$("$SBATCH" --parsable --dependency="afterany:${SLURM_JOB_ID}" --export="$export_list" "$0")
  echo "checkpoint_saved=true model_id=$REP_LABEL job=$SLURM_JOB_ID continuation_job=$next"
  exit 0
elif [ "$rc" -ne 0 ]; then
  echo "scientific_or_provenance_failure=true model_id=$REP_LABEL job=$SLURM_JOB_ID rc=$rc; refusing automatic continuation" >&2
  exit "$rc"
fi

echo "target_reached_or_already_complete=true model_id=$REP_LABEL job=$SLURM_JOB_ID"
