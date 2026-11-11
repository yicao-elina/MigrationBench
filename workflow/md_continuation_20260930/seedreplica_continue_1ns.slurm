#!/bin/bash -l
#SBATCH --job-name=seedreplica-1ns
#SBATCH --array=0-3%1
#SBATCH --account=pclancy3_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --output=/scratch16/pclancy3/yi/mace_seedreplica_1ns_20261005_ft600k_multit/logs/%A_%a.out
#SBATCH --error=/scratch16/pclancy3/yi/mace_seedreplica_1ns_20261005_ft600k_multit/logs/%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=4
PY=/data/pclancy3/yi/.conda/envs/mace/bin/python
ROOT=/scratch16/pclancy3/yi/mace_seedreplica_1ns_20261005_ft600k_multit
INPUT=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/AI4MAT25/mlff-benchmark-cr-sb2te3/results/MD/3-multi_T-fine-tuning/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz
SCRIPT=/scratch16/pclancy3/yi/mace_seedreplica_1ns_20261005_ft600k_multit/seedreplica_continue_1ns.py

case "${SLURM_ARRAY_TASK_ID}" in
  0) LABEL=FT-600K-seed234; TRAIN=234; MODEL=/scratch16/pclancy3/yi/revision1_migrationbench_runs/ft600k_grouped_A2/retrain_matrix_scratch16_20260927/ft600k_omatft_grouped_seed234/ft600k_omatft_grouped_seed234_compiled.model ;;
  1) LABEL=FT-600K-seed345; TRAIN=345; MODEL=/scratch16/pclancy3/yi/revision1_migrationbench_runs/ft600k_grouped_A2/retrain_matrix_scratch16_20260927/ft600k_omatft_grouped_seed345/ft600k_omatft_grouped_seed345_compiled.model ;;
  2) LABEL=FT-MultiT-seed234; TRAIN=234; MODEL=/scratch16/pclancy3/yi/revision1_migrationbench_runs/multit_seed_matrix_20261001/ft_multit_seed234/ft_multit_seed234_compiled.model ;;
  3) LABEL=FT-MultiT-seed345; TRAIN=345; MODEL=/scratch16/pclancy3/yi/revision1_migrationbench_runs/multit_seed_matrix_20261001/ft_multit_seed345/ft_multit_seed345_compiled.model ;;
  *) echo "invalid array index" >&2; exit 2 ;;
esac

SOURCE="$ROOT/smoke_source/$LABEL"
RUN="$ROOT/$LABEL"

set +e
child=0
forward_signal() {
  if [ "$child" -gt 0 ]; then
    kill -USR1 "$child" 2>/dev/null || true
  fi
}
trap forward_signal USR1 TERM
"$PY" "$SCRIPT" --source-dir "$SOURCE" --input "$INPUT" --model "$MODEL" \
  --model-label "$LABEL" --training-seed "$TRAIN" --md-seed 1100 \
  --run-dir "$RUN" --target-step 1000000 --temperature-K 600 --device cuda &
child=$!
wait "$child"
rc=$?
trap - USR1 TERM
set -e

if [ "$rc" -eq 99 ]; then
  next=$(sbatch --dependency="afterany:${SLURM_JOB_ID}" --array="${SLURM_ARRAY_TASK_ID}" "$0" | awk '{print $4}')
  echo "walltime/stop checkpoint saved; submitted dependent continuation ${next}" >&2
  exit 0
elif [ "$rc" -ne 0 ]; then
  echo "scientific or provenance failure; refusing automatic continuation (rc=$rc)" >&2
  exit "$rc"
fi
