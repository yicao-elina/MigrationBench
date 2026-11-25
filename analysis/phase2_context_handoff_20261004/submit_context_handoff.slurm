#!/bin/bash -l
# Additive context handoff for retained neighboring snapshots.
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

set -euo pipefail
export PYTHONPATH=/data/pclancy3/yi/.python_pkgs:/data/pclancy3/yi/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY=/data/pclancy3/yi/.conda/envs/mace/bin/python
ROOT=/scratch16/pclancy3/yi/phase2_context_handoff_20261004_retry1
case "$SLURM_ARRAY_TASK_ID" in
  0) SNAP=460000; MODEL=foundation-omat; MODEL_PATH=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/finetuned_MACE_compiled.model ;;
  1) SNAP=460000; MODEL=from-scratch; MODEL_PATH=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE_models_l1_0802/mace_l1_0802_compiled.model ;;
  2) SNAP=460000; MODEL=naive-fine-tuning; MODEL_PATH=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_600K/finetuned_MACE_multihead0804_compiled.model ;;
  3) SNAP=460000; MODEL=multi_T-fine-tuning; MODEL_PATH=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_Multi_T/finetuned_MACE_multihead0804.model ;;
  4) SNAP=462200; MODEL=foundation-omat; MODEL_PATH=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/finetuned_MACE_compiled.model ;;
  5) SNAP=462200; MODEL=from-scratch; MODEL_PATH=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE_models_l1_0802/mace_l1_0802_compiled.model ;;
  6) SNAP=462200; MODEL=naive-fine-tuning; MODEL_PATH=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_600K/finetuned_MACE_multihead0804_compiled.model ;;
  7) SNAP=462200; MODEL=multi_T-fine-tuning; MODEL_PATH=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_Multi_T/finetuned_MACE_multihead0804.model ;;
  *) echo "unsupported array index" >&2; exit 2 ;;
esac
INPUT="$ROOT/inputs/snapshot_${SNAP}.extxyz"
OUTPUT="$ROOT/runs/snapshot_${SNAP}/model_${MODEL}"
test -s "$INPUT"
test -r "$MODEL_PATH"
test ! -e "$OUTPUT"
exec "$PY" "$ROOT/handoff_md.py" --input "$INPUT" --model "$MODEL_PATH" \
  --output "$OUTPUT" --start-step "$SNAP" --steps 2000 --interval 10 \
  --temperature 600 --friction 0.01 --timestep-fs 1.0 --device cuda
