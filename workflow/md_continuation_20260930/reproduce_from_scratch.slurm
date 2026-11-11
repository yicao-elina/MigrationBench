#!/bin/bash -l
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=reproduce-%j.out
#SBATCH --error=reproduce-%j.err
set -euo pipefail
export MACE_ENV=/data/pclancy3/yi/.conda/envs/mace
export PATH=$MACE_ENV/bin:$PATH
export PYTHONPATH=/data/pclancy3/yi/.python_pkgs:/data/pclancy3/yi/src
cd "$OUT_DIR"
exec "$MACE_ENV/bin/python" /scratch16/pclancy3/yi/mace_four_model_MD_continuation_20260930/reproduce_from_scratch.py \
  --input "$INPUT" --model "$MODEL" --output "$OUT_DIR/run" --steps 20000 --seed 20261003 --device cuda
