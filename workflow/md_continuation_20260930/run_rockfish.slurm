#!/bin/bash -l
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
export MACE_ENV=/data/pclancy3/yi/.conda/envs/mace
export PATH=$MACE_ENV/bin:$PATH
export PYTHONPATH=/data/pclancy3/yi/.python_pkgs:/data/pclancy3/yi/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$RUN_DIR"
exec "$MACE_ENV/bin/python" /scratch16/pclancy3/yi/mace_four_model_MD_continuation_20260930/continue_md.py \
  --input "$INPUT" --model "$MODEL" --run-dir "$RUN_DIR" --start-step "$START_STEP"
