#!/bin/bash -l
# One isolated 2 ps handoff; caller must provide INPUT, MODEL, OUTPUT, START_STEP.
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
: "${INPUT:?}"; : "${MODEL:?}"; : "${OUTPUT:?}"; : "${START_STEP:?}"; : "${SCRIPT:?}"
export PYTHONPATH=/data/pclancy3/yi/.python_pkgs:/data/pclancy3/yi/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY=/data/pclancy3/yi/.conda/envs/mace/bin/python
exec "$PY" "$SCRIPT" --input "$INPUT" --model "$MODEL" --output "$OUTPUT" --start-step "$START_STEP" --steps 2000 --interval 10 --temperature 600 --friction 0.01 --timestep-fs 1.0 --device cuda
