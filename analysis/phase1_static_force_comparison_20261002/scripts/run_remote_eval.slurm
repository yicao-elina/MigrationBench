#!/bin/bash -l
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
export MACE_ENV=/data/pclancy3/yi/.conda/envs/mace
export PATH="$MACE_ENV/bin:$PATH"
export PYTHONPATH=/data/pclancy3/yi/.python_pkgs:/data/pclancy3/yi/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ROOT=/scratch16/pclancy3/yi/phase1_static_force_comparison_20261002
cd "$ROOT"
exec "$MACE_ENV/bin/python" "$ROOT/run_static_evaluations.py" \
  --snapshots "$ROOT/inputs" \
  --models "$ROOT/models_rockfish.json" \
  --output-dir "$ROOT/results" \
  --device cuda \
  --dtype float64
