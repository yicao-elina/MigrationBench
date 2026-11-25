#!/bin/bash -l
# Four-model local-cell force/energy diagnostic; additive and non-formal.
#SBATCH --job-name=mb_p3_recon_mace
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail

export PYTHONPATH=/data/pclancy3/yi/.python_pkgs:/data/pclancy3/yi/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PY=/data/pclancy3/yi/.conda/envs/mace/bin/python
ROOT=/scratch16/pclancy3/yi/phase3_dft_recovery_20261004
mkdir -p "$ROOT/mace_results"
exec "$PY" "$ROOT/evaluate_reconstructed_mace.py" \
  --input-root "$ROOT/inputs" \
  --output "$ROOT/mace_results/mace_reconstructed.json" \
  --device cuda
