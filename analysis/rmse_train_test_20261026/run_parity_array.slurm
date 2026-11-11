#!/bin/bash
#SBATCH --job-name=mb-rmse-parity
#SBATCH --partition=gpu
#SBATCH --array=0-23%8
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
source /etc/profile
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
/data/pclancy3/yi/.conda/envs/mace/bin/python evaluate_scatter.py --manifest parity_manifest.tsv --output-dir results
