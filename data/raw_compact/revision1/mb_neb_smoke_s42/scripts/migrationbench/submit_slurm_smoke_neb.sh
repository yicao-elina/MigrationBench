#!/bin/bash
#SBATCH --job-name=mb_neb_smoke_s42
#SBATCH --time=00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100
#SBATCH -A pclancy3_gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail

source /data/apps/go.sh
source /data/apps/helpers/singularity.sh
ml openmpi/4.1.6 || true
export OMP_NUM_THREADS=4
module restore || true
source ~/.bashrc
conda activate mace
set -u

echo "host=$(hostname)"
echo "user=$(whoami)"
echo "workdir=$(pwd)"
echo "seed=42"
python --version

python scripts/migrationbench/smoke_migrationbench_pipeline.py
