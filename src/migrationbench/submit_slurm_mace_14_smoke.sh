#!/bin/bash
#SBATCH --job-name=mb_mace14_smoke_s42
#SBATCH --time=00:20:00
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

RUN_ROOT="${MIGRATIONBENCH_RUN_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_runs}"
JOB_RUN_DIR="${RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${JOB_RUN_DIR}"
echo "run_root=${RUN_ROOT}"
echo "job_run_dir=${JOB_RUN_DIR}"

python scripts/migrationbench/run_mlff_neb.py \
  --images /data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/1.vdw_corr_DFT_D3/1-4/sb2te3.xyz \
  --out-dir "${JOB_RUN_DIR}/mace_omat_1-4_s42" \
  --calculator mace-model \
  --model-path /data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/MACE-matpes-pbe-omat-ft.model \
  --device cuda \
  --steps 2 \
  --seed 42 \
  --fmax 0.05 \
  --spring-constant 0.1
