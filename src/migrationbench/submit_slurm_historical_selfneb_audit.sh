#!/bin/bash
#SBATCH --job-name=mb_selfneb_audit_s42
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=parallel
#SBATCH -A pclancy3
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail

source /data/apps/go.sh
module restore || true
source ~/.bashrc
conda activate mace
set -u

SEED=42
CODE_ROOT="/scratch16/pclancy3/yi/revision1_migrationbench_pipeline"
SOURCE_ROOT="/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806-NEB/0-foundation-omat"
MODEL="/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/MACE-matpes-pbe-omat-ft.model"
RUN_DIR="/scratch16/pclancy3/yi/revision1_migrationbench_runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

python "${CODE_ROOT}/scripts/migrationbench/audit_historical_mlff_neb.py" \
  --trajectory "${SOURCE_ROOT}/neb_optimization.traj" \
  --optimizer-log "${SOURCE_ROOT}/slurm.10199010" \
  --model "${MODEL}" \
  --output-dir "${RUN_DIR}" \
  --n-images 12 \
  --device cpu \
  --seed "${SEED}"
