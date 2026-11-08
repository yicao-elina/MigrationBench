#!/bin/bash -l
# Compute-node loader/head smoke for representative default, MH, and MatPES checkpoints.
#SBATCH --job-name=mbf10smoke
#SBATCH --time=00:20:00
#SBATCH --partition=parallel
#SBATCH -A pclancy3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-2
#SBATCH --output=/scratch16/pclancy3/yi/foundation_interlayer_sliding_10model_20261024/logs/smoke-%A_%a.out
#SBATCH --error=/scratch16/pclancy3/yi/foundation_interlayer_sliding_10model_20261024/logs/smoke-%A_%a.err

set -euo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export PATH=/cm/shared/apps/slurm/current/bin:$PATH
ROOT=/scratch16/pclancy3/yi/foundation_interlayer_sliding_10model_20261024
PYTHON=/data/pclancy3/yi/.conda/envs/automatbench/bin/python
ENDPOINT="${ROOT}/inputs/endpoints/1.bulk/initial.extxyz"
case "${SLURM_ARRAY_TASK_ID}" in
  0) MODEL=/scratch16/pclancy3/yi/foundation_grouped_benchmark_20261001/models/mace-mp-0b3-medium.model; HEAD=;;
  1) MODEL=/scratch16/pclancy3/yi/foundation_grouped_benchmark_20261001/models/mace-mh-0.model; HEAD=omat_pbe;;
  2) MODEL=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/X-FORCE/xforce/models/Foundation-MACE-matpes-pbe-omat-ft.model; HEAD=;;
esac
exec "${PYTHON}" "${ROOT}/code/model_smoke.py" --model "${MODEL}" --head "${HEAD}" --endpoint "${ENDPOINT}"
