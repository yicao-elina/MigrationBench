#!/bin/bash -l
# Canonical 10-model fixed-path interlayer sliding array.
#SBATCH --job-name=mbf10fix
#SBATCH --time=04:00:00
#SBATCH --partition=parallel
#SBATCH -A pclancy3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-19%4
#SBATCH --output=/scratch16/pclancy3/yi/foundation_interlayer_sliding_10model_20261024/logs/fixed-%A_%a.out
#SBATCH --error=/scratch16/pclancy3/yi/foundation_interlayer_sliding_10model_20261024/logs/fixed-%A_%a.err

set -euo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export PATH=/cm/shared/apps/slurm/current/bin:$PATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
ROOT=/scratch16/pclancy3/yi/foundation_interlayer_sliding_10model_20261024
PYTHON=/data/pclancy3/yi/.conda/envs/automatbench/bin/python
mkdir -p "${ROOT}/logs"
exec "${PYTHON}" "${ROOT}/code/fixed/run_fixed_job.py" --root "${ROOT}" --job-table "${ROOT}/jobs/JOBS_FIXED.tsv" --index "${SLURM_ARRAY_TASK_ID}"
