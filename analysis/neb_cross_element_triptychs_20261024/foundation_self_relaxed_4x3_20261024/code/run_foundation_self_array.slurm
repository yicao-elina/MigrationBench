#!/bin/bash -l
# Foundation self-relaxed endpoint-only IDPP NEB: 4 elements x 3 paths x 9 models.
#SBATCH --job-name=mbfself
#SBATCH --time=08:00:00
#SBATCH --partition=parallel
#SBATCH -A pclancy3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-107%4
#SBATCH --output=/scratch16/pclancy3/yi/foundation_self_relaxed_4x3_20261024/logs/self-%A_%a.out
#SBATCH --error=/scratch16/pclancy3/yi/foundation_self_relaxed_4x3_20261024/logs/self-%A_%a.err

set -euo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export PATH=/cm/shared/apps/slurm/current/bin:$PATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHON=/data/pclancy3/yi/.conda/envs/automatbench/bin/python
ROOT=/scratch16/pclancy3/yi/foundation_self_relaxed_4x3_20261024
"${PYTHON}" "${ROOT}/code/run_one_job.py" \
  --root "${ROOT}" --job-table "${ROOT}/JOB_TABLE_IDPP.tsv" \
  --index "${SLURM_ARRAY_TASK_ID}"

