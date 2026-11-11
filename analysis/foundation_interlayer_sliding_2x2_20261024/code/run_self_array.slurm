#!/bin/bash -l
# Foundation interlayer-sliding self-relaxed endpoint-only IDPP NEB array.
#SBATCH --job-name=mbsx2x2
#SBATCH --time=08:00:00
#SBATCH --partition=parallel
#SBATCH -A pclancy3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-5%3
#SBATCH --output=/scratch16/pclancy3/yi/foundation_interlayer_sliding_2x2_20261024/logs/self-%A_%a.out
#SBATCH --error=/scratch16/pclancy3/yi/foundation_interlayer_sliding_2x2_20261024/logs/self-%A_%a.err

set -euo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export PATH=/cm/shared/apps/slurm/current/bin:$PATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHON=/data/pclancy3/yi/.conda/envs/automatbench/bin/python
ROOT=/scratch16/pclancy3/yi/foundation_interlayer_sliding_2x2_20261024
"${PYTHON}" "${ROOT}/code/self/run_self_job.py" \
  --root "${ROOT}" --job-table "${ROOT}/jobs/JOBS_SELF.tsv" \
  --index "${SLURM_ARRAY_TASK_ID}"
