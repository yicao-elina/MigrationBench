#!/bin/bash -l
# Scratch Lmax1 fixed-path full matrix: 6 models x 2 systems = 12 rows.
# All computation and output paths are explicitly under scratch16.
#SBATCH --job-name=mb-scr-l1-fix12
#SBATCH --account=pclancy3
#SBATCH --partition=parallel
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --chdir=/scratch16/pclancy3/yi/scratch_lmax1_interlayer_sliding_20261025_r01
#SBATCH --output=/scratch16/pclancy3/yi/scratch_lmax1_interlayer_sliding_20261025_r01/logs/fixed-full-%A_%a.out
#SBATCH --error=/scratch16/pclancy3/yi/scratch_lmax1_interlayer_sliding_20261025_r01/logs/fixed-full-%A_%a.err

set -euo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

ROOT=/scratch16/pclancy3/yi/scratch_lmax1_interlayer_sliding_20261025_r01
PYTHON=/data/pclancy3/yi/.conda/envs/mace/bin/python
SCRIPT=$ROOT/code/scratch_lmax1_g4_fixed_smoke.py
TABLE=$ROOT/jobs/JOBS_FIXED_FULL.tsv

test -x "$PYTHON"
test -s "$SCRIPT"
test -s "$TABLE"
mkdir -p "$ROOT/logs"
cd "$ROOT"
exec "$PYTHON" "$SCRIPT" --root "$ROOT" --job-table "$TABLE" --index "${SLURM_ARRAY_TASK_ID}"
