#!/bin/bash -l
# FT-600K / FT-MultiT Cr canonical three-path self-relaxed endpoint-only MLIP-NEB proxy.
# This is not a DFT MEP calculation.
#SBATCH --job-name=ft-interlayer-self-r02
#SBATCH --account=pclancy3
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-17
#SBATCH --chdir=/scratch16/pclancy3/yi/ft_interlayer_sliding_20261025_r02
#SBATCH --output=/scratch16/pclancy3/yi/ft_interlayer_sliding_20261025_r02/logs/self-%A_%a.out
#SBATCH --error=/scratch16/pclancy3/yi/ft_interlayer_sliding_20261025_r02/logs/self-%A_%a.err
set -euo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export PATH=/cm/shared/apps/slurm/current/bin:$PATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
ROOT=/scratch16/pclancy3/yi/ft_interlayer_sliding_20261025_r02
PYTHON=/data/pclancy3/yi/.conda/envs/mace/bin/python
RUNNER=$ROOT/code/self/run_self_job.py
TABLE=$ROOT/jobs/JOBS_SELF.tsv
FREEZE=$ROOT/ACCEPTANCE_FREEZE_FT_SELF_R02.json
HELPER=$ROOT/code/self/export_mlff_neb_iteration_history.py
test -x "$PYTHON"
test -s "$RUNNER"
test -s "$TABLE"
test -s "$FREEZE"
test -s "$HELPER"
mkdir -p "$ROOT/logs"
cd "$ROOT"
exec "$PYTHON" "$RUNNER" \
  --root "$ROOT" \
  --job-table "$TABLE" \
  --index "${SLURM_ARRAY_TASK_ID}" \
  --campaign-id "ft_interlayer_sliding_self_20261025_r02" \
  --schema-version "ft_interlayer_self_v1" \
  --freeze-name "ACCEPTANCE_FREEZE_FT_SELF_R02.json"
