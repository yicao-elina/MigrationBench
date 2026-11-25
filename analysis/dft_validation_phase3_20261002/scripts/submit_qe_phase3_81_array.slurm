#!/bin/bash
# Phase 3 isolated 81-atom fixed-geometry single-point array.
# No relaxation, NEB, retraining, or reuse of prior QE run directories.
#SBATCH --job-name=mb_p3_qe81_sp
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --partition=parallel
#SBATCH --account=pclancy3
#SBATCH --mem=160G
#SBATCH --array=0-1
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
source /data/apps/go.sh
module load qe/7.3.1-cpu
set -euo pipefail
export OMP_NUM_THREADS=1
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
ROOT=/scratch16/pclancy3/yi/migrationbench_phase3_dft_validation_20261002/qe_81_single_points
case "$SLURM_ARRAY_TASK_ID" in
  0) STEP=461800 ;;
  1) STEP=462000 ;;
  *) echo "unsupported array index" >&2; exit 2 ;;
esac
RUN="$ROOT/step_${STEP}/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RUN/out"
cp "$ROOT/inputs/step_${STEP}/scf.in" "$RUN/scf.in"
cd "$RUN"
echo "host=$(hostname)"
echo "job=$SLURM_JOB_ID array=$SLURM_ARRAY_TASK_ID step=$STEP"
echo "pw=$(command -v pw.x)"
sha256sum scf.in
mpirun -bootstrap slurm -np "$SLURM_NTASKS" pw.x -inp scf.in > scf.out
grep -E "!    total energy|Forces acting on atoms|Total force|convergence has been achieved|JOB DONE|Error in routine|stopping" scf.out || true
