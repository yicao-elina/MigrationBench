#!/bin/bash
# Phase 3 isolated 81-atom fixed-geometry single-point array.
# Direct one-process launch isolates QE from the unavailable MPI bootstrap.
#SBATCH --job-name=mb_p3_qe81_sp_direct
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=parallel
#SBATCH --account=pclancy3
# 32G is intentional: shared/express charge 4G per CPU, so 160G implied
# MinCPUsNode=41 and exceeded the account QOS CPU limit for this 81-atom SP.
#SBATCH --mem=32G
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
RUN="$ROOT/step_${STEP}/direct_job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RUN/out"
cp "$ROOT/inputs/step_${STEP}/scf.in" "$RUN/scf.in"
cd "$RUN"
echo "host=$(hostname)"
echo "job=$SLURM_JOB_ID array=$SLURM_ARRAY_TASK_ID step=$STEP"
echo "pw=$(command -v pw.x)"
ldd "$(command -v pw.x)" | grep -E 'mpi|iomp|mkl' || true
sha256sum scf.in
pw.x -inp scf.in > scf.out
grep -E "!    total energy|Forces acting on atoms|Total force|convergence has been achieved|JOB DONE|Error in routine|stopping" scf.out || true
