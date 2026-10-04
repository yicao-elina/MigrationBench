#!/bin/bash
# Additive local-cell DFT reconstruction; never overwrite the formal campaign.
#SBATCH --job-name=mb_p3_recon81
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --partition=parallel
#SBATCH --account=pclancy3
#SBATCH --mem=160G
#SBATCH --array=0-3
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

source /data/apps/go.sh
module load qe/7.3.1-cpu
set -euo pipefail
export OMP_NUM_THREADS=1
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

ROOT=/scratch16/pclancy3/yi/phase3_dft_recovery_20261004
case "$SLURM_ARRAY_TASK_ID" in
  0) CASE=normal_461800_initial ;;
  1) CASE=bounded_multiT_462000_step464000 ;;
  2) CASE=runaway_foundation_461800_step462120 ;;
  3) CASE=runaway_scratch_461800_step462060 ;;
  *) echo "unsupported array index" >&2; exit 2 ;;
esac

INPUT="$ROOT/inputs/$CASE/scf.in"
RUN="$ROOT/results/$CASE/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
test -s "$INPUT"
mkdir -p "$RUN/out"
cp "$INPUT" "$RUN/scf.in"
cd "$RUN"
echo "job=$SLURM_JOB_ID array=$SLURM_ARRAY_TASK_ID case=$CASE"
echo "pw=$(command -v pw.x)"
sha256sum scf.in
if grep -qiE 'nspin|starting_magnetization|tot_magnetization|constrained_magnetization' scf.in; then
  echo "magnetic keyword found; refusing to run" >&2
  exit 41
fi
set +e
mpirun -bootstrap slurm -np "$SLURM_NTASKS" pw.x -inp scf.in > scf.out 2> scf.err
rc=$?
set -e
echo "$rc" > exit_code.txt
grep -E '!    total energy|Forces acting on atoms|Total force|convergence has been achieved|JOB DONE|Error in routine|stopping' scf.out > gate_markers.txt || true
exit "$rc"
