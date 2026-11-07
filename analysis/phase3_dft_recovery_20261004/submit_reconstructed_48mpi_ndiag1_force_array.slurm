#!/bin/bash
# Isolated four-case retry after SCF memory growth exhausted 185G at 16/48 MPI.
# Release only after submit_reconstructed_48mpi_ndiag1_smoke.slurm passes.
# QE inputs and scientific settings are unchanged.
#SBATCH --job-name=mb_p3_recon81_force
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --partition=parallel
#SBATCH --account=pclancy3
#SBATCH --mem=185G
# c372 accepted the allocation but failed both assigned tasks before QE at
# the input/run-directory preflight; keep this retry isolated from that node.
#SBATCH --exclude=c372
#SBATCH --array=0-3
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

source /data/apps/go.sh
module load qe/7.3.1-cpu
set -euo pipefail
export OMP_NUM_THREADS=1
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf

INPUT_ROOT=/scratch16/pclancy3/yi/phase3_dft_recovery_20261004_force_inputs
ROOT=/scratch16/pclancy3/yi/phase3_dft_recovery_20261004_retry7_force_48mpi_ndiag1
case "$SLURM_ARRAY_TASK_ID" in
  0) CASE=normal_461800_initial ;;
  1) CASE=bounded_multiT_462000_step464000 ;;
  2) CASE=runaway_foundation_461800_step462120 ;;
  3) CASE=runaway_scratch_461800_step462060 ;;
  *) echo "unsupported array index" >&2; exit 2 ;;
esac

INPUT="$INPUT_ROOT/$CASE/scf.in"
RUN="$ROOT/results/$CASE/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [[ ! -s "$INPUT" || ! -r "$INPUT" ]]; then
  echo "preflight input missing or unreadable: $INPUT" >&2
  ls -ld "$INPUT_ROOT" "$INPUT_ROOT/$CASE" 2>&1 || true
  exit 43
fi
if ! mkdir -p "$RUN/out"; then
  echo "preflight run-directory creation failed: $RUN" >&2
  ls -ld "$ROOT" "$ROOT/results" "$ROOT/results/$CASE" 2>&1 || true
  exit 44
fi
if ! cp "$INPUT" "$RUN/scf.in"; then
  echo "preflight input copy failed: $INPUT -> $RUN/scf.in" >&2
  exit 45
fi
cd "$RUN"
echo "job=$SLURM_JOB_ID array=$SLURM_ARRAY_TASK_ID case=$CASE"
echo "pw=$(command -v pw.x)"
sha256sum scf.in
if grep -qiE 'nspin|starting_magnetization|tot_magnetization|constrained_magnetization' scf.in; then
  echo "magnetic keyword found; refusing to run" >&2
  exit 41
fi
set +e
mpirun -bootstrap fork -np "$SLURM_NTASKS" pw.x -ndiag 1 -inp scf.in > scf.out 2> scf.err
rc=$?
set -e
echo "$rc" > exit_code.txt
grep -E '!    total energy|Forces acting on atoms|Total force|convergence has been achieved|JOB DONE|Error in routine|stopping' scf.out > gate_markers.txt || true
exit "$rc"
