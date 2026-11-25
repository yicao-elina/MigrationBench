#!/bin/bash
# Phase 3 QE environment smoke using one direct pw.x process on one node.
# This isolates QE/UPF/SCF functionality from the unavailable MPI bootstrap.
# No production campaign paths are reused.
#SBATCH --job-name=mb_p3_qe_smoke_direct
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=parallel
#SBATCH --account=pclancy3
#SBATCH --mem=8G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
source /data/apps/go.sh
module load qe/7.3.1-cpu
set -euo pipefail
export OMP_NUM_THREADS=1
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
STAGE_ROOT=/scratch16/pclancy3/yi/migrationbench_phase3_dft_validation_20261002/qe_smoke
RUN_ROOT="$STAGE_ROOT/direct_run_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT/out"
cp "$STAGE_ROOT/scf_smoke.in" "$RUN_ROOT/scf_smoke.in"
cd "$RUN_ROOT"
echo "host=$(hostname)"
echo "pw=$(command -v pw.x)"
ldd "$(command -v pw.x)" | grep -E 'mpi|iomp|mkl' || true
sha256sum scf_smoke.in
pw.x -inp scf_smoke.in > scf_smoke.out
grep -E "!    total energy|Forces acting on atoms|Total force|convergence has been achieved|JOB DONE|Error in routine|stopping" scf_smoke.out || true
