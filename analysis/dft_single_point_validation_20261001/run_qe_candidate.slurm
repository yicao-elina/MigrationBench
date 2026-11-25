#!/bin/bash
# Isolated QE 7.2 SCF single-point runner.  This is intentionally separate
# from the existing 47-endpoint relaxation scripts and roots.
#SBATCH --job-name=mb-dft-sp-20261001
#SBATCH --account=pclancy3
#SBATCH --partition=parallel
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch16/pclancy3/yi/dft_single_point_validation_20261001/logs/%x-%j.out
#SBATCH --error=/scratch16/pclancy3/yi/dft_single_point_validation_20261001/logs/%x-%j.err

set -eo pipefail
# Make the Rockfish Slurm configuration explicit before loading site helpers.
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
MPI_LAUNCHER=mpirun
command -v "$MPI_LAUNCHER" >/dev/null
# Rockfish's batch environment can inherit nounset.  go.sh references
# optional variables (including RF_CONDA_SITE), so load it with nounset off.
set +u
source /data/apps/go.sh
module load qe/7.2
set -u

: "${CANDIDATE_ID:?set CANDIDATE_ID=cand_XXXX}"
CAMPAIGN_ROOT=/scratch16/pclancy3/yi/dft_single_point_validation_20261001
INPUT="$CAMPAIGN_ROOT/qe_inputs/${CANDIDATE_ID}.scf.in"
RUN_DIR="$CAMPAIGN_ROOT/results/${CANDIDATE_ID}"

case "$CAMPAIGN_ROOT" in /scratch16/pclancy3/yi/*) ;; *) exit 2 ;; esac
case "$INPUT" in /scratch16/pclancy3/yi/*) ;; *) exit 2 ;; esac

mkdir -p "$RUN_DIR/out" "$RUN_DIR/tmp"
cp "$INPUT" "$RUN_DIR/input.scf.in"
cd "$RUN_DIR"
export TMPDIR="$RUN_DIR/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export OMP_NUM_THREADS=1
ulimit -s unlimited || true

{
  echo "campaign_root=$CAMPAIGN_ROOT"
  echo "candidate_id=$CANDIDATE_ID"
  echo "job_id=$SLURM_JOB_ID"
  echo "run_dir=$RUN_DIR"
  echo "slurm_nodes=$SLURM_JOB_NUM_NODES"
  echo "slurm_ntasks=$SLURM_NTASKS"
  echo "tmpdir=$TMPDIR"
  sha256sum input.scf.in
  command -v pw.x
  module list 2>&1
} > runtime_provenance.txt

set +e
# QE 7.2 is linked against Intel MPI on Rockfish.  Use Intel MPI's Slurm
# bootstrap explicitly; Slurm PMI2 launch is incompatible with this build and
# fails before QE starts.
"$MPI_LAUNCHER" -bootstrap slurm -np "$SLURM_NTASKS" pw.x -inp input.scf.in > scf.out 2> scf.err
rc=$?
set -e
echo "$rc" > exit_code.txt
exit "$rc"
