#!/bin/bash
# Single 82-atom QE smoke test for the PBC-preserving primitive tile.
# This is intentionally separate from the earlier 120-atom array.
# Scientific result requires finite energy/forces, SCF convergence, and JOB DONE.
#SBATCH --job-name=mb-qe82-smoke
#SBATCH --account=pclancy3
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch16/pclancy3/yi/dft_ready_82atom_smoke_20261014/logs/%x-%A.out
#SBATCH --error=/scratch16/pclancy3/yi/dft_ready_82atom_smoke_20261014/logs/%x-%A.err

set -eo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
set +u
source /data/apps/go.sh
module load qe/7.2
set -u
export PATH=/cm/shared/apps/slurm/current/bin:$PATH

CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/scratch16/pclancy3/yi/dft_ready_82atom_smoke_20261014}"
INPUT="$CAMPAIGN_ROOT/input/dft_ready_primitive_tile.scf.in"
RUN_DIR="$CAMPAIGN_ROOT/results/primitive_tile"

case "$CAMPAIGN_ROOT" in /scratch16/pclancy3/yi/*) ;; *) exit 2 ;; esac
case "$INPUT" in /scratch16/pclancy3/yi/*) ;; *) exit 2 ;; esac
test -s "$INPUT"
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
  echo "candidate_id=primitive_tile_82atom"
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
mpirun -bootstrap slurm -n "$SLURM_NTASKS" pw.x -inp input.scf.in > scf.out 2> scf.err
rc=$?
set -e
echo "$rc" > exit_code.txt
exit "$rc"
