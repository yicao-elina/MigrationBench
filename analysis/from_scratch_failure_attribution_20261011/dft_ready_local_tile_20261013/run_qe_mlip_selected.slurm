#!/bin/bash
# Reduced-cell/local-shell DFT validation batch selected from the completed
# three-model MLIP force-disagreement analysis. One task is one candidate.
#SBATCH --job-name=mb-qe-mlip-sel
#SBATCH --account=pclancy3
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --array=0-19%2
#SBATCH --output=/scratch16/pclancy3/yi/dft_mlip_selected_20261013/logs/%x-%A_%a.out
#SBATCH --error=/scratch16/pclancy3/yi/dft_mlip_selected_20261013/logs/%x-%A_%a.err

set -eo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
set +u
source /data/apps/go.sh
module load qe/7.2
set -u
export PATH=/cm/shared/apps/slurm/current/bin:$PATH

CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-/scratch16/pclancy3/yi/dft_mlip_selected_20261013}"
INPUT_INDEX="$CAMPAIGN_ROOT/selected_input_index.json"
TASK_ID="${SLURM_ARRAY_TASK_ID}"
CANDIDATE_ID=$(python3 -c 'import json,os; print(json.load(open(os.environ["INPUT_INDEX"]))[int(os.environ["TASK_ID"])] ["candidate_id"])')
INPUT="$CAMPAIGN_ROOT/qe_inputs/${CANDIDATE_ID}.scf.in"
RUN_DIR="$CAMPAIGN_ROOT/results/$CANDIDATE_ID"

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
  echo "candidate_id=$CANDIDATE_ID"
  echo "array_task_id=$TASK_ID"
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
mpirun -launcher slurm -n "$SLURM_NTASKS" pw.x -inp input.scf.in > scf.out 2> scf.err
rc=$?
set -e
echo "$rc" > exit_code.txt
exit "$rc"
