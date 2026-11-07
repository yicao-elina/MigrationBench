#!/bin/bash
# Cr NEB smoke v3: explicit from-scratch, node-local QE/MPI temporary I/O.
# A short nstep_path=2 run may make QE return 1 after a complete max-step
# diagnostic; that raw QE rc is preserved, while the wrapper gate is 0 only
# when JOB DONE, finite ten-image output, and clean runtime markers pass.
#SBATCH --job-name=cr-neb-from-scratch-v3
#SBATCH --time=12:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --partition=azgpuh200
#SBATCH -A pclancy3_lab
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003/logs/neb-smoke-v3-%j.out
#SBATCH --error=/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003/logs/neb-smoke-v3-%j.err

set -euo pipefail
CAMPAIGN=/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003
ROOT="$CAMPAIGN/cr_substitution_neb_comparison"
NEB_ID="${NEB_ID:-2L-Cu__neb-QLs}"
INPUT_DIR="$ROOT/inputs/$NEB_ID"
SIF=/azwestus3/pclancy3_lab/ycao73/qe-7.3.1.sif
RESULT="$ROOT/results_smoke/${NEB_ID}_${SLURM_JOB_ID}_from_scratch_v3"
LOCAL_BASE="${SLURM_TMPDIR:-${LOCAL_SCRATCH:-/local/$USER/$SLURM_JOB_ID}}"
SCRATCH="$LOCAL_BASE/mb_cr_neb_${NEB_ID}_${SLURM_JOB_ID}_v3"
QE_OUT="$SCRATCH/qe_out"
TMP_ROOT="$SCRATCH/tmp"
mkdir -p "$RESULT" "$QE_OUT" "$TMP_ROOT"
export TMPDIR="$TMP_ROOT"
export APPTAINERENV_TMPDIR="$TMP_ROOT"
export APPTAINERENV_OMPI_MCA_prte_tmpdir_base="$TMP_ROOT"
export APPTAINERENV_OMPI_MCA_orte_tmpdir_base="$TMP_ROOT"
export SINGULARITY_BIND="${SINGULARITY_BIND:-}"
source /projects/pclancy3_lab/singularity.sh

for f in neb.in pw_1.in pw_2.in; do test -s "$INPUT_DIR/$f"; cp "$INPUT_DIR/$f" "$RESULT/source_$f"; done
cp "$INPUT_DIR/neb.in" "$SCRATCH/neb.in"
for f in pw_1.in pw_2.in; do cp "$INPUT_DIR/$f" "$SCRATCH/$f"; done

awk '
  BEGIN { in_path=0; have_restart=0; have_nstep=0 }
  /^&PATH/ { in_path=1 }
  in_path && /restart_mode[[:space:]]*=/ { print "  restart_mode = '\''from_scratch'\''"; have_restart=1; next }
  in_path && /nstep_path[[:space:]]*=/ { print "  nstep_path = 2"; have_nstep=1; next }
  in_path && /^\// { if (!have_restart) print "  restart_mode = '\''from_scratch'\''"; if (!have_nstep) print "  nstep_path = 2"; in_path=0 }
  { print }
' "$SCRATCH/neb.in" > "$SCRATCH/neb.in.tmp"
mv "$SCRATCH/neb.in.tmp" "$SCRATCH/neb.in"
for f in pw_1.in pw_2.in; do
  sed -E "s|outdir[[:space:]]*=.*|  outdir = '$QE_OUT',|" "$SCRATCH/$f" > "$SCRATCH/$f.tmp"
  mv "$SCRATCH/$f.tmp" "$SCRATCH/$f"
done
for f in neb.in pw_1.in pw_2.in; do
  if grep -q 'disk_io' "$SCRATCH/$f"; then sed -i -E "s/disk_io[[:space:]]*=.*/disk_io = 'none',/" "$SCRATCH/$f"; fi
done
cp "$SCRATCH/neb.in" "$RESULT/neb.in"; cp "$SCRATCH/pw_1.in" "$RESULT/pw_1.in"; cp "$SCRATCH/pw_2.in" "$RESULT/pw_2.in"
sha256sum "$RESULT/neb.in" "$RESULT/pw_1.in" "$RESULT/pw_2.in" > "$RESULT/input_sha256.txt"
{
  date; hostname; echo "job_id=$SLURM_JOB_ID"; echo "neb_id=$NEB_ID"; echo "run_mode=from_scratch"; echo "nstep_path=2"; echo "scratch=$SCRATCH"; echo "qe_outdir=$QE_OUT"; echo "container=$SIF"; sha256sum "$SIF"; df -h "$SCRATCH"; df -ih "$SCRATCH"; sha256sum "$RESULT/neb.in" "$RESULT/pw_1.in" "$RESULT/pw_2.in"
} > "$RESULT/runtime_provenance.txt"

set +e
singularity exec --nv --cleanenv "$SIF" neb.x -in "$SCRATCH/neb.in" > "$SCRATCH/neb.out" 2> "$SCRATCH/neb.err"
QE_RC=$?
set -e
cp "$SCRATCH/neb.out" "$RESULT/neb.out"; cp "$SCRATCH/neb.err" "$RESULT/neb.err"
job_done=0; grep -q 'JOB DONE' "$RESULT/neb.out" && job_done=1 || true
image_rows=$(grep -E '^[[:space:]]*[0-9]+[[:space:]]+[-+]?[0-9][0-9.eEdD+-]*([[:space:]]|$)' "$RESULT/neb.out" | wc -l)
barriers=$(grep -Ec 'activation energy \((->|<-)\)[[:space:]]*=[[:space:]]*[-+]?[0-9][0-9.eEdD+-]*' "$RESULT/neb.out" || true)
max_steps=0; grep -q 'neb: reached the maximum number of steps' "$RESULT/neb.out" && max_steps=1 || true
fatal=0; grep -Eiq 'Error in routine|stopping|MPI_Init|FIO|errno|nan|inf|CANNOT CREATE FIFO|convergence NOT achieved' "$RESULT/neb.out" "$RESULT/neb.err" && fatal=1 || true
gate=0
[[ "$QE_RC" == 0 || ( "$QE_RC" == 1 && "$max_steps" == 1 ) ]] || gate=1
[[ "$job_done" == 1 && "$image_rows" -ge 10 && "$barriers" -ge 2 && "$fatal" == 0 ]] || gate=1
{
  echo "qe_rc=$QE_RC"; echo "job_done=$job_done"; echo "image_numeric_rows=$image_rows"; echo "barrier_lines=$barriers"; echo "max_steps=$max_steps"; echo "fatal_markers=$fatal"; echo "wfc_bytes=0"; echo "restart_input=0"; echo "gate=$gate"; echo "scratch_cleanup=preserved_for_diagnosis"
} >> "$RESULT/runtime_provenance.txt"
cp "$RESULT/runtime_provenance.txt" "$RESULT/gate_markers.txt"
echo "$gate" > "$RESULT/exit_code.txt"
du -sb "$SCRATCH" > "$RESULT/scratch_usage_bytes_before_cleanup.txt" || true
exit "$gate"
