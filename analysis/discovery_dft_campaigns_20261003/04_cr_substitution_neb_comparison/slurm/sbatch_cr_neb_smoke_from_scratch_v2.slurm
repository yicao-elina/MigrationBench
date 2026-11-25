#!/bin/bash
# Cr-substitution NEB infrastructure smoke: explicit from-scratch run.
# This smoke deliberately does not require a restart artifact; it verifies
# the Cr input, node-local execution, 10-image initialization, finite output,
# and clean QE termination before any production path is released.
#SBATCH -A pclancy3_lab
#SBATCH --partition=azgpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=11
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=cr-neb-cr-smoke
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

ROOT="${CR_NEB_ROOT:?set CR_NEB_ROOT to the Cr campaign root}"
CASE="${CR_NEB_CASE:?set CR_NEB_CASE to the canonical Cr path id}"
INPUT="${CR_NEB_INPUT:?set CR_NEB_INPUT to the canonical neb.in}"
SIF="${QE_SIF:-/azwestus3/pclancy3_lab/ycao73/qe-7.3.1.sif}"

RESULT="$ROOT/results_smoke/${CASE}_${SLURM_JOB_ID}_from_scratch"
LOCAL_BASE="${SLURM_TMPDIR:-${LOCAL_SCRATCH:-/local/$USER/$SLURM_JOB_ID}}"
SCRATCH="$LOCAL_BASE/mb_cr_neb_${CASE}_${SLURM_JOB_ID}"
QE_OUT="$SCRATCH/qe_out"
TMP_ROOT="$SCRATCH/tmp"
mkdir -p "$RESULT" "$QE_OUT" "$TMP_ROOT"

export TMPDIR="$TMP_ROOT"
export OMPI_MCA_orte_tmpdir_base="$TMP_ROOT"
export OMPI_MCA_prte_tmpdir_base="$TMP_ROOT"
export SINGULARITY_BIND="${SINGULARITY_BIND:-}"
source /azwestus3/pclancy3_lab/singularity.sh

cp "$INPUT" "$RESULT/source_neb.in"
cp "${INPUT%/*}/pw_1.in" "$RESULT/source_pw_1.in"
cp "${INPUT%/*}/pw_2.in" "$RESULT/source_pw_2.in"
cp "$INPUT" "$SCRATCH/neb.in"
cp "${INPUT%/*}/pw_1.in" "$SCRATCH/pw_1.in"
cp "${INPUT%/*}/pw_2.in" "$SCRATCH/pw_2.in"

# Force a clean, bounded smoke without changing the canonical input artifact.
for f in "$SCRATCH/neb.in" "$SCRATCH/pw_1.in" "$SCRATCH/pw_2.in"; do
  sed -i -E "s/restart_mode[[:space:]]*=.*/restart_mode = 'from_scratch',/; s/nstep_path[[:space:]]*=.*/nstep_path = 2,/; s#outdir[[:space:]]*=.*#outdir = '$QE_OUT',#; s/disk_io[[:space:]]*=.*/disk_io = 'none',/" "$f"
done
cp "$SCRATCH/neb.in" "$RESULT/neb.in"
cp "$SCRATCH/pw_1.in" "$RESULT/pw_1.in"
cp "$SCRATCH/pw_2.in" "$RESULT/pw_2.in"

sha256sum "$RESULT/source_neb.in" "$RESULT/source_pw_1.in" "$RESULT/source_pw_2.in" \
  "$RESULT/neb.in" "$RESULT/pw_1.in" "$RESULT/pw_2.in" > "$RESULT/input_sha256.txt"

{
  date
  hostname
  echo "neb_id=$CASE"
  echo "job_id=$SLURM_JOB_ID"
  echo "run_kind=smoke_from_scratch"
  echo "campaign_root=$ROOT"
  echo "result=$RESULT"
  echo "scratch=$SCRATCH"
  echo "scratch_source=${SLURM_TMPDIR:+SLURM_TMPDIR}${SLURM_TMPDIR:-${LOCAL_SCRATCH:+LOCAL_SCRATCH}}"
  echo "container=$SIF"
  sha256sum "$SIF"
  sha256sum /home/ycao73/ONCVPseudoPack/SG15_v1.2/UPF/PBE/Cr_PBE_FR.SG15v1.2.UPF
  df -h "$SCRATCH"
  df -i "$SCRATCH"
} > "$RESULT/runtime_provenance.txt"

set +e
singularity exec --nv --cleanenv "$SIF" neb.x -in "$SCRATCH/neb.in" > "$SCRATCH/neb.out" 2> "$SCRATCH/neb.err"
RC=$?
set -e
cp "$SCRATCH/neb.out" "$RESULT/neb.out"
cp "$SCRATCH/neb.err" "$RESULT/neb.err"
printf '%s\n' "$RC" > "$RESULT/exit_code.txt"

JOB_DONE=0
IMAGE_COUNT=0
FINITE=0
BAD=0
grep -q 'JOB DONE' "$RESULT/neb.out" && JOB_DONE=1 || true
grep -Eq 'num_of_images[[:space:]]*=.*10|num_of_images[[:space:]]*=.*10' "$RESULT/neb.out" && IMAGE_COUNT=10 || true
grep -Eiq 'Error in routine|convergence NOT achieved|CANNOT CREATE FIFO|errno 524|PMIx|NAN|INF|floating point exception' "$RESULT/neb.out" "$RESULT/neb.err" && BAD=1 || true
grep -Eq '(^|[[:space:]])[-+]?[0-9]+\.[0-9]+([Ee][-+]?[0-9]+)?([[:space:]]|$)' "$RESULT/neb.out" && FINITE=1 || true

{
  echo "rc=$RC"
  echo "job_done=$JOB_DONE"
  echo "image_count=$IMAGE_COUNT"
  echo "finite_numeric_output=$FINITE"
  echo "fatal_markers=$BAD"
  if [[ "$RC" -eq 0 && "$JOB_DONE" -eq 1 && "$IMAGE_COUNT" -eq 10 && "$FINITE" -eq 1 && "$BAD" -eq 0 ]]; then
    echo 'gate=PASS'
  else
    echo 'gate=FAIL'
  fi
} > "$RESULT/gate_markers.txt"

# Retain scratch for every smoke; it is diagnostic evidence until the gate is audited.
du -sb "$SCRATCH" > "$RESULT/scratch_usage_bytes_before_cleanup.txt" || true
echo 'scratch_cleanup=preserved_for_diagnosis' >> "$RESULT/gate_markers.txt"
exit "$RC"
