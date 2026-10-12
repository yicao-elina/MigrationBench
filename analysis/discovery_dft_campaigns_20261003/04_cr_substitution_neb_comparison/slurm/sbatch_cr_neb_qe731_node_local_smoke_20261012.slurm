#!/bin/bash
# Independent Cr-NEB restart smoke.  This is not a production result.
# It keeps QE and MPI temporary I/O on Discovery's per-job node-local
# filesystem; the project filesystem stores only auditable text/results.
# Submit with: sbatch --export=ALL,NEB_ID=2L-Cu__neb-QLs <this-file>
# The restart artifact is taken from RESTART_FROM and is never modified.
#
# This wrapper is intentionally separate from the failed project-scratch
# smoke 6887697.  Do not overwrite that result or its preserved scratch.
#SBATCH --job-name=mb_neb_qe731
#SBATCH --time=08:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --partition=azgpuh200
#SBATCH -A pclancy3_lab
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003/logs/neb-smoke-%x-%j.out
#SBATCH --error=/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003/logs/neb-smoke-%x-%j.err

set -euo pipefail
: "${NEB_ID:?submit with --export=ALL,NEB_ID=2L-Cu__neb-QLs}"
[[ "$NEB_ID" =~ ^[A-Za-z0-9_.-]+$ ]]

CAMPAIGN=/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003
ROOT="$CAMPAIGN/cr_substitution_neb_comparison"
INPUT_DIR="${INPUT_ROOT:-$ROOT/inputs}/$NEB_ID"
RUN="$ROOT/results_smoke/${NEB_ID}_${SLURM_JOB_ID}_node_local_restart"
RESTART_FROM="${RESTART_FROM:-$ROOT/results_smoke/2L-Cu__neb-QLs_6887697}"
SIF=/azwestus3/pclancy3_lab/ycao73/qe-7.3.1.sif
PSEUDO=/home/ycao73/ONCVPseudoPack/SG15_v1.2/UPF/PBE
SMOKE_NSTEP="${SMOKE_NSTEP:-2}"
WRAPPER_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
WRAPPER_SHA256="$(sha256sum "$WRAPPER_PATH" | awk '{print $1}')"

[[ "$SMOKE_NSTEP" =~ ^[1-9][0-9]*$ ]]
for f in neb.in pw_1.in pw_2.in; do test -s "$INPUT_DIR/$f"; done
test -r "$SIF"

# A .path file alone is not a QE restart: neb.x also needs the per-image
# QE data/WFC state.  Detect that explicitly; otherwise use the canonical
# ten-image geometry from neb.in and run a genuine from_scratch smoke.
RESTART_FILE="$(find "$RESTART_FROM" -maxdepth 1 -type f -name '*.path' -print -quit 2>/dev/null || true)"
RESTART_IMAGE_COUNT=0
RUN_MODE=from_scratch
if [[ -n "$RESTART_FILE" && -s "$RESTART_FILE" ]]; then
  RESTART_IMAGE_COUNT="$(find "$RESTART_FROM" -type f \( -name 'data-file-schema.xml' -o -name 'data-file.xml' -o -name 'wfc*' \) -print 2>/dev/null | wc -l)"
  if (( RESTART_IMAGE_COUNT >= 10 )); then
    RUN_MODE=restart
  fi
fi

mkdir -p "$RUN"
cd "$RUN"
for f in neb.in pw_1.in pw_2.in; do
  cp "$INPUT_DIR/$f" "source_$f"
done
if [[ -n "$RESTART_FILE" && -s "$RESTART_FILE" ]]; then
  cp "$RESTART_FILE" "initial_$(basename "$RESTART_FILE")"
fi

# Prefer Discovery's per-job LOCAL_SCRATCH; SLURM_TMPDIR is a compatible
# fallback.  /tmp is retained only as a diagnostic portability fallback.
if [[ -n "${LOCAL_SCRATCH:-}" ]]; then
  LOCAL_BASE="$LOCAL_SCRATCH"
  SCRATCH_SOURCE=LOCAL_SCRATCH
elif [[ -n "${SLURM_TMPDIR:-}" ]]; then
  LOCAL_BASE="$SLURM_TMPDIR"
  SCRATCH_SOURCE=SLURM_TMPDIR
else
  LOCAL_BASE="/tmp/${USER}"
  SCRATCH_SOURCE=TMP_FALLBACK
fi
TMP_ROOT="${LOCAL_BASE%/}/mb_neb_${NEB_ID}_${SLURM_JOB_ID}"
QE_OUTDIR="$TMP_ROOT/qe_out"
TMP_DIR="$TMP_ROOT/tmp"
mkdir -p "$QE_OUTDIR" "$TMP_DIR"
chmod 700 "$TMP_ROOT" "$QE_OUTDIR" "$TMP_DIR"

# Make an immutable runtime copy with an explicit restart mode, short smoke
# path, disk_io=none, and node-local QE outdir.
awk -v n="$SMOKE_NSTEP" -v mode="$RUN_MODE" '
  BEGIN { in_path=0; have_restart=0; have_nstep=0 }
  /^[[:space:]]*&PATH/ { in_path=1; print; next }
  in_path && /^[[:space:]]*restart_mode[[:space:]]*=/ { print "  restart_mode = '\''" mode "'\''"; have_restart=1; next }
  in_path && /^[[:space:]]*nstep_path[[:space:]]*=/ { print "  nstep_path = " n; have_nstep=1; next }
  in_path && /^[[:space:]]*\/[[:space:]]*$/ {
    if (!have_restart) print "  restart_mode = '\''" mode "'\''"
    if (!have_nstep) print "  nstep_path = " n
    in_path=0
  }
  { print }
' source_neb.in > neb.in

for f in pw_1.in pw_2.in; do
  sed -E "s|outdir[[:space:]]*=.*|  outdir = '$QE_OUTDIR'|" "source_$f" > "$f"
done
for f in neb.in pw_1.in pw_2.in; do
  awk '
    /^[[:space:]]*&CONTROL/ { in_control=1; have_disk=0 }
    in_control && /^[[:space:]]*disk_io[[:space:]]*=/ { print "  disk_io = '\''none'\''"; have_disk=1; next }
    in_control && /^[[:space:]]*\/[[:space:]]*$/ { if (!have_disk) print "  disk_io = '\''none'\''"; in_control=0 }
    { print }
  ' "$f" > "$f.tmp"
  mv "$f.tmp" "$f"
done

export OMP_NUM_THREADS=11 OMP_STACKSIZE=2G
export TMPDIR="$TMP_DIR"
export APPTAINERENV_TMPDIR="$TMP_DIR"
export APPTAINERENV_TMP_ROOT="$TMP_ROOT"
export APPTAINERENV_QE_OUTDIR="$QE_OUTDIR"
export APPTAINERENV_OMPI_MCA_prte_tmpdir_base="$TMP_DIR"
export APPTAINERENV_OMPI_MCA_orte_tmpdir_base="$TMP_DIR"
export SINGULARITY_BIND="${SINGULARITY_BIND:-}"
source /projects/pclancy3_lab/singularity.sh
if [[ -n "${SINGULARITY_BIND:-}" ]]; then export APPTAINER_BIND="$SINGULARITY_BIND"; fi
unset SINGULARITY_BIND
export APPTAINERENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"
export APPTAINERENV_OMP_STACKSIZE="$OMP_STACKSIZE"
export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}$TMP_ROOT:$TMP_ROOT,$ROOT:$ROOT"

if [[ -n "$RESTART_FILE" && -s "$RESTART_FILE" ]]; then
  RESTART_SHA_BEFORE="$(sha256sum "initial_$(basename "$RESTART_FILE")" | awk '{print $1}')"
else
  RESTART_SHA_BEFORE=NA
fi
{
  date
  hostname
  echo "neb_id=$NEB_ID"
  echo "job_id=$SLURM_JOB_ID"
  echo "run=$RUN"
  echo "restart_from=$RESTART_FROM"
  echo "restart_file=$(basename "$RESTART_FILE")"
  echo "restart_sha_before=$RESTART_SHA_BEFORE"
  echo "run_mode=$RUN_MODE"
  echo "restart_image_state_files=$RESTART_IMAGE_COUNT"
  echo "scratch_root=$TMP_ROOT"
  echo "scratch_source=$SCRATCH_SOURCE"
  echo "qe_outdir=$QE_OUTDIR"
  echo "container=$SIF"
  echo "wrapper=$WRAPPER_PATH"
  echo "wrapper_sha256=$WRAPPER_SHA256"
  echo "qe_version=7.3.1"
  echo "runtime_overrides=restart_mode=$RUN_MODE,nstep_path=$SMOKE_NSTEP,disk_io=none,outdir=node_local"
  echo "mpi_tmpdir=$TMP_DIR"
  echo "launcher=container_direct_neb_x_qe731_nosrun"
  df -h "$TMP_ROOT"
  df -ih "$TMP_ROOT"
  sha256sum source_neb.in source_pw_1.in source_pw_2.in neb.in pw_1.in pw_2.in
  if [[ -n "$RESTART_FILE" && -s "initial_$(basename "$RESTART_FILE")" ]]; then
    sha256sum "initial_$(basename "$RESTART_FILE")"
  fi
  for p in Cr_PBE_FR.SG15v1.2.UPF Sb_PBE_FR.SG15v1.2.UPF Te_PBE_FR.SG15v1.2.UPF; do
    [[ -r "$PSEUDO/$p" ]] && sha256sum "$PSEUDO/$p"
  done
} > runtime_provenance.txt

set +e
singularity exec --nv --cleanenv "$SIF" \
  bash --noprofile --norc -c '
    set -euo pipefail
    export NVHPC=/opt/nvidia/hpc_sdk
    export NVARCH=Linux_x86_64
    export NVCOMPILERS=/opt/nvidia/hpc_sdk
    export PATH=/usr/local/qe/bin:/usr/local/bin:$PATH
    export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/openmpi/openmpi-3.1.5/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/24.3/compilers/lib:${LD_LIBRARY_PATH:-}
    export OMPI_MCA_btl=self
    export OMPI_MCA_pml=ob1
    export OMPI_MCA_mtl=^ofi
    export OMPI_MCA_btl_base_warn_component_unused=0
    export OMPI_MCA_orte_tmpdir_base="$TMPDIR"
    export OMPI_MCA_prte_tmpdir_base="$TMPDIR"
    export OMP_NUM_THREADS=11
    export OMP_STACKSIZE=2G
    export CUDA_LAUNCH_BLOCKING=1
    fifo_probe="$TMPDIR/mb_neb_fifo_probe"
    rm -f "$fifo_probe"
    mkfifo "$fifo_probe"
    rm -f "$fifo_probe"
    test -d "$QE_OUTDIR"
    test -w "$QE_OUTDIR"
    echo "preflight=node_local_fifo_and_qe_outdir_ok"
    echo "launcher=container_direct_neb_x_qe731_nosrun"
    exec neb.x -in neb.in
  ' > neb.out 2> neb.err
rc=$?
set -e

job_done=0
grep -q 'JOB DONE' neb.out && job_done=1 || true
image_dirs="$(find "$QE_OUTDIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
wfc_bytes="$(find "$QE_OUTDIR" -type f \( -name 'wfc*' -o -name '*.wfc*' \) -printf '%s\n' 2>/dev/null | awk '{s+=$1} END {print s+0}')"
# QE's NEB summary contains one numeric row per image with an energy and a
# force/error column. Require all ten rows and reject non-finite markers.
image_numeric_rows="$(grep -E '^[[:space:]]*[0-9]+[[:space:]]+[-+]?[0-9][0-9.eEdD+-]*([[:space:]]|$)' neb.out 2>/dev/null | wc -l)"
finite_image_rows=0
(( image_numeric_rows >= 10 )) && finite_image_rows=1 || true
if [[ -n "$RESTART_FILE" && -s "$RESTART_FILE" ]]; then
  restart_sha_after="$(sha256sum "initial_$(basename "$RESTART_FILE")" 2>/dev/null | awk '{print $1}')"
else
  restart_sha_after=NA
fi
restart_path_staged=0
[[ -n "$RESTART_FILE" && -s "initial_$(basename "$RESTART_FILE")" ]] && restart_path_staged=1
grep -Eq 'restart_mode[[:space:]]*=[[:space:]]*'"'"'restart'"'"'' neb.in && restart_input=1 || restart_input=0
# QE does not emit a uniform restart banner in every neb.x build.  A staged
# path plus a successful 10-image initialization is the auditable acceptance
# criterion; record whether neb.x rewrote the path as additional evidence.
restart_accepted=0
if [[ "$RUN_MODE" == restart && "$restart_path_staged" == 1 && "$restart_input" == 1 && "$image_dirs" -ge 10 ]]; then restart_accepted=1; fi
grep -E 'JOB DONE|neb:|image|Error in routine|MPI|FIO|errno|nan|inf|convergence NOT achieved|restart' neb.out neb.err > gate_markers.txt || true
find "$QE_OUTDIR" -maxdepth 3 -type f -printf '%P\t%s\n' 2>/dev/null | sort > qe_output_manifest.tsv || true
du -sh "$TMP_ROOT" > scratch_usage_before_cleanup.txt 2>/dev/null || true
du -sb "$TMP_ROOT" > scratch_usage_bytes_before_cleanup.txt 2>/dev/null || true

gate=0
[[ "$rc" == 0 ]] || gate=1
[[ "$job_done" == 1 ]] || gate=1
[[ "$image_dirs" -ge 10 ]] || gate=1
[[ "$finite_image_rows" == 1 ]] || gate=1
[[ "$wfc_bytes" == 0 ]] || gate=1
if [[ "$RUN_MODE" == restart ]]; then
  [[ "$restart_accepted" == 1 ]] || gate=1
else
  grep -Eq 'restart_mode[[:space:]]*=[[:space:]]*'"'"'from_scratch'"'"'' neb.in || gate=1
fi
! grep -qiE 'Error in routine|stopping|MPI_Init|FIO|errno|nan|inf|CANNOT CREATE FIFO' neb.out neb.err || gate=1
{
  echo "rc=$rc"
  echo "job_done=$job_done"
  echo "image_dirs=$image_dirs"
  echo "image_numeric_rows=$image_numeric_rows"
  echo "finite_image_rows=$finite_image_rows"
  echo "wfc_bytes=$wfc_bytes"
  echo "restart_input=$restart_input"
  echo "restart_path_staged=$restart_path_staged"
  echo "restart_sha_before=$RESTART_SHA_BEFORE"
  echo "restart_sha_after=$restart_sha_after"
  echo "restart_accepted=$restart_accepted"
  echo "gate=$gate"
  if (( gate == 0 )); then
    echo 'scratch_cleanup=removed_after_clean_smoke'
    rm -rf "$TMP_ROOT"
  else
    echo 'scratch_cleanup=preserved_for_diagnosis'
  fi
} >> runtime_provenance.txt
echo "$gate" > exit_code.txt
exit "$gate"
