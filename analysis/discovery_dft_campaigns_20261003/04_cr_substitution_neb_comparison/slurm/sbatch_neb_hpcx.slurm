#!/bin/bash
# Unified Discovery QE-NEB runner for the isolated 18-path comparison.
# Submit with:
#   sbatch --dependency=afterok:<main-job-ids> \
#     --export=ALL,NEB_KIND=cr,NEB_ID=2L-Cu__neb-QLs \
#     sbatch_neb_hpcx.slurm
# NEB_KIND=original expects original_clean_neb_9/inputs/<NEB_ID>.

#SBATCH --job-name=mb-neb
# Keep a full 72 h Slurm allocation.  QE inputs use max_seconds=250000 s
# (about 69.4 h), leaving a shutdown/restart margin before Slurm kills the job.
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --partition=azgpuh200
#SBATCH -A pclancy3_lab
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003/logs/neb-%x-%j.out
#SBATCH --error=/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003/logs/neb-%x-%j.err

set -euo pipefail
: "${NEB_KIND:?submit with NEB_KIND=original or NEB_KIND=cr}"
: "${NEB_ID:?submit with NEB_ID=<path-id>}"

CAMPAIGN=/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003
case "$NEB_KIND" in
  original)
    ROOT="$CAMPAIGN/original_clean_neb_9"
    ;;
  cr)
    ROOT="$CAMPAIGN/cr_substitution_neb_comparison"
    ;;
  *)
    echo "invalid NEB_KIND=$NEB_KIND" >&2
    exit 2
    ;;
esac

INPUT_ROOT="${INPUT_ROOT:-$ROOT/inputs}"
INPUT_DIR="$INPUT_ROOT/$NEB_ID"
RUN_ROOT="${RUN_ROOT:-$ROOT/results}"
RUN="$RUN_ROOT/$NEB_ID"
NEB_RESTART_MODE="${NEB_RESTART_MODE:-from_scratch}"
RESTART_SOURCE_DIR="${RESTART_SOURCE_DIR:-}"
SIF=/projects/pclancy3_lab/qe75-h200-v2.sif
PSEUDO=/home/ycao73/ONCVPseudoPack/SG15_v1.2/UPF/PBE

for f in neb.in pw_1.in pw_2.in; do
  test -s "$INPUT_DIR/$f"
done
test -r "$SIF"
grep -q 'string_method[[:space:]]*=[[:space:]]*'"'"'neb'"'"'' "$INPUT_DIR/neb.in"
grep -q 'num_of_images[[:space:]]*=[[:space:]]*10' "$INPUT_DIR/neb.in"
grep -q 'nat[[:space:]]*=[[:space:]]*81' "$INPUT_DIR/neb.in"
if grep -qiE '^[[:space:]]*(nspin|starting_magnetization|tot_magnetization|constrained_magnetization|angle1|angle2)[[:space:]]*=' "$INPUT_DIR/neb.in"; then
  echo "magnetic QE keyword found; refusing to run: $INPUT_DIR/neb.in" >&2
  exit 41
fi

mkdir -p "$RUN/out" "$RUN/tmp"
for f in neb.in pw_1.in pw_2.in; do
  cp "$INPUT_DIR/$f" "$RUN/source_$f"
  # Keep source input immutable while making a runtime copy valid on Discovery.
  sed "s|pseudo_dir[[:space:]]*=.*|pseudo_dir = '$PSEUDO/'|" "$INPUT_DIR/$f" > "$RUN/$f"
done

case "$NEB_RESTART_MODE" in
  from_scratch)
    ;;
  restart)
    test -n "$RESTART_SOURCE_DIR"
    test -s "$RESTART_SOURCE_DIR/sb2te3.path"
    for f in sb2te3.path sb2te3.path0 sb2te3.path1 sb2te3.broyden; do
      [[ -s "$RESTART_SOURCE_DIR/$f" ]] && cp "$RESTART_SOURCE_DIR/$f" "$RUN/$f"
    done
    sed -i -E "s/(restart_mode[[:space:]]*=)[[:space:]]*'from_scratch'/\1 'restart'/" "$RUN/neb.in"
    ;;
  *)
    echo "invalid NEB_RESTART_MODE=$NEB_RESTART_MODE" >&2
    exit 3
    ;;
esac
cd "$RUN"
export OMP_NUM_THREADS=11
export OMP_STACKSIZE=2G
export TMPDIR="$RUN/tmp"
ml purge
export SINGULARITY_BIND="${SINGULARITY_BIND:-}"
source /azwestus3/pclancy3_lab/singularity.sh
if [[ -n "${SINGULARITY_BIND:-}" ]]; then export APPTAINER_BIND="$SINGULARITY_BIND"; fi
unset SINGULARITY_BIND
export APPTAINERENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"
export APPTAINERENV_OMP_STACKSIZE="$OMP_STACKSIZE"

{
  date
  hostname
  echo "neb_kind=$NEB_KIND"
  echo "neb_id=$NEB_ID"
  echo "job_id=$SLURM_JOB_ID"
  echo "campaign_root=$ROOT"
  echo "run_root=$RUN_ROOT"
  echo "neb_restart_mode=$NEB_RESTART_MODE"
  echo "restart_source_dir=${RESTART_SOURCE_DIR:-none}"
  echo "container=$SIF"
  echo "qe_version=7.5"
  echo "mpi_launcher=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi/bin/mpirun"
  echo "omp_num_threads=$OMP_NUM_THREADS"
  sha256sum source_neb.in source_pw_1.in source_pw_2.in neb.in pw_1.in pw_2.in
  cat "$CAMPAIGN/endpoint_relax_47/container_sha256.txt"
  for p in Cr_PBE_FR.SG15v1.2.UPF Cu_PBE_FR.SG15v1.2.UPF Li_PBE_FR.SG15v1.2.UPF Na_PBE_FR.SG15v1.2.UPF Sb_PBE_FR.SG15v1.2.UPF Te_PBE_FR.SG15v1.2.UPF; do
    [[ -r "$PSEUDO/$p" ]] && sha256sum "$PSEUDO/$p"
  done
  grep -iE 'string_method|num_of_images|degauss|ecutwfc|ecutrho|pseudo_dir|nat|ntyp|K_POINTS' neb.in pw_1.in pw_2.in
  echo provenance_complete
} > runtime_provenance.txt

set +e
srun --export=ALL singularity exec --nv --cleanenv "$SIF" \
  bash --noprofile --norc -c '
    set -euo pipefail
    export NVHPC=/opt/nvidia/hpc_sdk
    export NVARCH=Linux_x86_64
    export NVCOMPILERS=/opt/nvidia/hpc_sdk
    export OPAL_PREFIX=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi
    export PATH=/opt/qe/bin:$OPAL_PREFIX/bin:/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/compilers/bin:$PATH
    export LD_LIBRARY_PATH=$OPAL_PREFIX/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/compilers/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/math_libs/lib64:/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/lib64:${LD_LIBRARY_PATH:-}
    export OMP_NUM_THREADS=11
    export OMP_STACKSIZE=2G
    export OMPI_MCA_opal_cuda_support=false
    export UCX_MEMTYPE_CACHE=n
    export UCX_TLS=self,shm,tcp
    export CUDA_LAUNCH_BLOCKING=1
    export HWLOC_HIDE_ERRORS=1
    export OMPI_MCA_hwloc_base_binding_policy=none
    export OMPI_MCA_rmaps_base_mapping_policy=slot
    REAL_MPIRUN="$OPAL_PREFIX/bin/mpirun"
    echo "REAL_MPIRUN=$REAL_MPIRUN"
    "$REAL_MPIRUN" --bind-to none --map-by slot -np 1 neb.x -in neb.in
  ' > neb.out 2> neb.err
rc=$?
set -e
echo "$rc" > exit_code.txt

gate=0
{
  [[ "$rc" == 0 ]] && echo QE_EXIT=PASS || { echo QE_EXIT=FAIL; gate=1; }
  grep -qiE 'JOB DONE|neb:[[:space:]]*convergence|convergence achieved' neb.out && echo PATH_TERMINATION=PASS || { echo PATH_TERMINATION=FAIL; gate=1; }
  ! grep -qiE 'Error in routine|stopping|convergence NOT achieved|MPI|container launch error|nan|inf' neb.out neb.err && echo ERROR_MARKERS=PASS || { echo ERROR_MARKERS=FAIL; gate=1; }
  grep -q 'num_of_images[[:space:]]*=[[:space:]]*10' neb.in && grep -q 'nat[[:space:]]*=[[:space:]]*81' neb.in && echo INPUT_IDENTITY=PASS || { echo INPUT_IDENTITY=FAIL; gate=1; }
  image_records=$(grep -Eic 'image[[:space:]]+[0-9]+|image[[:space:]]*=' neb.out || true)
  [[ "$image_records" -ge 10 ]] && echo IMAGE_RECORDS=PASS || { echo IMAGE_RECORDS=FAIL count="$image_records"; gate=1; }
} > gate_markers.txt
exit "$gate"
