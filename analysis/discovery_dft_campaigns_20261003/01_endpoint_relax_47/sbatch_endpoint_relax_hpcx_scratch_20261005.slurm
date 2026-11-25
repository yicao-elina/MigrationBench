#!/bin/bash
# Discovery endpoint relax runner; QE scratch stays on Discovery node-local
# LOCAL_SCRATCH (with a /tmp fallback only for diagnostics on other systems).
# Campaign-isolated wrapper for MigrationBench discovery_dft_20261003.
#SBATCH --job-name=mb_ep_relax
#SBATCH --time=71:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --partition=azgpuh200
#SBATCH -A pclancy3_lab
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003/logs/%x-%j.out
#SBATCH --error=/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003/logs/%x-%j.err

set -euo pipefail
: "${ENDPOINT_ID:?submit with --export=ALL,ENDPOINT_ID=unique_XXX}"
RELAX_NSTEP="${RELAX_NSTEP:-200}"
RUN_SUBDIR="${RUN_SUBDIR:-$ENDPOINT_ID}"
[[ "$RELAX_NSTEP" =~ ^[1-9][0-9]*$ ]]
[[ "$RUN_SUBDIR" =~ ^[A-Za-z0-9_.-]+$ ]]
ROOT=/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003/endpoint_relax_47
INPUT="$ROOT/inputs/${ENDPOINT_ID}/relax_${ENDPOINT_ID}.in"
RUN="$ROOT/results/${RUN_SUBDIR}"
SIF=/projects/pclancy3_lab/qe75-h200-v2.sif

test -s "$INPUT"
test -r "$SIF"
if grep -qiE 'nspin|starting_magnetization|tot_magnetization|constrained_magnetization|angle1|angle2' "$INPUT"; then
  echo "magnetic QE keyword found; refusing to run: $INPUT" >&2
  exit 41
fi
grep -q "calculation[[:space:]]*=[[:space:]]*'relax'" "$INPUT"

mkdir -p "$RUN/out" "$RUN/tmp"
cp "$INPUT" "$RUN/input.relax.in"
cd "$RUN"

# QE writes large wavefunction files during relaxation.  Keep them on the
# compute node and retain only auditable text/provenance in the campaign root.
# Discovery exposes a multi-terabyte per-job filesystem as LOCAL_SCRATCH;
# /tmp is only a portability fallback and is not expected on Discovery.
LOCAL_BASE="${LOCAL_SCRATCH:-/tmp/${USER}}"
TMP_ROOT="${LOCAL_BASE%/}/mb_ep_${ENDPOINT_ID}_${SLURM_JOB_ID}"
QE_OUTDIR="$TMP_ROOT/out"
mkdir -p "$QE_OUTDIR"
cp input.relax.in input.relax.runtime.in
awk -v d="$QE_OUTDIR" -v n="$RELAX_NSTEP" '
  BEGIN { q=sprintf("%c",34); s=sprintf("%c",39); in_control=0; has_disk=0; has_nstep=0 }
  /^[[:space:]]*&CONTROL/ { in_control=1; print; next }
  in_control && $0 ~ /^[[:space:]]*disk_io[[:space:]]*=/ { print "  disk_io = " s "none" s; has_disk=1; next }
  in_control && $0 ~ /^[[:space:]]*nstep[[:space:]]*=/ { print "  nstep = " n; has_nstep=1; next }
  in_control && $0 ~ /^[[:space:]]*\/[[:space:]]*$/ {
    if (!has_disk) print "  disk_io = " s "none" s
    if (!has_nstep) print "  nstep = " n
    in_control=0
  }
  $0 ~ /^[[:space:]]*outdir[[:space:]]*=/ { print "  outdir = " q d q; next }
  { print }
' input.relax.runtime.in > input.relax.runtime.in.tmp
mv input.relax.runtime.in.tmp input.relax.runtime.in
export OMP_NUM_THREADS=11
export OMP_STACKSIZE=2G
export TMPDIR="$TMP_ROOT"
export APPTAINERENV_TMPDIR="$TMP_ROOT"

export SINGULARITY_BIND="${SINGULARITY_BIND:-}"
source /projects/pclancy3_lab/singularity.sh
if [ -n "${SINGULARITY_BIND:-}" ]; then
  export APPTAINER_BIND="$SINGULARITY_BIND"
  unset SINGULARITY_BIND
fi
export APPTAINERENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"
export APPTAINERENV_OMP_STACKSIZE="$OMP_STACKSIZE"
export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}$TMP_ROOT:$TMP_ROOT"

{
  date
  hostname
  echo "candidate_id=$ENDPOINT_ID"
  echo "run_subdir=$RUN_SUBDIR"
  echo "job_id=$SLURM_JOB_ID"
  echo "campaign_root=$ROOT"
  echo "container=$SIF"
  echo "mpi_launcher=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi/bin/mpirun"
  echo "qe_version=7.5"
  echo "omp_num_threads=$OMP_NUM_THREADS"
  echo "omp_stacksize=$OMP_STACKSIZE"
  echo "scratch_root=$TMP_ROOT"
  echo "runtime_overrides=disk_io=none,nstep=$RELAX_NSTEP,outdir=$QE_OUTDIR"
  echo "relax_policy=relax_to_force_threshold_or_max_${RELAX_NSTEP}_steps"
  df -h "$TMP_ROOT"
  sha256sum input.relax.in
  sha256sum input.relax.runtime.in
  cat "$ROOT/container_sha256.txt"
  for p in Cr_PBE_FR.SG15v1.2.UPF Sb_PBE_FR.SG15v1.2.UPF Te_PBE_FR.SG15v1.2.UPF; do
    sha256sum "/home/ycao73/ONCVPseudoPack/SG15_v1.2/UPF/PBE/$p"
  done
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
    export OMPI_MCA_opal_cuda_support=false
    export UCX_MEMTYPE_CACHE=n
    export UCX_TLS=self,shm,tcp
    export CUDA_LAUNCH_BLOCKING=1
    export HWLOC_HIDE_ERRORS=1
    export OMPI_MCA_hwloc_base_binding_policy=none
    export OMPI_MCA_rmaps_base_mapping_policy=slot
    REAL_MPIRUN="$OPAL_PREFIX/bin/mpirun"
    echo "REAL_MPIRUN=$REAL_MPIRUN"
    "$REAL_MPIRUN" --bind-to none --map-by slot -np 1 pw.x -inp input.relax.runtime.in
  ' > scf.out 2> scf.err
rc=$?
{
  echo "QE_OUTDIR=$QE_OUTDIR"
  find "$QE_OUTDIR" -maxdepth 1 -type f -printf '%f\t%s\n' 2>/dev/null || true
} > qe_output_manifest.txt

# Preserve the last complete QE geometry and its final printed energy/force
# before removing node-local scratch.  These are diagnostic on a failed job;
# the scientific acceptance gate still requires rc=0 and JOB DONE.
if awk '
  /^ATOMIC_POSITIONS/ { seen=1; n=0; next }
  seen && n < 81 && NF >= 4 && $1 ~ /^[A-Za-z][A-Za-z0-9]*$/ { atom[++n]=$0; next }
  END {
    if (seen && n == 81) {
      print 81
      print "last printed QE ATOMIC_POSITIONS; acceptance requires QE rc=0 and JOB DONE"
      for (i=1; i<=n; i++) print atom[i]
      exit 0
    }
    exit 1
  }
' scf.out > relaxed_last_step.xyz.tmp; then
  mv relaxed_last_step.xyz.tmp relaxed_last_step.xyz
  {
    echo "candidate=$ENDPOINT_ID"
    echo "job_id=$SLURM_JOB_ID"
    echo "source=scf.out:last_complete_ATOMIC_POSITIONS"
    echo "nat=81"
    echo "acceptance=diagnostic_only_until_QE_rc_0_JOB_DONE_finite_force_and_identity_gates"
    grep '!    total energy' scf.out | tail -1 || true
    grep 'Total force' scf.out | tail -1 || true
    sha256sum input.relax.in input.relax.runtime.in scf.out relaxed_last_step.xyz
  } > relaxed_last_step_provenance.txt
else
  rm -f relaxed_last_step.xyz.tmp
  echo "no complete 81-atom ATOMIC_POSITIONS block found" > relaxed_last_step_provenance.txt
fi
rm -rf "$TMP_ROOT"
set -e
echo "$rc" > exit_code.txt
grep -E '!    total energy|Forces acting on atoms|Total force|convergence has been achieved|JOB DONE|Error in routine|stopping' scf.out > gate_markers.txt || true
exit "$rc"
