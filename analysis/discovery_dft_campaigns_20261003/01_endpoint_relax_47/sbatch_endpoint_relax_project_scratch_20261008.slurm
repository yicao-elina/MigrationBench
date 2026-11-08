#!/bin/bash
# MigrationBench Discovery endpoint-relax smoke/production variant.
# QE scratch is bound to a job-specific directory on the persistent project
# filesystem, but QE wavefunction writes are disabled.  The directory is
# retained on failure and removed only after a clean run.
# This file is intentionally separate from the existing node-local wrapper.

#SBATCH --job-name=mb_ep_relax_ps
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
SCRATCH_ROOT="${QE_SHARED_SCRATCH_ROOT:-$ROOT/qe_scratch}"
JOB_SCRATCH="$SCRATCH_ROOT/${ENDPOINT_ID}_${SLURM_JOB_ID}"
QE_OUTDIR="$JOB_SCRATCH/qe_out"
TMP_ROOT="$JOB_SCRATCH/tmp"
SIF=/projects/pclancy3_lab/qe75-h200-v2.sif

test -s "$INPUT"
test -r "$SIF"
if grep -qiE 'nspin|starting_magnetization|tot_magnetization|constrained_magnetization|angle1|angle2' "$INPUT"; then
  echo "magnetic QE keyword found; refusing to run: $INPUT" >&2
  exit 41
fi
grep -q "calculation[[:space:]]*=[[:space:]]*'relax'" "$INPUT"

mkdir -p "$RUN" "$JOB_SCRATCH" "$QE_OUTDIR" "$TMP_ROOT"
cp "$INPUT" "$RUN/input.relax.in"
cd "$RUN"

# Record capacity before QE starts.  This is project-filesystem capacity, not RAM.
df -h "$JOB_SCRATCH" > scratch_df_before.txt
df -ih "$JOB_SCRATCH" > scratch_df_inode_before.txt
df -Pk "$JOB_SCRATCH" | awk 'NR==2 {print $4}' > scratch_available_kb_before.txt

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
export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}$JOB_SCRATCH:$JOB_SCRATCH"
export APPTAINER_BIND="$APPTAINER_BIND,$ROOT:$ROOT"

{
  date
  hostname
  echo "candidate_id=$ENDPOINT_ID"
  echo "run_subdir=$RUN_SUBDIR"
  echo "job_id=$SLURM_JOB_ID"
  echo "campaign_root=$ROOT"
  echo "container=$SIF"
  echo "qe_version=7.5"
  echo "omp_num_threads=$OMP_NUM_THREADS"
  echo "scratch_mode=persistent_project_filesystem"
  echo "scratch_root=$JOB_SCRATCH"
  echo "qe_outdir=$QE_OUTDIR"
  echo "runtime_overrides=disk_io=none,nstep=$RELAX_NSTEP,outdir=$QE_OUTDIR"
  echo "cleanup_policy=remove_job_scratch_only_after_rc0_and_JOB_DONE"
  echo "relax_policy=relax_to_force_threshold_or_max_${RELAX_NSTEP}_steps"
  cat scratch_df_before.txt
  cat scratch_df_inode_before.txt
  sha256sum input.relax.in input.relax.runtime.in
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

job_done=0
grep -q 'JOB DONE' scf.out && job_done=1 || true
du -sh "$JOB_SCRATCH" > scratch_usage_before_cleanup.txt 2>/dev/null || true
du -sb "$JOB_SCRATCH" > scratch_usage_bytes_before_cleanup.txt 2>/dev/null || true
{
  echo "QE_OUTDIR=$QE_OUTDIR"
  find "$QE_OUTDIR" -maxdepth 1 -type f -printf '%f\t%s\n' 2>/dev/null || true
} > qe_output_manifest.txt

# Extract the last complete 81-atom block for audit, even if convergence fails.
if awk '
  /ATOMIC_POSITIONS/ { if (n == 81) { last_n=n; for (i=1;i<=n;i++) last[i]=atom[i] }; inpos=1; n=0; next }
  inpos && NF >= 4 && $1 ~ /^[A-Za-z][A-Za-z0-9]*$/ { atom[++n]=$0; next }
  inpos && n > 0 { if (n == 81) { last_n=n; for (i=1;i<=n;i++) last[i]=atom[i] }; inpos=0 }
  END { if (n == 81) { last_n=n; for (i=1;i<=n;i++) last[i]=atom[i] }; if (last_n != 81) exit 1; print 81; print "last printed QE ATOMIC_POSITIONS"; for (i=1;i<=81;i++) print last[i] }
' scf.out > relaxed_last_step.xyz.tmp; then
  mv relaxed_last_step.xyz.tmp relaxed_last_step.xyz
  {
    echo "candidate=$ENDPOINT_ID"
    echo "job_id=$SLURM_JOB_ID"
    echo "source=scf.out:last_complete_ATOMIC_POSITIONS"
    echo "nat=81"
    echo "scratch_mode=persistent_project_filesystem"
    grep '!    total energy' scf.out | tail -1 || true
    grep 'Total force' scf.out | tail -1 || true
    sha256sum input.relax.in input.relax.runtime.in scf.out relaxed_last_step.xyz
  } > relaxed_last_step_provenance.txt
else
  rm -f relaxed_last_step.xyz.tmp
  echo "no complete 81-atom ATOMIC_POSITIONS block found" > relaxed_last_step_provenance.txt
fi

echo "$rc" > exit_code.txt
grep -E '!    total energy|Forces acting on atoms|Total force|convergence has been achieved|JOB DONE|Error in routine|stopping|convergence NOT achieved' scf.out > gate_markers.txt || true
{
  echo "rc=$rc"
  echo "job_done=$job_done"
  if (( rc == 0 && job_done == 1 )); then
    echo "scratch_cleanup=removed_after_success"
    rm -rf "$JOB_SCRATCH"
  else
    echo "scratch_cleanup=preserved_for_diagnosis"
  fi
} >> runtime_provenance.txt

set -e
exit "$rc"
