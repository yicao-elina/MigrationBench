#!/bin/bash
# Generic Discovery QE relax runner for the isolated 47-endpoint campaign.
# Submit with: sbatch --export=ALL,ENDPOINT_ID=unique_041 sbatch_endpoint_relax_hpcx.slurm
# This wrapper is for QE relax, not single-point SCF or NEB.
#SBATCH --job-name=mb-ep-relax
#SBATCH --time=71:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --partition=azgpuh200
#SBATCH -A pclancy3_lab
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003/logs/endpoint-relax-%j.out
#SBATCH --error=/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003/logs/endpoint-relax-%j.err

set -euo pipefail
: "${ENDPOINT_ID:?submit with --export=ALL,ENDPOINT_ID=unique_XXX}"

ROOT=/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003/endpoint_relax_47
INPUT="$ROOT/inputs/$ENDPOINT_ID/relax_${ENDPOINT_ID}.in"
RUN="$ROOT/results/$ENDPOINT_ID"
SIF=/projects/pclancy3_lab/qe75-h200-v2.sif
PSEUDO=/home/ycao73/ONCVPseudoPack/SG15_v1.2/UPF/PBE

test -s "$INPUT"
test -r "$SIF"
grep -q "calculation[[:space:]]*=[[:space:]]*'relax'" "$INPUT"
grep -q "nat[[:space:]]*=[[:space:]]*81" "$INPUT"
if grep -qiE '^[[:space:]]*(nspin|starting_magnetization|tot_magnetization|constrained_magnetization|angle1|angle2)[[:space:]]*=' "$INPUT"; then
  echo "magnetic QE keyword found; refusing to run: $INPUT" >&2
  exit 41
fi

mkdir -p "$RUN/out" "$RUN/tmp"
cp "$INPUT" "$RUN/input.relax.in"
cd "$RUN"
# Prefer node-local SLURM_TMPDIR when Discovery provides it.  The fallback
# stays inside this isolated campaign root; /tmp can fill during long QE
# relaxations and turn an otherwise complete QE run into a false failure.
QE_OUTDIR="${SLURM_TMPDIR:-$RUN/tmp}/mb_ep_${ENDPOINT_ID}_${SLURM_JOB_ID}/out"
mkdir -p "$QE_OUTDIR"
sed -E "s|^([[:space:]]*outdir[[:space:]]*=).*|\1 '$QE_OUTDIR/'|" \
  input.relax.in > input.relax.runtime.in
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
  echo "endpoint_id=$ENDPOINT_ID"
  echo "job_id=$SLURM_JOB_ID"
  echo "campaign_root=$ROOT"
  echo "container=$SIF"
  echo "qe_version=7.5"
  echo "mpi_launcher=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi/bin/mpirun"
  echo "omp_num_threads=$OMP_NUM_THREADS"
  sha256sum input.relax.in input.relax.runtime.in
  cat "$ROOT/container_sha256.txt"
  for p in Cr_PBE_FR.SG15v1.2.UPF Sb_PBE_FR.SG15v1.2.UPF Te_PBE_FR.SG15v1.2.UPF; do sha256sum "$PSEUDO/$p"; done
  grep -iE 'calculation|degauss|ecutwfc|ecutrho|pseudo_dir|nat|ntyp|K_POINTS' input.relax.in
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
    "$OPAL_PREFIX/bin/mpirun" --bind-to none --map-by slot -np 1 pw.x -inp input.relax.runtime.in
  ' > relax.out 2> relax.err
rc=$?
if ! cp -a "$QE_OUTDIR/." "$RUN/out/"; then
  echo "failed to copy node-local QE outdir back to $RUN/out" >&2
  rc=74
fi
set -e
cp relax.out scf.out
cp relax.err scf.err
echo "$rc" > exit_code.txt

gate=0
{
  grep -q 'JOB DONE' relax.out && echo JOB_DONE=PASS || { echo JOB_DONE=FAIL; gate=1; }
  grep -qE '!    total energy' relax.out && ! grep -qiE 'nan|inf' relax.out && echo ENERGY_FINITE=PASS || { echo ENERGY_FINITE=FAIL; gate=1; }
  grep -q 'Forces acting on atoms' relax.out && ! grep -qiE 'nan|inf' relax.out && echo FORCES_FINITE=PASS || { echo FORCES_FINITE=FAIL; gate=1; }
  ! grep -qiE 'Error in routine|stopping|convergence NOT achieved|MPI|container launch error' relax.out relax.err && echo ERROR_MARKERS=PASS || { echo ERROR_MARKERS=FAIL; gate=1; }
  grep -q "calculation[[:space:]]*=[[:space:]]*'relax'" input.relax.in && grep -q 'nat[[:space:]]*=[[:space:]]*81' input.relax.in && echo INPUT_IDENTITY=PASS || { echo INPUT_IDENTITY=FAIL; gate=1; }
  [[ "$rc" == 0 ]] && echo QE_EXIT=PASS || { echo QE_EXIT=FAIL; gate=1; }
} > gate_markers.txt
exit "$gate"
