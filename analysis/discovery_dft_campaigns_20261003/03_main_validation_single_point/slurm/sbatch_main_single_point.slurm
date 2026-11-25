#!/bin/bash
# Discovery isolated main-campaign QE single-point runner.
# Do not use this script for relaxations or NEB.
#SBATCH --job-name=mb_main_sp
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
: "${CANDIDATE_ID:?submit with --export=ALL,CANDIDATE_ID=cand_XXXX}"

ROOT=/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003/main_single_point
INPUT_SUBDIR="${INPUT_SUBDIR:-qe_inputs_wave1}"
INPUT="$ROOT/$INPUT_SUBDIR/${CANDIDATE_ID}.scf.in"
RUN="$ROOT/results/${CANDIDATE_ID}"
SIF=/projects/pclancy3_lab/qe75-h200-v2.sif

test -s "$INPUT"
test -r "$SIF"
if grep -qiE 'nspin|starting_magnetization|tot_magnetization|constrained_magnetization|angle1|angle2' "$INPUT"; then
  echo "magnetic QE keyword found; refusing to run: $INPUT" >&2
  exit 41
fi
grep -q "calculation[[:space:]]*=[[:space:]]*'scf'" "$INPUT"

mkdir -p "$RUN/out" "$RUN/tmp"
cp "$INPUT" "$RUN/input.scf.in"
cd "$RUN"
export OMP_NUM_THREADS=11
export OMP_STACKSIZE=2G
export TMPDIR="$RUN/tmp"
ml purge
# The shared Discovery helper appends to SINGULARITY_BIND without a default.
# Initialize it before sourcing so this wrapper remains safe under set -u.
export SINGULARITY_BIND="${SINGULARITY_BIND:-}"
source /azwestus3/pclancy3_lab/singularity.sh
if [[ -n "${SINGULARITY_BIND:-}" ]]; then
  export APPTAINER_BIND="$SINGULARITY_BIND"
fi
unset SINGULARITY_BIND
export APPTAINERENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"
export APPTAINERENV_OMP_STACKSIZE="$OMP_STACKSIZE"

{
  date
  hostname
  echo "candidate_id=$CANDIDATE_ID"
  echo "job_id=$SLURM_JOB_ID"
  echo "campaign_root=$ROOT"
  echo "container=$SIF"
  echo "mpi_launcher=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi/bin/mpirun"
  echo "qe_version=7.5"
  echo "omp_num_threads=$OMP_NUM_THREADS"
  echo "omp_stacksize=$OMP_STACKSIZE"
  sha256sum input.scf.in
  cat "$ROOT/container_sha256.txt"
  grep -iE 'calculation|degauss|ecutwfc|ecutrho|pseudo_dir|nat|ntyp|K_POINTS' input.scf.in
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
    "$REAL_MPIRUN" --bind-to none --map-by slot -np 1 pw.x -inp input.scf.in
  ' > scf.out 2> scf.err
rc=$?
set -e
echo "$rc" > exit_code.txt

gate=0
{
  grep -q 'JOB DONE' scf.out && echo JOB_DONE=PASS || { echo JOB_DONE=FAIL; gate=1; }
  grep -qE '!    total energy' scf.out && ! grep -qiE 'nan|inf' scf.out && echo ENERGY_FINITE=PASS || { echo ENERGY_FINITE=FAIL; gate=1; }
  grep -q 'Forces acting on atoms' scf.out && ! grep -qiE 'nan|inf' scf.out && echo FORCES_FINITE=PASS || { echo FORCES_FINITE=FAIL; gate=1; }
  ! grep -qiE 'Error in routine|stopping|convergence NOT achieved|MPI|container launch error' scf.out scf.err && echo ERROR_MARKERS=PASS || { echo ERROR_MARKERS=FAIL; gate=1; }
  grep -q "calculation[[:space:]]*=[[:space:]]*'scf'" input.scf.in && grep -q 'nat[[:space:]]*=[[:space:]]*81' input.scf.in && echo INPUT_IDENTITY=PASS || { echo INPUT_IDENTITY=FAIL; gate=1; }
  [[ "$rc" == 0 ]] && echo QE_EXIT=PASS || { echo QE_EXIT=FAIL; gate=1; }
} > gate_markers.txt
exit "$gate"
