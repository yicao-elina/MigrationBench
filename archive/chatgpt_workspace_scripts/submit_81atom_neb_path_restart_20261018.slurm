#!/bin/bash
# QE NEB path-level restart from the historical final saved path.
# The run directory must contain the reconstructed sb2te3.path beside neb.in.
#SBATCH --job-name=cr81pathrst
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --partition=parallel
#SBATCH -A pclancy3
#SBATCH --mem=180G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail

source /data/apps/go.sh
module load qe/7.3.1-cpu
set -u

: "${MIGRATIONBENCH_PATH_NUM:?set MIGRATIONBENCH_PATH_NUM to 1..5}"
: "${MIGRATIONBENCH_CAMPAIGN_ROOT:?set MIGRATIONBENCH_CAMPAIGN_ROOT}"

case "${MIGRATIONBENCH_PATH_NUM}" in
  1|2|3|4|5) ;;
  *) echo "invalid path number: ${MIGRATIONBENCH_PATH_NUM}" >&2; exit 2 ;;
esac

CODE_ROOT="${MIGRATIONBENCH_CODE_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_pipeline}"
INPUT_DIR="${MIGRATIONBENCH_CAMPAIGN_ROOT}/inputs/neb_${MIGRATIONBENCH_PATH_NUM}"
RUN_DIR="${MIGRATIONBENCH_CAMPAIGN_ROOT}/runs/neb_${MIGRATIONBENCH_PATH_NUM}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}/out"

archive_slurm_logs() {
  local submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
  for suffix in out err; do
    local source="${submit_dir}/slurm-${SLURM_JOB_ID}.${suffix}"
    [ -e "${source}" ] && mv -f "${source}" "${RUN_DIR}/" || true
  done
}
trap archive_slurm_logs EXIT

for required in neb.in sb2te3.path launch_manifest.json path_reconstruction.json historical_final_path.xyz historical_final_profile.dat; do
  test -s "${INPUT_DIR}/${required}"
  cp "${INPUT_DIR}/${required}" "${RUN_DIR}/${required}"
done
cd "${RUN_DIR}"

grep -Eq "restart_mode[[:space:]]*=+[[:space:]]*'restart'" neb.in
grep -q '^RESTART INFORMATION$' sb2te3.path
test "$(grep -c '^Image:' sb2te3.path)" -eq 10
test "$(awk 'NF == 4 {n++} END {print n + 0}' historical_final_path.xyz)" -eq 810

ulimit -s unlimited || true
export OMP_NUM_THREADS=1

{
  echo "host=$(hostname)"
  echo "user=$(whoami)"
  echo "job_run_dir=${RUN_DIR}"
  echo "path_id=81_neb_${MIGRATIONBENCH_PATH_NUM}"
  echo "source_input_dir=${INPUT_DIR}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "slurm_ntasks=${SLURM_NTASKS}"
  echo "slurm_mem_per_node=${SLURM_MEM_PER_NODE:-unset}"
  echo "omp_num_threads=${OMP_NUM_THREADS}"
  echo "qe_module=qe/7.3.1-cpu"
  echo "restart_mode=$(grep -m1 "restart_mode" neb.in)"
  echo "path_header=$(sed -n '1,6p' sb2te3.path | tr '\n' ' ')"
  which neb.x
  which pw.x
} | tee launch_runtime.txt

if [ -f "${CODE_ROOT}/scripts/migrationbench/capture_runtime_provenance.py" ]; then
  python "${CODE_ROOT}/scripts/migrationbench/capture_runtime_provenance.py" \
    --output runtime_provenance.json --binary neb.x --binary pw.x \
    --input "neb_input=${RUN_DIR}/neb.in" \
    --input "restart_path=${RUN_DIR}/sb2te3.path" \
    --input "launch_manifest=${RUN_DIR}/launch_manifest.json" \
    --input "path_reconstruction=${RUN_DIR}/path_reconstruction.json"
fi

export I_MPI_PMI_LIBRARY=/cm/shared/apps/slurm/current/lib64/libpmi.so
export I_MPI_SHM_CELL_BWD_SIZE=2048000
set +e
/cm/shared/apps/slurm/current/bin/srun -n "${SLURM_NTASKS}" neb.x -nk 4 -inp neb.in > neb.out 2> neb.err
rc=$?
set -e
printf '%s\n' "${rc}" > exit_code.txt
exit "${rc}"
