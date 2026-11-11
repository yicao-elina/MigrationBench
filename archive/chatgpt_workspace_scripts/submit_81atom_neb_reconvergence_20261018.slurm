#!/bin/bash
#SBATCH --job-name=cr81rconv
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
mkdir -p "${RUN_DIR}"

archive_slurm_logs() {
  local submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
  for suffix in out err; do
    local source="${submit_dir}/slurm-${SLURM_JOB_ID}.${suffix}"
    [ -e "${source}" ] && mv -f "${source}" "${RUN_DIR}/" || true
  done
}
trap archive_slurm_logs EXIT

test -s "${INPUT_DIR}/neb.in"
test -s "${INPUT_DIR}/neb.manifest.json"
test -s "${INPUT_DIR}/launch_manifest.json"
cp "${INPUT_DIR}/neb.in" "${RUN_DIR}/neb.in"
cp "${INPUT_DIR}/neb.manifest.json" "${RUN_DIR}/neb.manifest.json"
cp "${INPUT_DIR}/launch_manifest.json" "${RUN_DIR}/launch_manifest.json"
cd "${RUN_DIR}"

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
  echo "kpoint_pools=4"
  which neb.x
  which pw.x
} | tee launch_runtime.txt

if [ -f "${CODE_ROOT}/scripts/migrationbench/capture_runtime_provenance.py" ]; then
  python "${CODE_ROOT}/scripts/migrationbench/capture_runtime_provenance.py" \
    --output runtime_provenance.json --binary neb.x --binary pw.x \
    --input "neb_input=${RUN_DIR}/neb.in" \
    --input "neb_manifest=${RUN_DIR}/neb.manifest.json" \
    --input "launch_manifest=${RUN_DIR}/launch_manifest.json"
fi

set +e
mpirun -np "${SLURM_NTASKS}" neb.x -nk 4 -inp neb.in > neb.out 2> neb.err
rc=$?
set -e
printf '%s\n' "${rc}" > exit_code.txt
exit "${rc}"
