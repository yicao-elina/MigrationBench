#!/bin/bash
#SBATCH --job-name=mb_qe_scfwarm_restart_s42
#SBATCH --time=24:00:00
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

: "${MIGRATIONBENCH_SCF_INPUT:?set MIGRATIONBENCH_SCF_INPUT}"
: "${MIGRATIONBENCH_SCF_MANIFEST:?set MIGRATIONBENCH_SCF_MANIFEST}"
CODE_ROOT="${MIGRATIONBENCH_CODE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN_ROOT="${MIGRATIONBENCH_RUN_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_runs}"
JOB_RUN_DIR="${RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${JOB_RUN_DIR}"
archive_slurm_logs() {
  local submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
  for suffix in out err; do
    local source="${submit_dir}/slurm-${SLURM_JOB_ID}.${suffix}"
    [ -e "${source}" ] && mv -f "${source}" "${JOB_RUN_DIR}/" || true
  done
}
trap archive_slurm_logs EXIT
cp "${MIGRATIONBENCH_SCF_INPUT}" "${JOB_RUN_DIR}/scf.in"
cp "${MIGRATIONBENCH_SCF_MANIFEST}" "${JOB_RUN_DIR}/scf_continuation_manifest.json"
cd "${JOB_RUN_DIR}"
MANIFEST="${JOB_RUN_DIR}/scf_continuation_manifest.json"

read -r EXPECTED_NTASKS EXPECTED_KPOINT_POOLS SOURCE_SAVE <<EOF
$(python - "${MANIFEST}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
resources = manifest["resources"]
print(resources["ntasks"], resources["kpoint_pools"], manifest["parent"]["source_save_dir"])
PY
)
EOF
KPOINT_POOLS="${MIGRATIONBENCH_KPOINT_POOLS:-${EXPECTED_KPOINT_POOLS}}"
test "${SLURM_NTASKS}" -eq "${EXPECTED_NTASKS}"
test "${KPOINT_POOLS}" -eq "${EXPECTED_KPOINT_POOLS}"

DESTINATION="${JOB_RUN_DIR}/out/sb2te3.save"
mkdir -p "${DESTINATION}"
rsync -a --exclude='wfc*.dat' "${SOURCE_SAVE}/" "${DESTINATION}/"
test -s "${DESTINATION}/charge-density.dat"
test -s "${DESTINATION}/data-file-schema.xml"

python "${CODE_ROOT}/scripts/migrationbench/capture_runtime_provenance.py" \
  --output "${JOB_RUN_DIR}/runtime_provenance.json" \
  --input "scf_input=${JOB_RUN_DIR}/scf.in" \
  --input "scf_manifest=${MANIFEST}" \
  --input "parent_charge_density=${DESTINATION}/charge-density.dat" \
  --input "parent_xml=${DESTINATION}/data-file-schema.xml" \
  --binary pw.x

ulimit -s unlimited || true
export OMP_NUM_THREADS=1
echo "host=$(hostname)"
echo "job_run_dir=${JOB_RUN_DIR}"
echo "source_save=${SOURCE_SAVE}"
echo "slurm_ntasks=${SLURM_NTASKS}"
echo "kpoint_pools=${KPOINT_POOLS}"
which pw.x

mpirun -np "${SLURM_NTASKS}" pw.x -nk "${KPOINT_POOLS}" -inp scf.in > scf.out
grep -q "convergence has been achieved" scf.out
grep -q "JOB DONE" scf.out
