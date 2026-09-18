#!/bin/bash
#SBATCH --job-name=mb_qe_neb_warm_s42
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

: "${MIGRATIONBENCH_NEB_INPUT:?set MIGRATIONBENCH_NEB_INPUT}"
: "${MIGRATIONBENCH_NEB_MANIFEST:?set MIGRATIONBENCH_NEB_MANIFEST}"
: "${MIGRATIONBENCH_NEB_LAUNCH_MANIFEST:?set MIGRATIONBENCH_NEB_LAUNCH_MANIFEST}"
RUN_ROOT="${MIGRATIONBENCH_RUN_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_runs}"
CODE_ROOT="${MIGRATIONBENCH_CODE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
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
cp "${MIGRATIONBENCH_NEB_INPUT}" "${JOB_RUN_DIR}/neb.in"
cp "${MIGRATIONBENCH_NEB_MANIFEST}" "${JOB_RUN_DIR}/warmstarted_neb_manifest.json"
cp "${MIGRATIONBENCH_NEB_LAUNCH_MANIFEST}" "${JOB_RUN_DIR}/production_launch_manifest.json"
cd "${JOB_RUN_DIR}"
MANIFEST="${JOB_RUN_DIR}/warmstarted_neb_manifest.json"

read -r EXPECTED_NTASKS EXPECTED_KPOINT_POOLS EXPECTED_MEMORY <<EOF
$(python - "${MANIFEST}" <<'PY'
import json, sys
resources = json.load(open(sys.argv[1]))["resources"]
print(resources["ntasks"], resources["kpoint_pools"], resources["memory"])
PY
)
EOF
KPOINT_POOLS="${MIGRATIONBENCH_KPOINT_POOLS:-${EXPECTED_KPOINT_POOLS}}"
test "${SLURM_NTASKS}" -eq "${EXPECTED_NTASKS}"
test "${KPOINT_POOLS}" -eq "${EXPECTED_KPOINT_POOLS}"
test "${SLURM_MEM_PER_NODE:-}" = "${EXPECTED_MEMORY%G}000" || \
  echo "warning: Slurm memory ${SLURM_MEM_PER_NODE:-unknown} MB differs from manifest ${EXPECTED_MEMORY}" >&2

python - "${MANIFEST}" <<'PY' > warmup_sources.tsv
import json, sys
for row in json.load(open(sys.argv[1]))["warmup_sources"]:
    print(row["image_index_qe"], row["source_save_dir"], sep="\t")
PY

while IFS=$'\t' read -r image source; do
  destination="${JOB_RUN_DIR}/out/sb2te3_${image}/sb2te3.save"
  mkdir -p "${destination}"
  rsync -a --exclude='wfc*.dat' "${source}/" "${destination}/"
  test -s "${destination}/charge-density.dat"
  test -s "${destination}/data-file-schema.xml"
done < warmup_sources.tsv

ulimit -s unlimited || true
export OMP_NUM_THREADS=1
echo "host=$(hostname)"
echo "job_run_dir=${JOB_RUN_DIR}"
echo "slurm_ntasks=${SLURM_NTASKS}"
echo "kpoint_pools=${KPOINT_POOLS}"
which neb.x
python "${CODE_ROOT}/scripts/migrationbench/capture_runtime_provenance.py" \
  --output runtime_provenance.json --binary neb.x --binary pw.x \
  --input "neb_input=${JOB_RUN_DIR}/neb.in" \
  --input "warmstart_manifest=${JOB_RUN_DIR}/warmstarted_neb_manifest.json" \
  --input "launch_manifest=${JOB_RUN_DIR}/production_launch_manifest.json"

mpirun -np "${SLURM_NTASKS}" neb.x -nk "${KPOINT_POOLS}" -inp neb.in > neb.out
