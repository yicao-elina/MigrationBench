#!/bin/bash
#SBATCH --job-name=mb_qe_neb_restart_s42
#SBATCH --time=48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --partition=parallel
#SBATCH -A pclancy3
#SBATCH --mem=160G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail
source /data/apps/go.sh
module load qe/7.3.1-cpu
set -u

: "${MIGRATIONBENCH_NEB_INPUT:?set MIGRATIONBENCH_NEB_INPUT}"
: "${MIGRATIONBENCH_NEB_RESTART_MANIFEST:?set MIGRATIONBENCH_NEB_RESTART_MANIFEST}"
: "${MIGRATIONBENCH_PARENT_RUN_DIR:?set MIGRATIONBENCH_PARENT_RUN_DIR}"
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
cp "${MIGRATIONBENCH_NEB_RESTART_MANIFEST}" "${JOB_RUN_DIR}/qe_restart_manifest.json"
cd "${JOB_RUN_DIR}"
MANIFEST="${JOB_RUN_DIR}/qe_restart_manifest.json"

read -r EXPECTED_NTASKS EXPECTED_KPOINT_POOLS EXPECTED_PARENT EXPECTED_PATH_ITERATION EXPECTED_PREFIX <<EOF
$(python - "${MANIFEST}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
resources = manifest["resources"]
parent = manifest["parent"]
print(
    resources["ntasks"],
    resources["kpoint_pools"],
    parent["remote_run_dir"],
    parent["latest_path_iteration"],
    parent.get("qe_prefix", "sb2te3"),
)
PY
)
EOF
KPOINT_POOLS="${MIGRATIONBENCH_KPOINT_POOLS:-${EXPECTED_KPOINT_POOLS}}"
test "${SLURM_NTASKS}" -eq "${EXPECTED_NTASKS}"
test "${KPOINT_POOLS}" -eq "${EXPECTED_KPOINT_POOLS}"
test "${MIGRATIONBENCH_PARENT_RUN_DIR}" = "${EXPECTED_PARENT}"
test -s "${EXPECTED_PARENT}/out/${EXPECTED_PREFIX}.path${EXPECTED_PATH_ITERATION}"

inventory() {
  local root="$1"
  (
    cd "${root}"
    find out -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
  )
}

inventory "${EXPECTED_PARENT}" > parent_restart_inventory.sha256
mkdir -p out
rsync -a --checksum "${EXPECTED_PARENT}/out/" "${JOB_RUN_DIR}/out/"
inventory "${JOB_RUN_DIR}" > copied_restart_inventory.sha256
diff -u parent_restart_inventory.sha256 copied_restart_inventory.sha256 > restart_inventory.diff

python - "${MANIFEST}" <<'PY'
import hashlib, json, sys
from pathlib import Path
path = Path(sys.argv[1])
manifest = json.loads(path.read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest["copy_verification"] = {
    "parent_inventory": "parent_restart_inventory.sha256",
    "parent_inventory_sha256": sha("parent_restart_inventory.sha256"),
    "copied_inventory": "copied_restart_inventory.sha256",
    "copied_inventory_sha256": sha("copied_restart_inventory.sha256"),
    "inventory_diff": "restart_inventory.diff",
    "inventory_match": Path("restart_inventory.diff").stat().st_size == 0,
}
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

if [ "${MIGRATIONBENCH_COPY_SMOKE_ONLY:-0}" = "1" ]; then
  echo "copy_smoke_only=passed"
  exit 0
fi

ulimit -s unlimited || true
export OMP_NUM_THREADS=1
echo "host=$(hostname)"
echo "job_run_dir=${JOB_RUN_DIR}"
echo "parent_run_dir=${EXPECTED_PARENT}"
echo "slurm_ntasks=${SLURM_NTASKS}"
echo "kpoint_pools=${KPOINT_POOLS}"
which neb.x
python "${CODE_ROOT}/scripts/migrationbench/capture_runtime_provenance.py" \
  --output runtime_provenance.json --binary neb.x --binary pw.x \
  --input "neb_input=${JOB_RUN_DIR}/neb.in" \
  --input "restart_manifest=${JOB_RUN_DIR}/qe_restart_manifest.json" \
  --input "parent_inventory=${JOB_RUN_DIR}/parent_restart_inventory.sha256"

mpirun -np "${SLURM_NTASKS}" neb.x -nk "${KPOINT_POOLS}" -inp neb.in > neb.out
