#!/bin/bash
#SBATCH --job-name=mb_pathrepair_pbc_s47
#SBATCH --time=00:15:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=parallel
#SBATCH -A pclancy3
#SBATCH --mem=8G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail

source /data/apps/go.sh
module restore || true
source ~/.bashrc
conda activate mace
set -u

SEED="${MIGRATIONBENCH_SEED:-47}"
CODE_ROOT="${MIGRATIONBENCH_CODE_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_pipeline}"
RUN_ROOT="${MIGRATIONBENCH_RUN_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_smoke/runs}"
JOB_RUN_DIR="${RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
OUT_DIR="${JOB_RUN_DIR}/coverage_gaps_pbc_v3"
mkdir -p "${OUT_DIR}"
cd "${CODE_ROOT}"

python -m pytest -q

for path_id in 1-3 1-5; do
  source="${CODE_ROOT}/data_raw/historical_neb/legacy_61/${path_id}/sb2te3.xyz"
  template="${CODE_ROOT}/data_raw/historical_neb/legacy_61/${path_id}/pw_1.in"
  python scripts/migrationbench/repair_neb_images.py \
    --images "${source}" --cell-template "${template}" \
    --output "${OUT_DIR}/${path_id}_periodic_direct.extxyz" \
    --manifest "${OUT_DIR}/${path_id}_periodic_direct.manifest.json" \
    --mode unwrap \
    --min-pair-distance-A 0 --repel-policy none --fail-on-invalid
  python scripts/migrationbench/repair_neb_images.py \
    --images "${source}" --cell-template "${template}" \
    --output "${OUT_DIR}/${path_id}_periodic_repaired.extxyz" \
    --manifest "${OUT_DIR}/${path_id}_periodic_repaired.manifest.json" \
    --mode unwrap \
    --min-pair-distance-A 1.8 --repel-policy migrant-only \
    --repel-steps 100 --repel-step-size-A 0.05 --fail-on-invalid
done

python - "${OUT_DIR}" "${SEED}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
assert int(sys.argv[2]) == 47
for manifest_path in sorted(root.glob("*.manifest.json")):
    manifest = json.load(open(manifest_path))
    assert manifest["output_pbc"] == [True, True, True]
    assert manifest["endpoints_pass"] is True
    if "repaired" in manifest_path.name:
        assert manifest["all_intermediate_images_pass"] is True
print("periodic path-repair smoke gates passed")
PY
