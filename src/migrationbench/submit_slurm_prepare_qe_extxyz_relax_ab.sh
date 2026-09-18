#!/bin/bash
#SBATCH --job-name=mb_prep_qegeom_ab_s47
#SBATCH --time=00:10:00
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

CODE_ROOT="${MIGRATIONBENCH_CODE_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_pipeline}"
RUN_ROOT="${MIGRATIONBENCH_RUN_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_smoke/runs}"
OUT="${RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}/qe_relax_ab"
ROOT="${CODE_ROOT}/data_processed/path_repairs/coverage_gaps_pbc_v3"
cd "${CODE_ROOT}"

python scripts/migrationbench/prepare_qe_extxyz_relax_ab.py \
  --direct "${ROOT}/1-3_periodic_direct.extxyz" \
  --transformed "${ROOT}/1-3_periodic_repaired.extxyz" \
  --transform-manifest "${ROOT}/1-3_periodic_repaired.manifest.json" \
  --template "${CODE_ROOT}/data_raw/historical_neb/legacy_61/1-3/pw_1.in" \
  --path-id 1-3 --image-index-zero 1 --image-index-zero 2 \
  --out-dir "${OUT}/1-3" --seed 47

python scripts/migrationbench/prepare_qe_extxyz_relax_ab.py \
  --direct "${ROOT}/1-5_periodic_direct.extxyz" \
  --transformed "${ROOT}/1-5_periodic_repaired.extxyz" \
  --transform-manifest "${ROOT}/1-5_periodic_repaired.manifest.json" \
  --template "${CODE_ROOT}/data_raw/historical_neb/legacy_61/1-5/pw_1.in" \
  --path-id 1-5 --image-index-zero 1 \
  --out-dir "${OUT}/1-5" --seed 47

python - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
manifests = sorted(root.glob("*/*/*/endpoint_relax_manifest.json"))
assert len(manifests) == 6
rows = [json.load(open(path)) for path in manifests]
assert all(row["seed"] == 47 for row in rows)
assert all(row["transform"]["maximum_host_displacement_A"] <= 1e-8 for row in rows)
print("QE geometry-transform A/B preparation gates passed")
PY
