#!/bin/bash
#SBATCH --job-name=mb_siteprep_s42
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

SEED="${MIGRATIONBENCH_SEED:-42}"
CODE_ROOT="${MIGRATIONBENCH_CODE_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_pipeline}"
RUN_ROOT="${MIGRATIONBENCH_RUN_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_smoke/runs}"
JOB_RUN_DIR="${RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
OUT_DIR="${JOB_RUN_DIR}/prospective_dft_v3_s42"
mkdir -p "${OUT_DIR}"
cd "${CODE_ROOT}"

test "${SEED}" = "42"
python scripts/migrationbench/prepare_site_proposer_dft_validation.py \
  --site-table data_processed/site_stability_proposer/v1_mace_screening/site_descriptors_and_oof_predictions.csv \
  --qe-input-root data_raw/site_search_qe/qe_inputs \
  --proposer-manifest data_processed/site_stability_proposer/v1_mace_screening/proposer_manifest.json \
  --reference-cell-input data_raw/site_search_qe/2L_octo_Cr1.in \
  --out-dir "${OUT_DIR}" --pairs 6 --seed "${SEED}" \
  --walltime 24:00:00 --max-seconds 84600 --nstep 200 --ntasks 2 --memory 160G \
  --prospective-dft-validation

python scripts/migrationbench/audit_qe_calculator_identity.py \
  --root "prospective=${OUT_DIR}" --out-dir "${OUT_DIR}/calculator_identity_audit"

python - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
batch = json.loads((root / "endpoint_relax_batch_manifest.json").read_text())
audit = json.loads((root / "calculator_identity_audit/calculator_identity_manifest.json").read_text())
assert len(batch["pairs"]) == 6
assert len(batch["jobs"]) == 12
assert len({job["void_family_group"] for job in batch["jobs"]}) == 12
assert batch["selection_frozen_before_dft_submission"] is True
assert audit["counts"] == {
    "inputs": 12,
    "parse_errors": 0,
    "calculator_identities": 1,
    "incomplete_identities": 0,
    "soc_enabled_inputs": 0,
    "spin_polarized_inputs": 12,
}
print("prospective site-proposer DFT preparation gates passed")
PY

python -m pytest -q \
  tests/test_representative_path_pipeline.py::RepresentativePathPipelineTest::test_site_proposer_prospective_batch_is_independent_and_cell_complete \
  tests/test_representative_path_pipeline.py::RepresentativePathPipelineTest::test_site_proposer_comparator_withholds_censored_yield \
  tests/test_representative_path_pipeline.py::RepresentativePathPipelineTest::test_site_proposer_comparator_binds_runtime_and_input_hashes \
  tests/test_representative_path_pipeline.py::RepresentativePathPipelineTest::test_endpoint_discovery_planner_requires_repeat_then_acceptance \
  | tee "${JOB_RUN_DIR}/pytest.out"
