#!/bin/bash
#SBATCH --job-name=mb_qenebprov_s42
#SBATCH --time=00:05:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=parallel
#SBATCH -A pclancy3
#SBATCH --mem=4G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail

source /data/apps/go.sh
source ~/.bashrc
conda activate mace
module load qe/7.3.1-cpu
set -u

SEED="${MIGRATIONBENCH_SEED:-42}"
CODE_ROOT="${MIGRATIONBENCH_CODE_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_pipeline}"
RUN_ROOT="${MIGRATIONBENCH_RUN_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_smoke/runs}"
JOB_RUN_DIR="${RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${JOB_RUN_DIR}"
cd "${CODE_ROOT}"

test "${SEED}" = "42"
printf 'BEGIN\nEND\n' > "${JOB_RUN_DIR}/neb.in"
python scripts/migrationbench/capture_runtime_provenance.py \
  --output "${JOB_RUN_DIR}/runtime_provenance.json" \
  --binary neb.x --binary pw.x \
  --input "neb_input=${JOB_RUN_DIR}/neb.in"

python - "${JOB_RUN_DIR}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/migrationbench")
from validate_runtime_provenance import validate_runtime_provenance

root = Path(sys.argv[1])
expected = hashlib.sha256((root / "neb.in").read_bytes()).hexdigest()
result = validate_runtime_provenance(
    root / "runtime_provenance.json",
    {"neb_input": expected},
    ["neb.x", "pw.x"],
    str(root.name.rsplit("_", 1)[-1]),
)
assert result["passed"], json.dumps(result, indent=2)
(root / "runtime_validation.json").write_text(json.dumps(result, indent=2) + "\n")
print("real_qe_runtime_validation=pass")
PY

python -m pytest -q \
  tests/test_representative_path_pipeline.py::RepresentativePathPipelineTest::test_qe_neb_ab_comparison_rejects_prefix_and_mechanism_switch \
  tests/test_representative_path_pipeline.py::RepresentativePathPipelineTest::test_qe_neb_runtime_gate_requires_neb_and_pw_for_every_segment \
  | tee "${JOB_RUN_DIR}/pytest.out"
