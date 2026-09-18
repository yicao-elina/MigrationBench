#!/bin/bash
#SBATCH --job-name=mb_remape2e_s42
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
mkdir -p "${JOB_RUN_DIR}"
cd "${CODE_ROOT}"

test "${SEED}" = "42"
python -m pytest -q \
  tests/test_representative_path_pipeline.py::RepresentativePathPipelineTest::test_endpoint_remap_materializes_both_historical_segments_end_to_end \
  | tee "${JOB_RUN_DIR}/pytest.out"

python - "${JOB_RUN_DIR}" "${SEED}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(sys.argv[1])
seed = int(sys.argv[2])
pytest_output = run_dir / "pytest.out"
manifest = {
    "schema_version": "1.0",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "scientific_role": "endpoint_remap_full_materialization_rockfish_smoke",
    "seed": seed,
    "test_output": str(pytest_output),
    "test_output_sha256": hashlib.sha256(pytest_output.read_bytes()).hexdigest(),
    "status": "passed",
}
(run_dir / "smoke_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY
