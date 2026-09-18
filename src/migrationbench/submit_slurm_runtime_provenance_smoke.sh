#!/bin/bash
#SBATCH --job-name=mb_runtimeprov_s42
#SBATCH --time=00:10:00
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
module restore || true
source ~/.bashrc
conda activate mace
module load qe/7.3.1-cpu
set -u

SEED="${MIGRATIONBENCH_SEED:-42}"
test "${SEED}" -eq 42
CODE_ROOT="${MIGRATIONBENCH_CODE_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_pipeline}"
RUN_ROOT="${MIGRATIONBENCH_RUN_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_smoke/runs}"
JOB_RUN_DIR="${RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${JOB_RUN_DIR}"
printf 'runtime-provenance-smoke\n' > "${JOB_RUN_DIR}/fixture.in"

python "${CODE_ROOT}/scripts/migrationbench/capture_runtime_provenance.py" \
  --output "${JOB_RUN_DIR}/runtime_provenance.json" \
  --binary python --binary pw.x --binary neb.x \
  --input "fixture=${JOB_RUN_DIR}/fixture.in"

python - "${JOB_RUN_DIR}/runtime_provenance.json" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1]))
assert record["environment"]["SLURM_JOB_ID"]
assert record["environment"]["CONDA_DEFAULT_ENV"] == "mace"
assert record["packages"]["ase"]["module_version"] == "3.25.0"
assert record["packages"]["mace-torch"]["distribution_version"] == "0.3.12"
assert record["packages"]["torch"]["module_version"] == "2.7.0+cu118"
assert record["packages"]["pandas"]["module_version"] == "2.3.1"
assert record["packages"]["pandas"]["distribution_version"] == "2.2.3"
for package in ("ase", "pandas", "torch"):
    assert len(record["packages"][package]["module_file_sha256"]) == 64
for name in ("python", "pw.x", "neb.x"):
    assert record["binaries"][name]["exists"], name
    assert len(record["binaries"][name]["sha256"]) == 64, name
assert record["inputs"]["fixture"]["exists"]
assert len(record["inputs"]["fixture"]["sha256"]) == 64
print("runtime provenance smoke gates passed")
PY
