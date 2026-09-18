#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --partition=parallel
#SBATCH --account=pclancy3

set -euo pipefail

: "${MIGRATIONBENCH_AUDIT_CONFIG:?set MIGRATIONBENCH_AUDIT_CONFIG}"
: "${MIGRATIONBENCH_AUDIT_SCRIPT:?set MIGRATIONBENCH_AUDIT_SCRIPT}"
: "${MIGRATIONBENCH_AUDIT_RUN_ROOT:?set MIGRATIONBENCH_AUDIT_RUN_ROOT}"

set +u
source ~/.bashrc
set -u
MIGRATIONBENCH_AUDIT_RUN_DIR="${MIGRATIONBENCH_AUDIT_RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${MIGRATIONBENCH_AUDIT_RUN_DIR}"
cp "${MIGRATIONBENCH_AUDIT_CONFIG}" "${MIGRATIONBENCH_AUDIT_RUN_DIR}/audit_config.json"
cp "${MIGRATIONBENCH_AUDIT_SCRIPT}" "${MIGRATIONBENCH_AUDIT_RUN_DIR}/audit_neb_training_leakage.py"

mapfile -d '' -t python_args < <(
  python -c 'import json,sys; [sys.stdout.write(value + "\0") for value in json.load(open(sys.argv[1]))["argv"]]' \
    "${MIGRATIONBENCH_AUDIT_CONFIG}"
)
cd "${MIGRATIONBENCH_AUDIT_RUN_DIR}"
conda run -n mace python audit_neb_training_leakage.py "${python_args[@]}" > audit.stdout 2> audit.stderr
test -s results/leakage_audit.json
