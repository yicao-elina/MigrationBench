#!/bin/bash
#SBATCH --job-name=mb_mlff_neb_cpu_s42
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=parallel
#SBATCH -A pclancy3
#SBATCH --mem=48G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail

source /data/apps/go.sh
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
module restore || true
source ~/.bashrc
conda activate mace
set -u

: "${MIGRATIONBENCH_IMAGES:?set MIGRATIONBENCH_IMAGES to an ASE-readable image trajectory}"
: "${MIGRATIONBENCH_PATH_ID:?set MIGRATIONBENCH_PATH_ID, e.g. 81_neb_2}"
SEED="${MIGRATIONBENCH_SEED:-42}"
STEPS="${MIGRATIONBENCH_STEPS:-500}"
FMAX="${MIGRATIONBENCH_FMAX:-0.05}"
SPRING="${MIGRATIONBENCH_SPRING:-0.1}"
MIGRANT_ELEMENT="${MIGRATIONBENCH_MIGRANT_ELEMENT:-Cr}"
MIGRANT_TETHER_K="${MIGRATIONBENCH_MIGRANT_TETHER_K:-0.0}"
HOST_TETHER_K="${MIGRATIONBENCH_HOST_TETHER_K:-0.0}"
CALCULATOR="${MIGRATIONBENCH_CALCULATOR:-mace-model}"
MODEL_PATH="${MIGRATIONBENCH_MODEL_PATH:-}"
RUN_ROLE="${MIGRATIONBENCH_RUN_ROLE:-production_preconditioner}"
SOURCE_MANIFEST="${MIGRATIONBENCH_SOURCE_CANDIDATE_MANIFEST:-}"
SUBMITTED_CODE_DIR="${MIGRATIONBENCH_SUBMITTED_CODE_DIR:-}"
if [ "${CALCULATOR}" = "mace-model" ] && [ -z "${MODEL_PATH}" ]; then
  echo "MIGRATIONBENCH_MODEL_PATH is required for mace-model" >&2
  exit 2
fi
RUN_ROOT="${MIGRATIONBENCH_RUN_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_runs}"
CODE_ROOT="${MIGRATIONBENCH_CODE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
JOB_RUN_DIR="${RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
OUT_DIR="${JOB_RUN_DIR}/mlff_${MIGRATIONBENCH_PATH_ID}_s${SEED}"
mkdir -p "${OUT_DIR}"

echo "host=$(hostname)"
echo "user=$(whoami)"
echo "seed=${SEED}"
echo "path_id=${MIGRATIONBENCH_PATH_ID}"
echo "images=${MIGRATIONBENCH_IMAGES}"
echo "calculator=${CALCULATOR}"
echo "model_path=${MODEL_PATH}"
echo "run_role=${RUN_ROLE}"
echo "source_candidate_manifest=${SOURCE_MANIFEST}"
echo "submitted_code_dir=${SUBMITTED_CODE_DIR}"
echo "steps=${STEPS}"
echo "fmax=${FMAX}"
echo "spring=${SPRING}"
echo "migrant_element=${MIGRANT_ELEMENT}"
echo "migrant_tether_k=${MIGRANT_TETHER_K}"
echo "host_tether_k=${HOST_TETHER_K}"
echo "device=cpu"
echo "run_root=${RUN_ROOT}"
echo "job_run_dir=${JOB_RUN_DIR}"
python --version
PROVENANCE_ARGS=(--output "${JOB_RUN_DIR}/runtime_provenance.json" --binary python \
  --input "images=${MIGRATIONBENCH_IMAGES}")
if [ -n "${MODEL_PATH}" ]; then
  PROVENANCE_ARGS+=(--input "model=${MODEL_PATH}")
fi
if [ -n "${SOURCE_MANIFEST}" ]; then
  PROVENANCE_ARGS+=(--input "source_manifest=${SOURCE_MANIFEST}")
fi
if [ -n "${SUBMITTED_CODE_DIR}" ]; then
  PROVENANCE_ARGS+=(--input "submitted_runner=${SUBMITTED_CODE_DIR}/run_mlff_neb.py")
  PROVENANCE_ARGS+=(--input "submitted_wrapper=${SUBMITTED_CODE_DIR}/submit_slurm_mlff_neb_cpu.sh")
  PROVENANCE_ARGS+=(--input "submitted_history_exporter=${SUBMITTED_CODE_DIR}/export_mlff_neb_iteration_history.py")
  PROVENANCE_ARGS+=(--input "submitted_code_checksums=${SUBMITTED_CODE_DIR}/SHA256SUMS")
fi
python "${CODE_ROOT}/scripts/migrationbench/capture_runtime_provenance.py" "${PROVENANCE_ARGS[@]}"

python scripts/migrationbench/run_mlff_neb.py \
  --images "${MIGRATIONBENCH_IMAGES}" \
  --out-dir "${OUT_DIR}" \
  --calculator "${CALCULATOR}" \
  ${MODEL_PATH:+--model-path "${MODEL_PATH}"} \
  --device cpu \
  --steps "${STEPS}" \
  --seed "${SEED}" \
  --fmax "${FMAX}" \
  --spring-constant "${SPRING}" \
  --migrant-element "${MIGRANT_ELEMENT}" \
  --migrant-tether-k "${MIGRANT_TETHER_K}" \
  --host-tether-k "${HOST_TETHER_K}" \
  --run-role "${RUN_ROLE}" \
  ${SOURCE_MANIFEST:+--source-candidate-manifest "${SOURCE_MANIFEST}"}
