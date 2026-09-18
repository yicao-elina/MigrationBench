#!/bin/bash
#SBATCH --job-name=mb_mlff_neb_s42
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100
#SBATCH -A pclancy3_gpu
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail

source /data/apps/go.sh
source /data/apps/helpers/singularity.sh
ml openmpi/4.1.6 || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
module restore || true
source ~/.bashrc
conda activate mace
set -u

: "${MIGRATIONBENCH_IMAGES:?set MIGRATIONBENCH_IMAGES to an ASE-readable image trajectory}"
: "${MIGRATIONBENCH_PATH_ID:?set MIGRATIONBENCH_PATH_ID, e.g. 1-4}"
: "${MIGRATIONBENCH_MODEL_PATH:?set MIGRATIONBENCH_MODEL_PATH to a MACE checkpoint}"

SEED="${MIGRATIONBENCH_SEED:-42}"
STEPS="${MIGRATIONBENCH_STEPS:-500}"
FMAX="${MIGRATIONBENCH_FMAX:-0.05}"
SPRING="${MIGRATIONBENCH_SPRING:-0.1}"
CALCULATOR="${MIGRATIONBENCH_CALCULATOR:-mace-model}"
RUN_ROOT="${MIGRATIONBENCH_RUN_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_runs}"
CODE_ROOT="${MIGRATIONBENCH_CODE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
JOB_RUN_DIR="${RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
OUT_DIR="${JOB_RUN_DIR}/mlff_${MIGRATIONBENCH_PATH_ID}_s${SEED}"
mkdir -p "${OUT_DIR}"

echo "host=$(hostname)"
echo "user=$(whoami)"
echo "workdir=$(pwd)"
echo "seed=${SEED}"
echo "path_id=${MIGRATIONBENCH_PATH_ID}"
echo "images=${MIGRATIONBENCH_IMAGES}"
echo "calculator=${CALCULATOR}"
echo "model_path=${MIGRATIONBENCH_MODEL_PATH}"
echo "steps=${STEPS}"
echo "fmax=${FMAX}"
echo "spring=${SPRING}"
echo "run_root=${RUN_ROOT}"
echo "job_run_dir=${JOB_RUN_DIR}"
python --version
PROVENANCE_ARGS=(--output "${JOB_RUN_DIR}/runtime_provenance.json" --binary python \
  --input "images=${MIGRATIONBENCH_IMAGES}" --input "model=${MIGRATIONBENCH_MODEL_PATH}")
python "${CODE_ROOT}/scripts/migrationbench/capture_runtime_provenance.py" "${PROVENANCE_ARGS[@]}"

python scripts/migrationbench/run_mlff_neb.py \
  --images "${MIGRATIONBENCH_IMAGES}" \
  --out-dir "${OUT_DIR}" \
  --calculator "${CALCULATOR}" \
  --model-path "${MIGRATIONBENCH_MODEL_PATH}" \
  --device cuda \
  --steps "${STEPS}" \
  --seed "${SEED}" \
  --fmax "${FMAX}" \
  --spring-constant "${SPRING}"
