#!/bin/bash
#SBATCH --job-name=crsb2te3_grp_smoke
#SBATCH --time=00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=parallel
#SBATCH -A pclancy3
#SBATCH --mem=48G

set -eo pipefail
source /data/apps/go.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
set -u
: "${CAMPAIGN_ROOT:?}"

PYTHON=/data/pclancy3/yi/.conda/envs/mace/bin/python
INPUT_ROOT=/scratch16/pclancy3/yi/revision1_migrationbench_inputs/endpoint_mace_relax_s42_20260915/inputs
REGISTRY="${CAMPAIGN_ROOT}/model_registry.tsv"
line=$((SLURM_ARRAY_TASK_ID + 2))
MODEL_ID=$(awk -F '\t' -v n="${line}" 'NR==n {print $1}' "${REGISTRY}")
MODEL_PATH=$(awk -F '\t' -v n="${line}" 'NR==n {print $2}' "${REGISTRY}")
TASK_JSON="${CAMPAIGN_ROOT}/task_json/epmlff_001_unique_020_base.json"
OUT_DIR="${CAMPAIGN_ROOT}/smoke/${MODEL_ID}/attempt_${SLURM_JOB_ID}"
mkdir -p "${OUT_DIR}"
"${PYTHON}" "${CAMPAIGN_ROOT}/code/run_endpoint_relax.py" \
  --task-json "${TASK_JSON}" --input-root "${INPUT_ROOT}" --out-dir "${OUT_DIR}" \
  --model-label "${MODEL_ID}" --model-path "${MODEL_PATH}" --device cpu --steps 500 --fmax 0.03 --smoke-only
