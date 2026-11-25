#!/bin/bash
#SBATCH --job-name=crsb2te3_grp_ep
#SBATCH --time=08:00:00
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
TASKS="${CAMPAIGN_ROOT}/task_json"

line=$((SLURM_ARRAY_TASK_ID + 2))
MODEL_ID=$(awk -F '\t' -v n="${line}" 'NR==n {print $1}' "${REGISTRY}")
MODEL_PATH=$(awk -F '\t' -v n="${line}" 'NR==n {print $2}' "${REGISTRY}")

: "${MODEL_ID:?}" "${MODEL_PATH:?}"
RESULTS="${CAMPAIGN_ROOT}/runs/${MODEL_ID}"
mkdir -p "${RESULTS}"

for task_json in "${TASKS}"/*.json; do
  task_id=$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_id"])' "${task_json}")
  task_root="${RESULTS}/${task_id}"
  if find "${task_root}" -name mace_relax_summary.json -type f -exec "${PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("SUCCESS" if d.get("scientific_success") is True else "")' {} \; 2>/dev/null | grep -q '^SUCCESS$'; then
    continue
  fi
  out_dir="${task_root}/attempt_${SLURM_JOB_ID}"
  mkdir -p "${out_dir}"
  "${PYTHON}" "${CAMPAIGN_ROOT}/code/run_endpoint_relax.py" \
    --task-json "${task_json}" --input-root "${INPUT_ROOT}" --out-dir "${out_dir}" \
    --model-label "${MODEL_ID}" --model-path "${MODEL_PATH}" --device cpu --steps 500 --fmax 0.03
done
