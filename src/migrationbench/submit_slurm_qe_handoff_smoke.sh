#!/bin/bash
#SBATCH --job-name=mb_qehandoff_sm_s44
#SBATCH --time=00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=shared
#SBATCH -A pclancy3
#SBATCH --mem=4G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail

: "${MIGRATIONBENCH_PAIR_BUNDLE:?set MIGRATIONBENCH_PAIR_BUNDLE}"
RUN_ROOT="${MIGRATIONBENCH_RUN_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_runs}"
JOB_RUN_DIR="${RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${JOB_RUN_DIR}"
cp "${MIGRATIONBENCH_PAIR_BUNDLE}/qe_handoff_pair_manifest.json" "${JOB_RUN_DIR}/"
cp -R "${MIGRATIONBENCH_PAIR_BUNDLE}/direct_baseline" "${JOB_RUN_DIR}/"
cp -R "${MIGRATIONBENCH_PAIR_BUNDLE}/mlff_preconditioned" "${JOB_RUN_DIR}/"
cd "${SLURM_SUBMIT_DIR}"

source /data/apps/go.sh
module load anaconda
source activate mace
set -u

printf 'job_id=%s\njob_name=%s\nseed=44\nrun_dir=%s\n' \
  "${SLURM_JOB_ID}" "${SLURM_JOB_NAME}" "${JOB_RUN_DIR}" > "${JOB_RUN_DIR}/slurm_identity.txt"

python scripts/migrationbench/validate_qe_neb_handoff_pair.py \
  --pair-manifest "${JOB_RUN_DIR}/qe_handoff_pair_manifest.json" \
  --output "${JOB_RUN_DIR}/qe_handoff_pair_validation.json"
