#!/bin/bash
#SBATCH --job-name=mb_qe_relax_s42
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --partition=parallel
#SBATCH -A pclancy3
#SBATCH --mem=160G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -eo pipefail

source /data/apps/go.sh
module load qe/7.3.1-cpu
set -u

: "${MIGRATIONBENCH_RELAX_INPUT:?set MIGRATIONBENCH_RELAX_INPUT}"
: "${MIGRATIONBENCH_RELAX_MANIFEST:?set MIGRATIONBENCH_RELAX_MANIFEST}"
RUN_ROOT="${MIGRATIONBENCH_RUN_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_runs}"
CODE_ROOT="${MIGRATIONBENCH_CODE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
JOB_RUN_DIR="${RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${JOB_RUN_DIR}"
cp "${MIGRATIONBENCH_RELAX_INPUT}" "${JOB_RUN_DIR}/relax.in"
cp "${MIGRATIONBENCH_RELAX_MANIFEST}" "${JOB_RUN_DIR}/endpoint_relax_manifest.json"
cd "${JOB_RUN_DIR}"

ulimit -s unlimited || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "host=$(hostname)"
echo "user=$(whoami)"
echo "job_run_dir=${JOB_RUN_DIR}"
echo "source_relax_input=${MIGRATIONBENCH_RELAX_INPUT}"
echo "source_manifest=${MIGRATIONBENCH_RELAX_MANIFEST}"
echo "slurm_ntasks=${SLURM_NTASKS}"
echo "omp_num_threads=${OMP_NUM_THREADS}"
which pw.x
python "${CODE_ROOT}/scripts/migrationbench/capture_runtime_provenance.py" \
  --output runtime_provenance.json --binary pw.x \
  --input "relax_input=${JOB_RUN_DIR}/relax.in" \
  --input "relax_manifest=${JOB_RUN_DIR}/endpoint_relax_manifest.json"

mpirun -np "${SLURM_NTASKS}" pw.x -inp relax.in > relax.out
