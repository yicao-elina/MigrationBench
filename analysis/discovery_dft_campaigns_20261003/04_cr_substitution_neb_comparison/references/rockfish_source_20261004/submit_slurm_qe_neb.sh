#!/bin/bash
#SBATCH --job-name=mb_qe_neb_s42
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
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

: "${MIGRATIONBENCH_NEB_INPUT:?set MIGRATIONBENCH_NEB_INPUT to the generated QE neb.in}"
RUN_ROOT="${MIGRATIONBENCH_RUN_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_runs}"
CODE_ROOT="${MIGRATIONBENCH_CODE_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
QE_EXTRA_ARGS="${MIGRATIONBENCH_QE_EXTRA_ARGS:-}"
JOB_RUN_DIR="${RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "${JOB_RUN_DIR}"

cp "${MIGRATIONBENCH_NEB_INPUT}" "${JOB_RUN_DIR}/neb.in"
cd "${JOB_RUN_DIR}"

ulimit -s unlimited || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "host=$(hostname)"
echo "user=$(whoami)"
echo "seed=42"
echo "qe_module=qe/7.3.1-cpu"
echo "job_run_dir=${JOB_RUN_DIR}"
echo "source_neb_input=${MIGRATIONBENCH_NEB_INPUT}"
echo "slurm_ntasks=${SLURM_NTASKS}"
echo "slurm_mem_per_node=${SLURM_MEM_PER_NODE:-unset}"
echo "omp_num_threads=${OMP_NUM_THREADS}"
echo "qe_extra_args=${QE_EXTRA_ARGS}"
which neb.x
python "${CODE_ROOT}/scripts/migrationbench/capture_runtime_provenance.py" \
  --output runtime_provenance.json --binary neb.x --binary pw.x \
  --input "neb_input=${JOB_RUN_DIR}/neb.in"

mpirun -np "${SLURM_NTASKS}" neb.x ${QE_EXTRA_ARGS} -inp neb.in > neb.out
