#!/bin/bash
#SBATCH --job-name=mb_qe_restart_s42
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

: "${MIGRATIONBENCH_NEB_INPUT:?set MIGRATIONBENCH_NEB_INPUT to the generated QE neb.in}"
: "${MIGRATIONBENCH_QE_MAX_SECONDS:?set MIGRATIONBENCH_QE_MAX_SECONDS below the Slurm walltime}"

RUN_ROOT="${MIGRATIONBENCH_RUN_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_runs}"
PARENT_RUN_DIR="${MIGRATIONBENCH_PARENT_RUN_DIR:-}"
RESTART_MODE="${MIGRATIONBENCH_RESTART_MODE:-from_scratch}"
CHAIN_ID="${MIGRATIONBENCH_CHAIN_ID:-${SLURM_JOB_NAME}}"
WALLTIME="${MIGRATIONBENCH_WALLTIME:-${SLURM_TIMELIMIT:-unknown}}"
SAFETY_SECONDS="${MIGRATIONBENCH_SAFETY_SECONDS:-1800}"
QE_EXTRA_ARGS="${MIGRATIONBENCH_QE_EXTRA_ARGS:-}"
JOB_RUN_DIR="${RUN_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"

mkdir -p "${JOB_RUN_DIR}"

if [ -n "${PARENT_RUN_DIR}" ]; then
  echo "copying_restart_parent=${PARENT_RUN_DIR}"
  rsync -a \
    --exclude='slurm-*.out' \
    --exclude='slurm-*.err' \
    --exclude='neb.out' \
    --exclude='neb.in' \
    "${PARENT_RUN_DIR}/" "${JOB_RUN_DIR}/"
fi

python3 scripts/migrationbench/qe_neb_prepare_restart.py \
  --input "${MIGRATIONBENCH_NEB_INPUT}" \
  --output "${JOB_RUN_DIR}/neb.in" \
  --manifest "${JOB_RUN_DIR}/qe_restart_manifest.json" \
  --restart-mode "${RESTART_MODE}" \
  --max-seconds "${MIGRATIONBENCH_QE_MAX_SECONDS}" \
  --walltime "${WALLTIME}" \
  --safety-seconds "${SAFETY_SECONDS}" \
  --parent-run-dir "${PARENT_RUN_DIR}" \
  --chain-id "${CHAIN_ID}" \
  --notes "QE max_seconds is intentionally below Slurm walltime so neb.x can stop cleanly and write restart state."

cd "${JOB_RUN_DIR}"

ulimit -s unlimited || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "host=$(hostname)"
echo "user=$(whoami)"
echo "seed=42"
echo "qe_module=qe/7.3.1-cpu"
echo "job_run_dir=${JOB_RUN_DIR}"
echo "source_neb_input=${MIGRATIONBENCH_NEB_INPUT}"
echo "parent_run_dir=${PARENT_RUN_DIR}"
echo "restart_mode=${RESTART_MODE}"
echo "chain_id=${CHAIN_ID}"
echo "max_seconds=${MIGRATIONBENCH_QE_MAX_SECONDS}"
echo "slurm_ntasks=${SLURM_NTASKS}"
echo "slurm_mem_per_node=${SLURM_MEM_PER_NODE:-unset}"
echo "omp_num_threads=${OMP_NUM_THREADS}"
echo "qe_extra_args=${QE_EXTRA_ARGS}"
which neb.x

mpirun -np "${SLURM_NTASKS}" neb.x ${QE_EXTRA_ARGS} -inp neb.in > neb.out
