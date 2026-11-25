#!/bin/bash
# Independent single-point-only candidate evaluation. No relaxation and no writes to prior campaigns.
#SBATCH --job-name=mb-candidate-sp-20261001
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail
RUN_ROOT="${RUN_ROOT:-/scratch16/pclancy3/yi/mace_candidate_eval_20261001}"
source /etc/profile.d/modules.sh
set -u
MACE_ENV="${MACE_ENV:-/data/pclancy3/yi/.conda/envs/mace}"
MANIFEST="${MANIFEST:-$RUN_ROOT/candidate_manifest_remote.csv}"
DEVICE="${DEVICE:-cpu}"
export PYTHONPATH="$RUN_ROOT/python_pkgs:$RUN_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

"$MACE_ENV/bin/python" "$RUN_ROOT/run_single_points.py" \
  --manifest "$MANIFEST" \
  --models "$RUN_ROOT/models_rockfish.tsv" \
  --out "$RUN_ROOT/results" \
  --device "$DEVICE"
