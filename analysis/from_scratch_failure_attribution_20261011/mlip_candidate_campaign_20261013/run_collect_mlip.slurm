#!/bin/bash -l
#SBATCH --job-name=mb-mlip-collect
#SBATCH --account=pclancy3
#SBATCH --partition=parallel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
source /etc/profile.d/modules.sh
RUN_ROOT="${RUN_ROOT:-/scratch16/pclancy3/yi/mace_candidate_eval_20261013}"
MACE_ENV="${MACE_ENV:-/data/pclancy3/yi/.conda/envs/mace}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$RUN_ROOT/analysis" "$RUN_ROOT/logs"
"$MACE_ENV/bin/python" "$RUN_ROOT/collect_mlip_results.py" \
  --root "$RUN_ROOT" \
  --manifest "$RUN_ROOT/candidate_manifest_639.csv" \
  --models "$RUN_ROOT/models_rockfish.tsv" \
  --result-dir results_production \
  --out "$RUN_ROOT/analysis"
