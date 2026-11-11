#!/bin/bash -l
# Collect and audit all canonical static-screen chunks after the array ends.
#SBATCH --job-name=mb-canonical-639-collect
#SBATCH --account=pclancy3
#SBATCH --partition=parallel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch16/pclancy3/yi/canonical_model_matrix_20261016/639_screen_v1/logs/%x-%j.out
#SBATCH --error=/scratch16/pclancy3/yi/canonical_model_matrix_20261016/639_screen_v1/logs/%x-%j.err

set -euo pipefail
ROOT=/scratch16/pclancy3/yi/canonical_model_matrix_20261016/639_screen_v1
MACE_ENV=/data/pclancy3/yi/.conda/envs/mace
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$ROOT/analysis" "$ROOT/logs"
"$MACE_ENV/bin/python" "$ROOT/collect_mlip_results.py" \
  --root "$ROOT" \
  --manifest "$ROOT/candidate_manifest_639.csv" \
  --models "$ROOT/models_rockfish.tsv" \
  --result-dir results_production \
  --out "$ROOT/analysis" \
  --expected-chunks 64
