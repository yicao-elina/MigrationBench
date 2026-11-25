#!/bin/bash
# Independent unconverged-DFT diagnostic tier; no prior campaign is touched.
#SBATCH --job-name=mb-uconv-mace
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
ROOT="${RUN_ROOT:-/scratch16/pclancy3/yi/task1_foundation_path_diagnostic_20261003}"
mkdir -p "$ROOT/logs"
MACE_ENV="${MACE_ENV:-/data/pclancy3/yi/.conda/envs/mace}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
"$MACE_ENV/bin/python" "$ROOT/run_unconverged_mace_diagnostic.py" \
  --model "/scratch16/pclancy3/yi/cross_element_generalization_repair_20261002/broad_roster_ft_seed123/broad_roster_ft_seed123_compiled.model" \
  --out "$ROOT/results" \
  --paths 2L-Cu__neb-QLs 2L-Cu__neb-gap-diffuse 2L-Cu__neb-gap2QL \
          2L-Li__neb-QLs 2L-Li__neb-gap-diffuse 2L-Li__neb-gap2Ql \
          2L-Na__neb-gap-diffuse 2L-Na__neb-gap2QL
