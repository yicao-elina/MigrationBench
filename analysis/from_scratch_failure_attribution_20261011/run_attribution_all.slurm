#!/bin/bash -l
#SBATCH --job-name=fromscratch-attrib-all
#SBATCH --account=pclancy3
#SBATCH --partition=parallel
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%A.out
#SBATCH --error=slurm-%A.err

source /etc/profile
set -euo pipefail
export MACE_ENV=/data/pclancy3/yi/.conda/envs/mace
export PATH="$MACE_ENV/bin:$PATH"
export PYTHONPATH=/data/pclancy3/yi/.python_pkgs:/data/pclancy3/yi/src
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

ROOT=/scratch16/pclancy3/yi/mace_from_scratch_failure_attribution_20261011
ORIGINAL=/scratch16/pclancy3/yi/mace_from_scratch_ablation_MD_20261005
REPRO1=/scratch16/pclancy3/yi/mace_from_scratch_reproduce_20261011
REPRO2=/scratch16/pclancy3/yi/mace_from_scratch_reproduce_20261011_r02
OUT="$ROOT/results_all_replicates"
DATA=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/data
NEB=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/neb_benchmark/data

exec "$MACE_ENV/bin/python" "$ROOT/diagnose_from_scratch_all.py" \
  --original-root "$ORIGINAL" \
  --reproduce-root "$REPRO1" \
  --reproduce-label reproduce-1 \
  --reproduce-root "$REPRO2" \
  --reproduce-label reproduce-2 \
  --output "$OUT" \
  --data-root "$DATA" \
  --neb-root "$NEB" \
  --frames-per-reference 250 \
  --force-disagreement \
  --prepare-dft-dir "$OUT/dft_candidates_prepared_only"
