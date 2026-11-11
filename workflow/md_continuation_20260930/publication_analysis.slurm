#!/bin/bash -l
#SBATCH --account=pclancy3
#SBATCH --partition=parallel
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=publication-%j.out
#SBATCH --error=publication-%j.err
set -euo pipefail
export MACE_ENV=/data/pclancy3/yi/.conda/envs/mace
export PATH=$MACE_ENV/bin:$PATH
export PYTHONPATH=/data/pclancy3/yi/.python_pkgs:/data/pclancy3/yi/src
export MPLBACKEND=Agg
exec "$MACE_ENV/bin/python" /scratch16/pclancy3/yi/mace_four_model_MD_continuation_20260930/publication_metrics.py \
  --root /scratch16/pclancy3/yi/mace_four_model_MD_continuation_20260930 \
  --output /scratch16/pclancy3/yi/mace_four_model_MD_analysis_20261003 \
  --mode "$MODE" --model-dir "$MODEL_DIR"
