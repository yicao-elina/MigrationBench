#!/bin/bash -l
#SBATCH --account=pclancy3
#SBATCH --partition=parallel
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=analysis-%j.out
#SBATCH --error=analysis-%j.err
set -euo pipefail
export MACE_ENV=/data/pclancy3/yi/.conda/envs/mace
export PATH=$MACE_ENV/bin:$PATH
export PYTHONPATH=/data/pclancy3/yi/.python_pkgs:/data/pclancy3/yi/src
export MPLBACKEND=Agg
cd "$RUN_DIR"
step=$(sed -n 's/.*"last_step": \([0-9]*\).*/\1/p' status.json)
test "$step" -ge 1000000 || { echo "INCOMPLETE last_step=$step"; exit 2; }
exec "$MACE_ENV/bin/python" /scratch16/pclancy3/yi/mace_four_model_MD_continuation_20260930/analyze_md.py \
  --name "$MODEL_NAME" --original "$INPUT" --segment "$RUN_DIR"/segment_from_*.extxyz --out "$RUN_DIR/analysis_final"
