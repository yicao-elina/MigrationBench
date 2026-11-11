#!/bin/bash -l
#SBATCH --job-name=seedreplica-smoke
#SBATCH --array=0-7
#SBATCH --account=pclancy3_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch16/pclancy3/yi/mace_four_model_seedreplica_smoke_20261004/logs/%A_%a.out
#SBATCH --error=/scratch16/pclancy3/yi/mace_four_model_seedreplica_smoke_20261004/logs/%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=4
PY=/data/pclancy3/yi/.conda/envs/mace/bin/python
ROOT=${SMOKE_ROOT:-/scratch16/pclancy3/yi/mace_four_model_seedreplica_smoke_20261004}
INPUT=${SMOKE_INPUT:-/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/AI4MAT25/mlff-benchmark-cr-sb2te3/results/MD/3-multi_T-fine-tuning/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz}
SCRIPT=${SMOKE_SCRIPT:-$ROOT/seedreplica_smoke.py}

case "${SLURM_ARRAY_TASK_ID}" in
  0) LABEL=Foundation-medium-mpa-0; TRAIN=0; MODEL=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/X-FORCE/xforce/models/mace-mpa-0-medium.model ;;
  1) LABEL=Foundation-MH-1; TRAIN=1; MODEL=/scratch16/pclancy3/yi/foundation_grouped_benchmark_20261001/models/mace-mh-1.model ;;
  2) LABEL=From-scratch-seed234; TRAIN=234; MODEL=/scratch16/pclancy3/yi/revision1_migrationbench_runs/ft600k_grouped_A2/retrain_matrix_scratch16_20260927/scratch_grouped_seed234/scratch_grouped_seed234_compiled.model ;;
  3) LABEL=From-scratch-seed345; TRAIN=345; MODEL=/scratch16/pclancy3/yi/revision1_migrationbench_runs/ft600k_grouped_A2/retrain_matrix_scratch16_20260927/scratch_grouped_seed345/scratch_grouped_seed345_compiled.model ;;
  4) LABEL=FT-600K-seed234; TRAIN=234; MODEL=/scratch16/pclancy3/yi/revision1_migrationbench_runs/ft600k_grouped_A2/retrain_matrix_scratch16_20260927/ft600k_omatft_grouped_seed234/ft600k_omatft_grouped_seed234_compiled.model ;;
  5) LABEL=FT-600K-seed345; TRAIN=345; MODEL=/scratch16/pclancy3/yi/revision1_migrationbench_runs/ft600k_grouped_A2/retrain_matrix_scratch16_20260927/ft600k_omatft_grouped_seed345/ft600k_omatft_grouped_seed345_compiled.model ;;
  6) LABEL=FT-MultiT-seed234; TRAIN=234; MODEL=/scratch16/pclancy3/yi/revision1_migrationbench_runs/multit_seed_matrix_20261001/ft_multit_seed234/ft_multit_seed234_compiled.model ;;
  7) LABEL=FT-MultiT-seed345; TRAIN=345; MODEL=/scratch16/pclancy3/yi/revision1_migrationbench_runs/multit_seed_matrix_20261001/ft_multit_seed345/ft_multit_seed345_compiled.model ;;
  *) echo "invalid array index" >&2; exit 2 ;;
esac

exec "$PY" "$SCRIPT" --input "$INPUT" --model "$MODEL" \
  --model-label "$LABEL" --training-seed "$TRAIN" --md-seed 1100 \
  --output "$ROOT/$LABEL" --steps 20000 --start-step 200000 \
  --temperature-K 600 --device cuda
