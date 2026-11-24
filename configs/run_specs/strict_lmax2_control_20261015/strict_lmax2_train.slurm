#!/bin/bash -l
# Strict same-parameter Lmax2 training; only TRAINING_SEED and OUTPUT_ROOT vary.
# This file is deployed to Rockfish after local hash capture.
#SBATCH --job-name=mb-strict-lmax2
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
source /etc/profile.d/modules.sh
source /data/apps/go.sh
source /data/apps/helpers/singularity.sh
ml openmpi/4.1.6
export OMP_NUM_THREADS=4
module restore
source ~/.bashrc
conda activate mace

BASE=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer
OUTPUT_ROOT=/scratch16/pclancy3/yi/migrationbench_strict_lmax2_20261015
SEEDS=(123 345)
TRAINING_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="strict_lmax2_seed_${TRAINING_SEED}"
RESULTS_DIR="${OUTPUT_ROOT}/${MODEL_NAME}"
LOG_DIR="${RESULTS_DIR}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
cd "$BASE"

echo "strict_lmax2_blueprint=20261015"
echo "model_name=$MODEL_NAME"
echo "training_seed=$TRAINING_SEED"
echo "base=$BASE"
echo "python=$(command -v python)"
python --version
# Keep this legacy argument identical to the seed234 launch. MACE logs
# confirm that num_channels+max_L deterministically construct the actual
# hidden irreps used by the model.
mace_run_train \
    --name="$MODEL_NAME" \
    --model_dir="$RESULTS_DIR" \
    --log_dir="$LOG_DIR" \
    --checkpoints_dir="$RESULTS_DIR" \
    --results_dir="$RESULTS_DIR" \
    --train_file="$BASE/data/train.xyz" \
    --valid_file="$BASE/data/valid.xyz" \
    --test_file="$BASE/data/test.xyz" \
    --model="MACE" \
    --num_channels="32" \
    --max_L="2" \
    --correlation="2" \
    --r_max="6.0" \
    --hidden_irreps="128x0e" \
    --batch_size="32" \
    --max_num_epochs="1000" \
    --forces_weight="10.0" \
    --swa --start_swa=800 --ema --ema_decay=0.99 \
    --device="cuda" --seed="$TRAINING_SEED" --patience=20 \
    --energy_key="energy" --forces_key="forces" \
    --E0s="isolated" --valid_batch_size=16 --eval_interval=50

echo "training_complete=$MODEL_NAME"
