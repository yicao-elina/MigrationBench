#!/bin/bash
#SBATCH --job-name=nbr
#SBATCH --time=24:00:0
#SBATCH -N  1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --partition=a100
#SBATCH -A pclancy3_gpu
#SBATCH --gres=gpu:1

source /data/apps/go.sh
source /data/apps/helpers/singularity.sh
ml openmpi/4.1.6
export OMP_NUM_THREADS=4
date
module restore
source ~/.bashrc

conda activate mace
module list
pwd

# Define model name and directories
MODEL_NAME="finetuned_neb" # Changed from mace_l1
MODEL_DIR="../finetuned_model/neb"     
LOG_DIR="../finetuned_model/neb"
CHECKPOINTS_DIR="../finetuned_model/neb"
RESULTS_DIR="../finetuned_model/neb"

# Define data file paths
dat_dir="../data/data_neb"
TRAIN_FILE="${dat_dir}/train.xyz"
VALID_FILE="${dat_dir}/valid.xyz"
TEST_FILE="${dat_dir}/test.xyz"

# Set device (CPU, GPU, or Apple Silicon)
DEVICE="cuda"

# Energy and forces keys in the extxyz file
ENERGY_KEY="free_energy"
FORCES_KEY="forces"

# max_L, r_max, and hidden_irreps are set in the model
MAX_L=1
R_MAX=6.0

# Define training hyperparameters
BATCH_SIZE=32
VALID_BATCH_SIZE=16
MAX_NUM_EPOCHS=1000
SWA_START=200           
EVAL_INTERVAL=50
PATIENCE=20
SEED=234

# E0 settings
E0S="isolated"

# Define additional MACE parameters from the config
FORCES_WEIGHT=10.0         # Added
ENERGY_WEIGHT=1.0          # Added
STRESS_WEIGHT=0.0          # Added
FOUNDATION_MODEL="/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/MACE-matpes-pbe-omat-ft.model" # Added
MULTIHEADS_FINETUNING="True" # Added
RESTART="True" # Added

# Start the training using mace_run_train
mace_run_train \
    --name="${MODEL_NAME}" \
    --model_dir="${MODEL_DIR}" \
    --log_dir="${LOG_DIR}" \
    --checkpoints_dir="${CHECKPOINTS_DIR}" \
    --results_dir="${RESULTS_DIR}" \
    --train_file="${TRAIN_FILE}" \
    --valid_file="${VALID_FILE}" \
    --test_file="${TEST_FILE}" \
    --model="MACE" \
    --num_channels=32 \
    --max_L="${MAX_L}" \
    --r_max="${R_MAX}" \
    --hidden_irreps='128x0e + 128x1o' \
    --batch_size="${BATCH_SIZE}" \
    --valid_batch_size="${VALID_BATCH_SIZE}" \
    --max_num_epochs="${MAX_NUM_EPOCHS}" \
    --swa \
    --start_swa="${SWA_START}" \
    --ema \
    --ema_decay=0.99 \
    --device="${DEVICE}" \
    --energy_key="${ENERGY_KEY}" \
    --forces_key="${FORCES_KEY}" \
    --E0s="${E0S}" \
    --seed="${SEED}" \
    --eval_interval="${EVAL_INTERVAL}" \
    --patience="${PATIENCE}" \
    --forces_weight="${FORCES_WEIGHT}" \
    --energy_weight="${ENERGY_WEIGHT}" \
    --stress_weight="${STRESS_WEIGHT}" \
    --foundation_model="${FOUNDATION_MODEL}" \
    --multiheads_finetuning="${MULTIHEADS_FINETUNING}" \
    --restart_latest 
# Can be added: 
#     --restart_latest \

date