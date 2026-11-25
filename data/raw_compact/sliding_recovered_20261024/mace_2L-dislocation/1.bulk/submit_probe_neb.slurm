#!/bin/bash
#SBATCH --job-name=nbr
#SBATCH --time=2:00:0
#SBATCH -N  1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=a100
#SBATCH -A pclancy3_gpu
#SBATCH --gres=gpu:1

source /data/apps/go.sh
source /data/apps/helpers/singularity.sh
date
source ~/.bashrc

conda activate sklearn-env
module list

echo "Running compare_mace_representations.py"

SCRATCH_MODEL_PATH="/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE_models_l1_0802/mace_l1_0802_run-123.model"
FINETUNED_MODEL_PATH="/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_600K/finetuned_MACE_multihead0804_run-123.model"
PROBE_DATASET_PATH="/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/data_Multi_T/test.xyz"

SCRATCH_COMPILED_MODEL_PATH="/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE_models_l1_0802/mace_l1_0802_compiled.model"
FINETUNED_COMPILED_MODEL_PATH="/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_600K/finetuned_MACE_multihead0804_compiled.model"

# python compare_mace_representations.py \
# --scratch_model_path  $SCRATCH_COMPILED_MODEL_PATH \
# --finetuned_model_path $FINETUNED_COMPILED_MODEL_PATH \
# --probe_dataset_path $PROBE_DATASET_PATH

python mace_neb_1-calc.py
echo "mace_neb_1-calc.py completed"
python /data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/representation/1-2/mace_neb_2-plot.py
date