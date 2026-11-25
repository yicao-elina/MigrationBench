#!/bin/bash -l
# Additive repair of the historical FT-600K random-seed reproduction.
# The failed 20261016 wrapper stopped before training because its Conda path
# did not exist. This wrapper uses a verified Rockfish Conda installation and
# writes only into the 20261019 repair root.
#SBATCH --job-name=mb-ft600k-random-r01
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch16/pclancy3/yi/ft_600k_random_reproduction_20261019_repair_r01/logs/%x-%A_%a.out
#SBATCH --error=/scratch16/pclancy3/yi/ft_600k_random_reproduction_20261019_repair_r01/logs/%x-%A_%a.err

set -eo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export RF_CONDA_SITE="${RF_CONDA_SITE:-}"
export SINGULARITY_BIND="${SINGULARITY_BIND:-}"
source /etc/profile.d/modules.sh

# Rockfish bootstrap helpers may reference optional unset variables.
set +u
source /data/apps/go.sh
source /data/apps/helpers/singularity.sh
module restore
module unload openmpi/3.1.6 >/dev/null 2>&1 || true
ml openmpi/4.1.6
source /data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh
conda activate mace
set -u

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

DATA_ROOT=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/data_600K
FOUNDATION=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/MACE-matpes-pbe-omat-ft.model
OUT_ROOT=/scratch16/pclancy3/yi/ft_600k_random_reproduction_20261019_repair_r01
SEEDS=(234 345)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="ft_600k_random_reproduction_repair_r01_seed${SEED}"
MODEL_DIR="$OUT_ROOT/models/$MODEL_NAME"
COMPILED="$MODEL_DIR/${MODEL_NAME}_stagetwo_compiled.model"

for f in train.xyz valid.xyz test.xyz; do
  test -s "$DATA_ROOT/$f"
done
test -s "$FOUNDATION"
mkdir -p "$MODEL_DIR"
if test -e "$COMPILED"; then
  echo "refusing overwrite: $COMPILED" >&2
  exit 8
fi

{
  echo "campaign_id=ft_600k_random_reproduction_20261019_repair_r01"
  echo "model_family=ft_600k_random_reproduction_repair_r01"
  echo "model_name=$MODEL_NAME"
  echo "training_seed=$SEED"
  echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
  echo "array_task_id=${SLURM_ARRAY_TASK_ID:-unknown}"
  echo "host=$(hostname)"
  command -v mace_run_train
  python --version
  python -c 'import torch, ase, mace; print("torch", torch.__version__); print("ase", ase.__version__); print("mace", getattr(mace, "__version__", "unknown"))'
  sha256sum "$DATA_ROOT"/train.xyz "$DATA_ROOT"/valid.xyz "$DATA_ROOT"/test.xyz "$FOUNDATION"
  echo "dataset_split=historical_data_600K_random_split"
  echo "anchor_seed=123"
} | tee "$MODEL_DIR/preflight.txt"

cd /data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer
mace_run_train \
  --name="$MODEL_NAME" --model_dir="$MODEL_DIR" --log_dir="$MODEL_DIR" \
  --checkpoints_dir="$MODEL_DIR" --results_dir="$MODEL_DIR" \
  --train_file="$DATA_ROOT/train.xyz" --valid_file="$DATA_ROOT/valid.xyz" --test_file="$DATA_ROOT/test.xyz" \
  --model=MACE --num_channels=32 --max_L=1 --correlation=3 --r_max=6.0 \
  --hidden_irreps='128x0e + 128x1o' --batch_size=32 --valid_batch_size=16 \
  --max_num_epochs=1000 --swa --start_swa=200 --ema --ema_decay=0.99 \
  --device=cuda --energy_key=energy --forces_key=forces --E0s=isolated \
  --seed="$SEED" --eval_interval=50 --patience=20 \
  --forces_weight=10.0 --energy_weight=1.0 --stress_weight=0.0 \
  --foundation_model="$FOUNDATION" --multiheads_finetuning=True

test -s "$COMPILED"
sha256sum "$COMPILED" | tee "$MODEL_DIR/model_hash.txt"
echo "training_complete=$MODEL_NAME"
date
