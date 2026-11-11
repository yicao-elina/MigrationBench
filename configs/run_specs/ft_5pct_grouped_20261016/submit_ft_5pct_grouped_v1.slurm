#!/bin/bash -l
# Strict 5% FT matrix: six jobs, three seeds for each grouped data family.
#SBATCH --job-name=mb-ft-5pct-grouped
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/scratch16/pclancy3/yi/ft_5pct_grouped_v1/logs/%x-%A_%a.out
#SBATCH --error=/scratch16/pclancy3/yi/ft_5pct_grouped_v1/logs/%x-%A_%a.err

set -euo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export RF_CONDA_SITE="${RF_CONDA_SITE:-}"
export SINGULARITY_BIND="${SINGULARITY_BIND:-}"
source /data/apps/go.sh
source /data/apps/helpers/singularity.sh
module restore
module unload openmpi/3.1.6 >/dev/null 2>&1 || true
ml openmpi/4.1.6
source /data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh
conda activate mace
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

BASE=/scratch16/pclancy3/yi/ft_5pct_grouped_v1
FOUNDATION=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/MACE-matpes-pbe-omat-ft.model
SEEDS=(123 234 345)
case "${SLURM_ARRAY_TASK_ID}" in
  0) FAMILY=ft_600k_5pct_grouped_v1; SEED=123; DATA_DIR="$BASE/data/ft_600k_5pct_grouped_v1";;
  1) FAMILY=ft_600k_5pct_grouped_v1; SEED=234; DATA_DIR="$BASE/data/ft_600k_5pct_grouped_v1";;
  2) FAMILY=ft_600k_5pct_grouped_v1; SEED=345; DATA_DIR="$BASE/data/ft_600k_5pct_grouped_v1";;
  3) FAMILY=ft_multit_5pct_grouped_v1; SEED=123; DATA_DIR="$BASE/data/ft_multit_5pct_grouped_v1";;
  4) FAMILY=ft_multit_5pct_grouped_v1; SEED=234; DATA_DIR="$BASE/data/ft_multit_5pct_grouped_v1";;
  5) FAMILY=ft_multit_5pct_grouped_v1; SEED=345; DATA_DIR="$BASE/data/ft_multit_5pct_grouped_v1";;
  *) echo "invalid array index" >&2; exit 2;;
esac

MODEL_NAME="${FAMILY}_seed${SEED}"
MODEL_DIR="$BASE/models/$MODEL_NAME"
for f in train.xyz valid.xyz test.xyz dataset_manifest.json; do test -s "$DATA_DIR/$f" || { echo "missing $DATA_DIR/$f" >&2; exit 3; }; done
test -s "$FOUNDATION" || { echo "missing foundation $FOUNDATION" >&2; exit 4; }
mkdir -p "$MODEL_DIR"
if test -e "$MODEL_DIR/${MODEL_NAME}_stagetwo_compiled.model"; then
  echo "refusing overwrite of existing compiled model" >&2
  exit 8
fi
{
  echo "campaign_id=ft_5pct_grouped_20261016"
  echo "family=$FAMILY seed=$SEED job=${SLURM_JOB_ID:-unknown} array=${SLURM_ARRAY_TASK_ID:-unknown}"
  command -v mace_run_train
  python -c 'import torch, ase, mace; print("torch", torch.__version__); print("ase", ase.__version__); print("mace", getattr(mace, "__version__", "unknown"))'
  sha256sum "$DATA_DIR"/train.xyz "$DATA_DIR"/valid.xyz "$DATA_DIR"/test.xyz "$DATA_DIR"/dataset_manifest.json "$FOUNDATION"
} | tee "$MODEL_DIR/preflight.txt"

mace_run_train \
  --name="$MODEL_NAME" --model_dir="$MODEL_DIR" --log_dir="$MODEL_DIR" \
  --checkpoints_dir="$MODEL_DIR" --results_dir="$MODEL_DIR" \
  --train_file="$DATA_DIR/train.xyz" --valid_file="$DATA_DIR/valid.xyz" --test_file="$DATA_DIR/test.xyz" \
  --model=MACE --num_channels=32 --max_L=1 --r_max=6.0 \
  --hidden_irreps='128x0e + 128x1o' --batch_size=32 --valid_batch_size=16 \
  --max_num_epochs=1000 --swa --start_swa=200 --ema --ema_decay=0.99 \
  --device=cuda --energy_key=energy --forces_key=forces \
  --seed="$SEED" --eval_interval=50 --patience=20 \
  --forces_weight=10.0 --energy_weight=1.0 --stress_weight=0.0 \
  --E0s=isolated --foundation_model="$FOUNDATION" --multiheads_finetuning=True

COMPILED="$MODEL_DIR/${MODEL_NAME}_stagetwo_compiled.model"
test -s "$COMPILED" || { echo "compiled model missing: $COMPILED" >&2; exit 9; }
sha256sum "$COMPILED" | tee "$MODEL_DIR/model_hash.txt"
echo "training_complete=$MODEL_NAME"
