#!/bin/bash -l
# New isolated broad-roster fine-tuning. Original models/campaigns are read-only.
#SBATCH --account=pclancy3_gpu
#SBATCH --job-name=mb-broad-roster-ft
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --output=/scratch16/pclancy3/yi/cross_element_generalization_repair_20261002/logs/%x-%A.out
#SBATCH --error=/scratch16/pclancy3/yi/cross_element_generalization_repair_20261002/logs/%x-%A.err

set -euo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export RF_CONDA_SITE="${RF_CONDA_SITE:-}"
export SINGULARITY_BIND="${SINGULARITY_BIND:-}"
source /data/apps/go.sh
source /data/apps/helpers/singularity.sh
module restore
source /data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh
conda activate mace
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

RUN_ROOT="${RUN_ROOT:-/scratch16/pclancy3/yi/cross_element_generalization_repair_20261002}"
DATA_ROOT="${DATA_ROOT:-/scratch16/pclancy3/yi/revision1_migrationbench_runs/multit_grouped_A2_20261001/data_grouped_multiT}"
FOUNDATION="${FOUNDATION:-/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/2.aimd/3-2.2-layer_CrSb2Te3/../..}"
FOUNDATION="${FOUNDATION_PATH:-/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/2.aimd/3-2.2-layer_CrSb2Te3/../../3.fine-tuning/2-layer/MACE-omat/MACE-matpes-pbe-omat-ft.model}"
NAME="${NAME:-broad_roster_ft_seed123}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
MODEL_DIR="$RUN_ROOT/$NAME"
mkdir -p "$RUN_ROOT/logs" "$MODEL_DIR"
cd "$RUN_ROOT"

for f in train.xyz valid.xyz test.xyz; do test -s "$DATA_ROOT/$f" || { echo "missing $DATA_ROOT/$f" >&2; exit 3; }; done
test -s "$FOUNDATION" || { echo "missing Foundation $FOUNDATION" >&2; exit 4; }
{
  echo "host=$(hostname) job=${SLURM_JOB_ID} name=$NAME max_epochs=$MAX_EPOCHS"
  sha256sum "$DATA_ROOT"/train.xyz "$DATA_ROOT"/valid.xyz "$DATA_ROOT"/test.xyz "$FOUNDATION"
  echo 'target_atomic_numbers=[3,11,23,24,25,29,35,42,51,52,55,82]'
} | tee "$MODEL_DIR/preflight.txt"

mace_run_train \
  --name="$NAME" --model_dir="$MODEL_DIR" --log_dir="$MODEL_DIR" \
  --checkpoints_dir="$MODEL_DIR" --results_dir="$MODEL_DIR" \
  --train_file="$DATA_ROOT/train.xyz" --valid_file="$DATA_ROOT/valid.xyz" --test_file="$DATA_ROOT/test.xyz" \
  --model=MACE --num_channels=32 --max_L=1 --r_max=6.0 \
  --hidden_irreps='128x0e + 128x1o' --batch_size=32 --valid_batch_size=16 \
  --max_num_epochs="$MAX_EPOCHS" --ema --ema_decay=0.99 \
  --device=cuda --energy_key=energy --forces_key=forces \
  --seed=123 --eval_interval=50 --patience=20 \
  --forces_weight=10.0 --energy_weight=1.0 --stress_weight=0.0 \
  --E0s=foundation \
  --atomic_numbers='[3,11,23,24,25,29,35,42,51,52,55,82]' \
  --foundation_model="$FOUNDATION" --foundation_model_elements=True --multiheads_finetuning=True
