#!/bin/bash -l
#SBATCH --account=pclancy3_gpu
#SBATCH --job-name=multit-grouped-A2
#SBATCH --output=slurm.%x.%A_%a.out
#SBATCH --error=slurm.%x.%A_%a.err
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --array=0-2

# Array mapping: Multi-T OMAT-FT grouped seeds 123/234/345.
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export RF_CONDA_SITE="${RF_CONDA_SITE:-}"
source /data/apps/go.sh
source /data/apps/helpers/singularity.sh
ml openmpi/4.1.6
module restore
source /data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh
conda activate mace
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

BASE=/scratch16/pclancy3/yi/revision1_migrationbench_runs/multit_grouped_A2_20261001
DATA_ROOT="$BASE/data_grouped_multiT"
OUT_ROOT="$BASE/retrain_multit_scratch16_20261001"
mkdir -p "$OUT_ROOT"

case "${SLURM_ARRAY_TASK_ID}" in
  0) FAMILY=multit_omatft; SEED=123; DATA_DIR="$DATA_ROOT"; FOUNDATION=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/MACE-matpes-pbe-omat-ft.model;;
  1) FAMILY=multit_omatft; SEED=234; DATA_DIR="$DATA_ROOT"; FOUNDATION=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/MACE-matpes-pbe-omat-ft.model;;
  2) FAMILY=multit_omatft; SEED=345; DATA_DIR="$DATA_ROOT"; FOUNDATION=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/MACE-matpes-pbe-omat-ft.model;;
  *) echo "invalid array index" >&2; exit 2;;
esac

for f in train.xyz valid.xyz test.xyz; do test -s "$DATA_DIR/$f" || { echo "missing dataset: $DATA_DIR/$f" >&2; exit 3; }; done
if [[ "$FAMILY" == scratch5 ]]; then
  test -s "$SCRATCH5_ROOT/scratch5_manifest.json" || { echo "refusing Scratch-5% run: missing scratch5_manifest.json" >&2; exit 4; }
fi

NAME="${FAMILY}_grouped_seed${SEED}"
MODEL_DIR="$OUT_ROOT/$NAME"
mkdir -p "$MODEL_DIR"
{
  echo "host=$(hostname) job=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID} family=$FAMILY seed=$SEED"
  command -v mace_run_train
  python -c 'import torch, ase; print("torch", torch.__version__, "ase", ase.__version__)'
  sha256sum "$DATA_DIR"/train.xyz "$DATA_DIR"/valid.xyz "$DATA_DIR"/test.xyz
  [[ -z "$FOUNDATION" ]] || sha256sum "$FOUNDATION"
} | tee "$MODEL_DIR/preflight.txt"

ARGS=(
  --name="$NAME" --model_dir="$MODEL_DIR" --log_dir="$MODEL_DIR"
  --checkpoints_dir="$MODEL_DIR" --results_dir="$MODEL_DIR"
  --train_file="$DATA_DIR/train.xyz" --valid_file="$DATA_DIR/valid.xyz" --test_file="$DATA_DIR/test.xyz"
  --model=MACE --num_channels=32 --max_L=1 --r_max=6.0
  --hidden_irreps='128x0e + 128x1o' --batch_size=32 --valid_batch_size=16
  --max_num_epochs=1000 --swa --start_swa=200 --ema --ema_decay=0.99
  --device=cuda --energy_key=energy --forces_key=forces
  --seed="$SEED" --eval_interval=50 --patience=20
  --forces_weight=10.0 --energy_weight=1.0 --stress_weight=0.0
)
ARGS+=(--E0s=isolated)
if [[ -n "$FOUNDATION" ]]; then ARGS+=(--foundation_model="$FOUNDATION" --multiheads_finetuning=True); fi
mace_run_train "${ARGS[@]}"
