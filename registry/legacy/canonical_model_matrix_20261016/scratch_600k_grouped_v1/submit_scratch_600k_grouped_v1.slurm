#!/bin/bash
# Canonical strict from-scratch scratch-600K matrix: only TRAINING_SEED varies.
# Dataset and protocol are frozen by EXPERIMENT_MANIFEST.json.
#SBATCH --job-name=mb-canonical-scratch-600k
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/scratch16/pclancy3/yi/canonical_model_matrix_20261016/scratch_600k_grouped_v1/logs/%x-%A_%a.out
#SBATCH --error=/scratch16/pclancy3/yi/canonical_model_matrix_20261016/scratch_600k_grouped_v1/logs/%x-%A_%a.err

set -eo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export RF_CONDA_SITE="${RF_CONDA_SITE:-}"
source /etc/profile.d/modules.sh
source /data/apps/go.sh
source /data/apps/helpers/singularity.sh
module restore
module unload openmpi/3.1.6 >/dev/null 2>&1 || true
ml openmpi/4.1.6
export OMP_NUM_THREADS=4
date
source /data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh
conda activate mace
module list
set -u

DATA_ROOT=/scratch16/pclancy3/yi/revision1_migrationbench_runs/ft600k_grouped_A2/data_grouped
UPSTREAM_MANIFEST=/scratch16/pclancy3/yi/revision1_migrationbench_runs/ft600k_grouped_A2/manifest_proposed_A2.json
OUT_ROOT=/scratch16/pclancy3/yi/canonical_model_matrix_20261016/scratch_600k_grouped_v1
SEEDS=(123 234 345)
TRAINING_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="scratch_600k_grouped_v1_seed_${TRAINING_SEED}"
MODEL_DIR="$OUT_ROOT/models/$MODEL_NAME"

for f in train.xyz valid.xyz test.xyz; do
  test -s "$DATA_ROOT/$f" || { echo "missing dataset: $DATA_ROOT/$f" >&2; exit 3; }
done
test -s "$UPSTREAM_MANIFEST" || { echo "missing upstream manifest" >&2; exit 4; }
mkdir -p "$MODEL_DIR"
COMPILED="$MODEL_DIR/${MODEL_NAME}_stagetwo_compiled.model"
if test -e "$COMPILED"; then
  echo "refusing overwrite of existing compiled model: $COMPILED" >&2
  exit 8
fi

{
  echo "campaign_id=scratch_600k_grouped_v1"
  echo "model_name=$MODEL_NAME"
  echo "training_seed=$TRAINING_SEED"
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "array_task_id=${SLURM_ARRAY_TASK_ID:-unknown}"
  echo "host=$(hostname)"
  echo "python=$(command -v python)"
  python --version
  python -c 'import torch, ase, mace; print("torch", torch.__version__); print("ase", ase.__version__); print("mace", getattr(mace, "__version__", "unknown"))'
  sha256sum "$DATA_ROOT"/train.xyz "$DATA_ROOT"/valid.xyz "$DATA_ROOT"/test.xyz "$UPSTREAM_MANIFEST"
  echo "foundation_model=null"
  echo "elements=Cr,Sb,Te"
} | tee "$MODEL_DIR/preflight.txt"

cd /data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer
mace_run_train \
    --name="$MODEL_NAME" \
    --model_dir="$MODEL_DIR" \
    --log_dir="$MODEL_DIR" \
    --checkpoints_dir="$MODEL_DIR" \
    --results_dir="$MODEL_DIR" \
    --train_file="$DATA_ROOT/train.xyz" \
    --valid_file="$DATA_ROOT/valid.xyz" \
    --test_file="$DATA_ROOT/test.xyz" \
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

test -s "$COMPILED" || { echo "compiled model missing: $COMPILED" >&2; exit 9; }
sha256sum "$COMPILED" | tee "$MODEL_DIR/model_hash.txt"
echo "training_complete=$MODEL_NAME"
date
