#!/bin/bash -l
# One-GPU compiled-model acceptance gate for registered and completed models.
# This gate is read-only with respect to source models and refuses no inputs.
#SBATCH --job-name=mb-canonical-model-gate
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch16/pclancy3/yi/canonical_model_matrix_20261016/model_gate_audit/logs/%x-%j.out
#SBATCH --error=/scratch16/pclancy3/yi/canonical_model_matrix_20261016/model_gate_audit/logs/%x-%j.err

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

ROOT=/scratch16/pclancy3/yi/canonical_model_matrix_20261016/model_gate_audit
mkdir -p "$ROOT/logs"
python "$ROOT/model_gate_audit.py" \
  --registry "$ROOT/MODEL_REGISTRY_CANONICAL_PRELIM.tsv" \
  --output "$ROOT" \
  --device cuda
