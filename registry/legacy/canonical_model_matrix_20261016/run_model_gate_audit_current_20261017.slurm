#!/bin/bash -l
# Versioned compiled-model gate for the current canonical registry.
# The previous gate output remains preserved under model_gate_audit/.
#SBATCH --job-name=mb-canonical-gate-current
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch16/pclancy3/yi/canonical_model_matrix_20261016/model_gate_audit_32114532/logs/%x-%j.out
#SBATCH --error=/scratch16/pclancy3/yi/canonical_model_matrix_20261016/model_gate_audit_32114532/logs/%x-%j.err

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

ROOT=/scratch16/pclancy3/yi/canonical_model_matrix_20261016/model_gate_audit_32114532
mkdir -p "$ROOT/logs"
python "$ROOT/model_gate_audit.py" \
  --registry "$ROOT/MODEL_REGISTRY_CANONICAL_CURRENT_20261017.tsv" \
  --output "$ROOT" \
  --device cuda
