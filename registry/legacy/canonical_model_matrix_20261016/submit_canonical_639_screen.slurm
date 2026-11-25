#!/bin/bash -l
# Canonical accepted-model static screen.  This job must remain held until
# SCREEN_INPUT_MANIFEST.json says READY_FOR_SUBMISSION and all model gates pass.
#SBATCH --job-name=mb-canonical-639
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-63%4
#SBATCH --output=/scratch16/pclancy3/yi/canonical_model_matrix_20261016/639_screen_v1/logs/%x-%A_%a.out
#SBATCH --error=/scratch16/pclancy3/yi/canonical_model_matrix_20261016/639_screen_v1/logs/%x-%A_%a.err

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
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

ROOT=/scratch16/pclancy3/yi/canonical_model_matrix_20261016/639_screen_v1
mkdir -p "$ROOT/logs" "$ROOT/results_production"
python "$ROOT/run_mlip_candidate_chunk.py" \
  --manifest "$ROOT/candidate_manifest_639.csv" \
  --models "$ROOT/models_rockfish.tsv" \
  --chunk-index "${SLURM_ARRAY_TASK_ID}" --chunk-size 10 \
  --out "$ROOT/results_production" --device cuda
