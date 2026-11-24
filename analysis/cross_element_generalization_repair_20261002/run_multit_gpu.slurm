#!/bin/bash -l
# Independent FT-MultiT GPU/load/element-support smoke. Does not alter any model.
#SBATCH --account=pclancy3_gpu
#SBATCH --job-name=generalization-multit-smoke
#SBATCH --partition=a100
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/scratch16/pclancy3/yi/cross_element_generalization_repair_20261002/logs/%x-%j.out
#SBATCH --error=/scratch16/pclancy3/yi/cross_element_generalization_repair_20261002/logs/%x-%j.err

set -euo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export RF_CONDA_SITE="${RF_CONDA_SITE:-}"
export SINGULARITY_BIND="${SINGULARITY_BIND:-}"
source /data/apps/go.sh
source /data/apps/helpers/singularity.sh
module restore
source /data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh
conda activate mace
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

RUN_ROOT="${RUN_ROOT:-/scratch16/pclancy3/yi/cross_element_generalization_repair_20261002}"
MODEL="${MODEL:-/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/X-FORCE/xforce/models/FT_Multi_T.model}"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/results"
cd "$RUN_ROOT"
python "$RUN_ROOT/gpu_multit_smoke.py" --model "$MODEL" --device cuda --out "$RUN_ROOT/results/ft_multit_gpu_smoke.json"
