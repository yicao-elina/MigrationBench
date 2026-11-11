#!/bin/bash -l
# Shared GPU/environment/data preflight for the 20261019 additive repairs.
# This job does not train or modify any model output.
#SBATCH --job-name=mb-repair-preflight-r01
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch16/pclancy3/yi/rockfish_training_repair_preflight_r01-%j.out
#SBATCH --error=/scratch16/pclancy3/yi/rockfish_training_repair_preflight_r01-%j.err

set -eo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export RF_CONDA_SITE="${RF_CONDA_SITE:-}"
export SINGULARITY_BIND="${SINGULARITY_BIND:-}"
source /etc/profile.d/modules.sh
set +u
source /data/apps/go.sh
source /data/apps/helpers/singularity.sh
module restore
module unload openmpi/3.1.6 >/dev/null 2>&1 || true
ml openmpi/4.1.6
source /data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh
conda activate mace
set -u

FT_DATA=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/data_600K
GROUPED_600K=/scratch16/pclancy3/yi/revision1_migrationbench_runs/ft600k_grouped_A2/data_grouped
GROUPED_MULTIT=/scratch16/pclancy3/yi/revision1_migrationbench_runs/multit_grouped_A2_20261001/data_grouped_multiT
FOUNDATION=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/MACE-matpes-pbe-omat-ft.model

for d in "$FT_DATA" "$GROUPED_600K" "$GROUPED_MULTIT"; do
  for f in train.xyz valid.xyz test.xyz; do
    test -s "$d/$f"
  done
done
test -s "$FOUNDATION"
command -v mace_run_train
python --version
python -c 'import torch, ase, mace; assert torch.cuda.is_available(); print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0)); print("ase", ase.__version__, "mace", getattr(mace, "__version__", "unknown"))'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
sha256sum "$FT_DATA"/train.xyz "$GROUPED_600K"/train.xyz "$GROUPED_MULTIT"/train.xyz "$FOUNDATION"
echo PREFLIGHT_OK
