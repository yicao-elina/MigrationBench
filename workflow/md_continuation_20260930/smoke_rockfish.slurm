#!/bin/bash -l
#SBATCH --job-name=mace_smoke_rf
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=smoke_rockfish_%j.out
#SBATCH --error=smoke_rockfish_%j.err
set -euo pipefail
export MACE_ENV=/data/pclancy3/yi/.conda/envs/mace
export PATH=$MACE_ENV/bin:$PATH
export PYTHONPATH=/data/pclancy3/yi/.python_pkgs:/data/pclancy3/yi/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
$MACE_ENV/bin/python - <<'PY'
import os, torch
from ase.io import read
from mace.calculators import MACECalculator
traj = '/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/1-from-scratch/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz'
model = '/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE_models_l1_0802/mace_l1_0802_compiled.model'
print('HOST', os.uname().nodename)
print('PYTORCH', torch.__version__, 'CUDA', torch.cuda.is_available(), 'N', torch.cuda.device_count())
atoms = read(traj, index=-1)
atoms.calc = MACECalculator(model_paths=model, device='cuda', default_dtype='float64')
print('NATOMS', len(atoms), 'ENERGY', atoms.get_potential_energy(), 'MAXF', abs(atoms.get_forces()).max())
PY
