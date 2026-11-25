#!/bin/bash -l
#SBATCH --job-name=mace_smoke_sj
#SBATCH --account=pclancy3_dualx
#SBATCH --partition=med
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --output=smoke_skipjack_%j.out
#SBATCH --error=smoke_skipjack_%j.err
set -euo pipefail
module purge
module load gcc/9.3.0
module load pytorch/2.10.0-b200-py3.11
export PYTHONPATH=/home/jhu/ycao73/scratch_pclancy3/yi/mace_four_model_MD_continuation_20260930/python_pkgs:/home/jhu/ycao73/scratch_pclancy3/yi/mace_four_model_MD_continuation_20260930/models
export LD_LIBRARY_PATH=/home/jhu/ycao73/scratch_pclancy3/yi/mace_four_model_MD_continuation_20260930/python_pkgs/scipy.libs:/home/jhu/ycao73/scratch_pclancy3/yi/mace_four_model_MD_continuation_20260930/python_pkgs/numpy.libs:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python - <<'PY'
import os, sys
print('PYTHON', sys.version, sys.path[:5])
import numpy
print('NUMPY', numpy.__version__, numpy.__file__)
import torch
from ase.io import read
from mace.calculators import MACECalculator
traj = '/home/jhu/ycao73/scratch_pclancy3/yi/mace_four_model_MD_transfer_20260929/extracted/MD/0-foundation-omat/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz'
model = '/home/jhu/ycao73/scratch_pclancy3/yi/mace_four_model_MD_continuation_20260930/models/MACE-omat/finetuned_MACE_compiled.model'
print('HOST', os.uname().nodename)
print('PYTORCH', torch.__version__, 'CUDA', torch.cuda.is_available())
atoms = read(traj, index=-1)
atoms.calc = MACECalculator(model_paths=model, device='cpu', default_dtype='float64')
print('NATOMS', len(atoms), 'ENERGY', atoms.get_potential_energy(), 'MAXF', abs(atoms.get_forces()).max())
PY
