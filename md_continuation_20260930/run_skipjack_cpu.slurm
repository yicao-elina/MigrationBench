#!/bin/bash -l
#SBATCH --account=pclancy3_dualx
#SBATCH --partition=med
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --signal=USR1@120
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
module purge
module load gcc/9.3.0 pytorch/2.10.0-b200-py3.11
export PKG=/home/jhu/ycao73/scratch_pclancy3/yi/mace_four_model_MD_continuation_20260930/python_pkgs
export PYTHONPATH=$PKG:/home/jhu/ycao73/scratch_pclancy3/yi/mace_four_model_MD_continuation_20260930/models
export LD_LIBRARY_PATH=$PKG/scipy.libs:$PKG/numpy.libs:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd "$RUN_DIR"
exec python /home/jhu/ycao73/scratch_pclancy3/yi/mace_four_model_MD_continuation_20260930/continue_md.py \
  --input "$INPUT" --model "$MODEL" --run-dir "$RUN_DIR" --start-step "$START_STEP" --device cpu
