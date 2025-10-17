#!/bin/bash -l
#SBATCH --account=pclancy3
#SBATCH --export=ALL

#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --job-name="pbs"
#SBATCH --output="slurm.%j"

#SBATCH --partition=shared
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24

# module reset

source /data/apps/go.sh
source ~/.bashrc

ml
date
echo "This is sbatch-represent.slurm"
conda activate sklearn-env

# Verify we're using the correct Python
echo "Current Python version: $(python --version)"
echo "Current Python path: $(which python)"
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Use the exact same Python path as your terminal
PYTHON_PATH="/data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/bin/python"
echo "Target Python path: $PYTHON_PATH"
echo "Target Python version: $($PYTHON_PATH --version)"

date
echo "This is sbatch-represent.slurm"

SCRIPT="tsne.py"
echo "Running $SCRIPT with target Python"
$PYTHON_PATH $SCRIPT  # Use $PYTHON_PATH instead of python
echo "Running $SCRIPT done"
date
