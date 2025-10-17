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

module reset

source /data/apps/go.sh

# bash commands go here. Don't forget to load the necessary modules!
ml intel
ml qe/7.2

ml
date
echo "This is sbatch-represent.slurm"
conda activate sklearn-env

SCRIPT="tsne.py"
echo "Running $SCRIPT"
# python mace_probe_blackbox.py
python $SCRIPT 
echo "Running $SCRIPT done"

date
