#!/bin/bash -l
#SBATCH --account=pclancy3
#SBATCH --export=ALL

#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --job-name="mace_md_pp"
#SBATCH --output="slurm.%j"

#SBATCH --partition=shared
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24

date

source /data/apps/go.sh

source ~/.bashrc
conda activate mace
ml

# # this script is for post-processing the MACE MD results 
# # after running the mace_md_eval0814.py script,
# # which generates the csv files for the MACE MD results.
# SCRIPT="mace_md_pp.py"
# echo "Running $SCRIPT"
# python $SCRIPT 
# echo "Running $SCRIPT done"

# this script only need the xyz file
# SCRIPT2="mace_md_pp_xyz.py"
SCRIPT2="/data/pclancy3/yi/cyqe/mlff/mace_md_pp_xyz.py"
echo "Running $SCRIPT2"
python $SCRIPT2
echo "Running $SCRIPT2 done"
date