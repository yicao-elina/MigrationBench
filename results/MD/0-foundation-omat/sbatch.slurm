#!/bin/bash -l
#SBATCH --account=pclancy3
#SBATCH --export=ALL

#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --job-name="relax_2L_octo_Cr2_v2"
#SBATCH --output="slurm.%j"

#SBATCH --partition=shared
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24

date

source /data/apps/go.sh
source /data/apps/helpers/singularity.sh

module restore
source ~/.bashrc
 
ml openmpi/4.1.6
conda activate mace
ml
SCRIPT="mace_md_eval.py"
echo "Running $SCRIPT"
python $SCRIPT 
echo "Running $SCRIPT done"
date