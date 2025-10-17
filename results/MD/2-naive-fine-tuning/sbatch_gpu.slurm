#!/bin/bash
#SBATCH --job-name=nbr 
#SBATCH --time=6:00:0
#SBATCH -N  1   
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100
#SBATCH -A pclancy3_gpu 
#SBATCH --gres=gpu:1
source /data/apps/go.sh
source /data/apps/helpers/singularity.sh
export OMP_NUM_THREADS=4
date
module restore
source ~/.bashrc
 

ml openmpi/4.1.6
conda activate mace
ml
SCRIPT="mace_md_eval_0814.py"
echo "Running $SCRIPT"
python $SCRIPT 
echo "Running $SCRIPT done"
date
