#!/bin/bash -l
#SBATCH --account=pclancy3
#SBATCH --export=ALL

#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --job-name="mace-train"
#SBATCH --output="slurm.%j"

#SBATCH --partition=shared
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
module reset

source /data/apps/go.sh
module restore
source ~/.bashrc


# bash commands go here. Don't forget to load the necessary modules!
ml
date
echo "This is sbatch-represent.slurm"
conda activate sklearn-env

# SCRIPT="mace_probe_blackbox0818.py"
SCRIPT1="1-shap-data.py"
SCRIPT2="2-shap-plot.py"
echo "Running $SCRIPT1"
# python mace_probe_blackbox.py
python $SCRIPT1
echo "Running $SCRIPT1 done"
echo "Running $SCRIPT2"
python $SCRIPT2
echo "Running $SCRIPT2 done"

date
