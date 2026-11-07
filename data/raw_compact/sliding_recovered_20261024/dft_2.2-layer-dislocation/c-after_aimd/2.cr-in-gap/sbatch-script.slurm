#!/bin/bash

#SBATCH --job-name=qe-neb
#SBATCH --time=71:00:0
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --partition=a100
#SBATCH -A pclancy3_gpu
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

# Load the necessary modules
ml
date
echo "This is sbatch-script.slurm"
## Azure HPC environment setup
# source /azwestus3/pclancy3_lab/singularity.sh
# ml openmpi/gcc/64/4.1.5

# Rocky Linux environment setup
source /data/apps/go.sh
source /data/apps/helpers/singularity.sh
ml openmpi/4.1.6

export OMP_NUM_THREADS=2

singularity exec --nv \
    --bind $(pwd):/input \
    /home/ycao73/data-pclancy3/yi/qe-7.3.1.sif \
    mpirun -np 2 neb.x -in /input/neb.in > neb.out


# singularity run --nv \
#     /azwestus3/pclancy3_lab/quantum_espresso_qe-7.3.1.sif \
#     mpirun -np 4 neb.x -inp neb.in > neb.out

echo "NEB calculation completed."
date
