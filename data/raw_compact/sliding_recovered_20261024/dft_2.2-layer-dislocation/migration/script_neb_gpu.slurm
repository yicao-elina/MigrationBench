#!/bin/bash
#SBATCH --job-name=nbr 
#SBATCH --time=71:00:0
#SBATCH -N  1   
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --partition=a100
#SBATCH -A pclancy3_gpu
#SBATCH --mem=0         
#SBATCH --gres=gpu:2
 
source /data/apps/go.sh
source /data/apps/helpers/singularity.sh
 
 
ml openmpi/4.1.6
export OMP_NUM_THREADS=4

# detect the name of input file: "neb*.in"
input_file=$(ls neb*.in 2>/dev/null)
if [ -z "$input_file" ]; then
    echo "No input file found matching 'neb*.in'."
    exit 1
fi
output_file="${input_file%.in}.out"
# mpirun neb.x -inp neb.in > neb.out
# singularity run --nv \
# /scratch16/jcombar1/QE731/quantum_espresso_qe-7.3.1.sif \
# mpirun -np 4 neb.x -inp $input_file > $output_file
ml intel intel-mpi intel-mkl
ml qe/7.2
mpirun neb.x -inp $input_file > $output_file
date