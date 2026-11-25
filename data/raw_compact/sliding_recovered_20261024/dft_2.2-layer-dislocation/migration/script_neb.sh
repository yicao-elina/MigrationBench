#!/bin/bash -l
#SBATCH --job-name=bulk
#SBATCH --time=48:0:0
#SBATCH --nodes=2
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=48
#SBATCH --mail-type=end
#SBATCH --mail-user=ycao73@jh.edu 
date
ml intel intel-mpi intel-mkl

ml qe/7.2

# detect the name of input file: "neb*.in"
input_file=$(ls neb*.in 2>/dev/null)
if [ -z "$input_file" ]; then
    echo "No input file found matching 'neb*.in'."
    exit 1
fi
output_file="${input_file%.in}.out"
# mpirun neb.x -inp neb.in > neb.out
echo "Running NEB calculation for: $input_file"
mpirun neb.x -inp $input_file > $output_file

date