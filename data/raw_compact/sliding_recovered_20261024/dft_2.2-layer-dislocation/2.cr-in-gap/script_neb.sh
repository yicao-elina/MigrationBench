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

mpirun neb.x -inp neb.in > neb.out

date