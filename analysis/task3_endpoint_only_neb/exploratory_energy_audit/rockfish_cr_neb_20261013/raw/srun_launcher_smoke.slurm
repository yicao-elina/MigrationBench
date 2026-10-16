#!/bin/bash -l
#SBATCH --job-name=neb_srun_smoke
#SBATCH --time=00:05:00
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=48
#SBATCH --mail-type=end
#SBATCH --mail-user=ycao73@jh.edu 
date
ml intel intel-mpi intel-mkl

ml qe/7.2

srun -n 48 hostname > srun_launch.out

date