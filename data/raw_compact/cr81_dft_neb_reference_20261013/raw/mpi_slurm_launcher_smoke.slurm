#!/bin/bash -l
#SBATCH --job-name=neb_mpi_slurm
#SBATCH --time=00:05:00
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=48
#SBATCH --mail-type=end
#SBATCH --mail-user=ycao73@jh.edu 
date
ml intel intel-mpi intel-mkl

ml qe/7.2
export PATH=/cm/shared/apps/slurm/current/bin:$PATH

mpirun -launcher slurm -n 48 hostname > mpi_launch.out

date