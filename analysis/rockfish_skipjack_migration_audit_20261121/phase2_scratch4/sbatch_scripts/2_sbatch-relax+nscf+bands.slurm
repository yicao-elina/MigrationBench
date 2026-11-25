#!/bin/bash -l
#SBATCH --account=pclancy3
#SBATCH --export=ALL

#SBATCH --mail-type=begin
#SBATCH --mail-type=end

#SBATCH --job-name="WSe2-2d"
#SBATCH --output="slurm.%j"

#SBATCH --partition=parallel
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48

module reset

source /data/apps/go.sh

# bash commands go here. Don't forget to load the necessary modules!
ml intel
ml qe/7.2

ml
date
echo "This is sbatch-relax+nscf+bands.slurm"

mpirun /data/apps/extern/qe/7.2/bin/pw.x < WSe2_supercell_z2_vacuum.in > relax.out
# mpirun /data/apps/extern/qe/7.2/bin/pw.x < '2-nscf.in' > '2-nscf.out'
# mpirun /data/apps/extern/qe/7.2/bin/pw.x < '3-bands.in' > '3-bands.out'
# mpirun /data/apps/extern/qe/7.2/bin/pw.x < aimd_s2t3_221.in > aimd_s2t3_221.out
# ml restore lammps-flare
# conda activate flare-b
# sh /scratch16/pclancy3/yi/FLARE/sb2te3-cr2te4/01_py/1.vc-relax_pp.sh

# mpirun pp.x < "/scratch16/pclancy3/yi/FLARE/sb2te3-cr2te4/01_example-file/charge_density.in" > charge_density.out
# sbatch sbatch-all-post.slurm

date
