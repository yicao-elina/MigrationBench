#!/bin/bash
#SBATCH --job-name=mb_ep_relax_unique_057
#SBATCH --time=71:00:0
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --partition=azgpuh200
#SBATCH -A pclancy3_lab
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=/azwestus3/pclancy3_lab/ycao73/MigrationBench/endpoint_relax_47/logs/%x-%j.out
#SBATCH --error=/azwestus3/pclancy3_lab/ycao73/MigrationBench/endpoint_relax_47/logs/%x-%j.err
ml
date
hostname
source /azwestus3/pclancy3_lab/singularity.sh
ml openmpi/gcc/64/4.1.5
export OMP_NUM_THREADS=4
cd /azwestus3/pclancy3_lab/ycao73/MigrationBench/endpoint_relax_47/unique_057
echo "=== pre-submit magnetism check ==="
grep -iE "nspin|magnetization" relax_unique_057.in && { echo "MAGNETIC INPUT DETECTED - ABORTING"; exit 1; }
echo "calculation line:"; grep "calculation" relax_unique_057.in
sha256sum relax_unique_057.in
singularity run --nv \
        /azwestus3/pclancy3_lab/quantum_espresso_qe-7.3.1.sif \
        mpirun -np 4 pw.x -inp 'relax_unique_057.in' > relax.out
tail -5 relax.out
