#!/bin/bash
#SBATCH --job-name=mb_qeneb_1-4_ingap
#SBATCH --time=71:00:0
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --partition=azgpuh200
#SBATCH -A pclancy3_lab
#SBATCH --mem=400G
#SBATCH --gres=gpu:4
#SBATCH --output=/azwestus3/pclancy3_lab/ycao73/MigrationBench/qe_neb_1-4_in_gap/logs/%x-%j.out
#SBATCH --error=/azwestus3/pclancy3_lab/ycao73/MigrationBench/qe_neb_1-4_in_gap/logs/%x-%j.err

ml
date
echo "This is sbatch_qe_neb_pathway1-4.slurm for pathway 1-4 (in-gap) QE-NEB production"
echo "Running on $(hostname)"
echo "Working directory: $(pwd)"

source /azwestus3/pclancy3_lab/singularity.sh
ml openmpi/gcc/64/4.1.5

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NP=$SLURM_NTASKS
export MPI_OPTS="--bind-to none -np $NP"
export SIF="/azwestus3/pclancy3_lab/ycao73/qe-7.3.1.sif"

echo "=== pre-submit magnetism check (must be empty) ==="
grep -iE "nspin|magnetization" neb.in pw_*.in && { echo "MAGNETIC INPUT DETECTED - ABORTING"; exit 1; }
echo "clean (non-magnetic) confirmed"
sha256sum neb.in

echo "Starting QE NEB calculation (pathway 1-4, in-gap)..."
singularity exec --nv --bind $(pwd):/input $SIF mpirun $MPI_OPTS neb.x -inp /input/neb.in > neb.out
echo "QE NEB calculation completed."

date
echo "Job mb_qeneb_1-4_ingap finished."
