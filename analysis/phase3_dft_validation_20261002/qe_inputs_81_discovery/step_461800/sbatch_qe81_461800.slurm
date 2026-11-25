#!/bin/bash
#SBATCH --job-name=mb_phase3_qe81_461800
#SBATCH --time=71:00:0
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --partition=azgpuh200
#SBATCH -A pclancy3_lab
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=/azwestus3/pclancy3_lab/ycao73/MigrationBench/phase3_dft/logs/%x-%j.out
#SBATCH --error=/azwestus3/pclancy3_lab/ycao73/MigrationBench/phase3_dft/logs/%x-%j.err
ml purge
date
hostname
source /azwestus3/pclancy3_lab/singularity.sh
ml openmpi/gcc/64/4.1.5
export OMP_NUM_THREADS=4
cd /azwestus3/pclancy3_lab/ycao73/MigrationBench/phase3_dft/step_461800
echo "=== pre-submit magnetism check ==="
grep -iE "nspin|magnetization|calculation" scf_461800.in
sha256sum scf_461800.in
singularity run --nv \
        /azwestus3/pclancy3_lab/quantum_espresso_qe-7.3.1.sif \
        mpirun -np 4 pw.x -inp 'scf_461800.in' > scf.out
grep -E "!    total energy|Forces acting on atoms|Total force|convergence has been achieved|JOB DONE|Error in routine|stopping" scf.out || true
