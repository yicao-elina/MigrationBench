#!/bin/bash
# Minimal Discovery smoke: one directory, one input, one sbatch submission.
# Scientific validation is performed after the job; this script only launches QE.
#SBATCH --job-name=mb_simple_smoke
#SBATCH --time=71:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --partition=azgpuh200
#SBATCH -A pclancy3_lab
#SBATCH --mem=96G
#SBATCH --gres=gpu:1

cd "${SLURM_SUBMIT_DIR}"
ml
date
export OMP_NUM_THREADS=4

/usr/bin/singularity exec --nv --bind /azwestus3:/azwestus3 \
  /azwestus3/pclancy3_lab/ycao73/qe-7.3.1.sif \
  pw.x -inp cand_0001.scf.in > scf.out 2> scf.err
rc=$?
printf '%s\n' "$rc" > exit_code.txt
exit "$rc"
