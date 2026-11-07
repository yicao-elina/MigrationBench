#!/bin/bash
#SBATCH -A pclancy3_lab
#SBATCH --partition=azgpuh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --job-name=cr-neb
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail

ROOT="${CR_NEB_ROOT:?set CR_NEB_ROOT to the lab-space campaign directory}"
CASE="${CR_NEB_CASE:?set CR_NEB_CASE to one generated path id}"
WORK="$ROOT/$CASE"
mkdir -p "$WORK/out"
cp "$CR_NEB_INPUT" "$WORK/neb.in"
cd "$WORK"

export SINGULARITY_BIND="${SINGULARITY_BIND:-}"
source /azwestus3/pclancy3_lab/singularity.sh

QE_SIF="${QE_SIF:?set QE_SIF to the validated QE container}"
NEBX="${NEBX:-neb.x}"
if command -v "$NEBX" >/dev/null 2>&1; then
  "$NEBX" -in neb.in > neb.out
elif command -v mpirun >/dev/null 2>&1; then
  mpirun -np "${SLURM_NTASKS:-1}" singularity exec --nv "$QE_SIF" neb.x -in neb.in > neb.out
else
  singularity exec --nv "$QE_SIF" neb.x -in neb.in > neb.out
fi

grep -q "JOB DONE" neb.out
! grep -q "SCF convergence NOT achieved" neb.out
