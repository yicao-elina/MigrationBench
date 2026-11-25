#!/bin/bash -l
# Rockfish-only continuation wrapper for the three existing Cr-substitution NEBs.
# The source run is never modified. Submit from one prepared restart directory.
#SBATCH --time=71:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ycao73@jh.edu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# The preparation script overrides --nodes to match the original path shape:
# two nodes for within-QL/gap-to-QL and one node for within-gap.

set -euo pipefail

RUN_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
cd "$RUN_DIR"

date -Is > runtime_start.txt
{
  echo "campaign=MigrationBench_rockfish_cr_neb_20261013"
  echo "scope=Rockfish-only; Discovery is out of scope"
  echo "run_dir=$RUN_DIR"
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "submit_dir=$SLURM_SUBMIT_DIR"
  echo "host=$(hostname)"
  echo "user=${USER:-unknown}"
  echo "slurm_nnodes=${SLURM_NNODES:-unknown}"
  echo "slurm_ntasks=${SLURM_NTASKS:-unknown}"
  echo "restart_mode=$(grep -E "^[[:space:]]*restart_mode" neb.in | head -1 || true)"
  echo "container=none; native Rockfish QE module"
} > runtime_provenance.txt

ml intel intel-mpi intel-mkl
ml qe/7.2
export PATH=/cm/shared/apps/slurm/current/bin:$PATH

{
  echo "--- module list ---"
  module list 2>&1 || true
  echo "--- executables ---"
  type mpirun 2>&1 || true
  type neb.x 2>&1 || true
  command -v neb.x 2>/dev/null || true
  echo "--- input and checkpoint hashes ---"
  sha256sum neb.in pw_1.in pw_2.in sb2te3.path* 2>/dev/null || true
  echo "--- pseudopotential hashes ---"
  sha256sum \
    /data/pclancy3/yi/ONCVPseudoPack/SG15_v1.2/UPF/PBE/Cr_PBE_FR.SG15v1.2.UPF \
    /data/pclancy3/yi/ONCVPseudoPack/SG15_v1.2/UPF/PBE/Sb_PBE_FR.SG15v1.2.UPF \
    /data/pclancy3/yi/ONCVPseudoPack/SG15_v1.2/UPF/PBE/Te_PBE_FR.SG15v1.2.UPF 2>/dev/null || true
} >> runtime_provenance.txt

echo "restart wrapper: source path checkpoints were copied into this isolated directory" >> runtime_provenance.txt
echo "restart wrapper: old source out/ was intentionally not copied; this run uses a fresh ./out/" >> runtime_provenance.txt

set +e
mpirun -launcher slurm -n "${SLURM_NTASKS}" neb.x -inp neb.in > neb.out 2> neb.err
RC=$?
set -e

printf '%s\n' "$RC" > exit_code.txt
{
  echo "job_id=${SLURM_JOB_ID:-unknown}"
  echo "rc=$RC"
  if grep -q "JOB DONE" neb.out 2>/dev/null; then echo "JOB_DONE=1"; else echo "JOB_DONE=0"; fi
  if grep -q "activation energy (->)" neb.out 2>/dev/null; then echo "FORWARD_BARRIER=1"; else echo "FORWARD_BARRIER=0"; fi
  if grep -q "activation energy (<-)" neb.out 2>/dev/null; then echo "REVERSE_BARRIER=1"; else echo "REVERSE_BARRIER=0"; fi
  if grep -Eiq "nan|inf|error in routine|stopping|mpi|hydra|bstrap_proxy|convergence NOT achieved|cannot create fifo|errno" neb.err neb.out 2>/dev/null; then echo "ERROR_OR_WARNING_MARKER=1"; else echo "ERROR_OR_WARNING_MARKER=0"; fi
  echo "last_iteration=$(grep -E "iteration[[:space:]]+[0-9]+" neb.out 2>/dev/null | tail -1 | sed -E "s/.*iteration[[:space:]]+([0-9]+).*/\1/" || true)"
} > gate_markers.txt

date -Is > runtime_end.txt
exit "$RC"
