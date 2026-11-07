#!/bin/bash
# Minimal project-filesystem scratch smoke test; QE WFC writes are disabled.
#SBATCH --job-name=mb_ep_psmoke
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --partition=azgpuh200
#SBATCH -A pclancy3_lab
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003/logs/%x-%j.out
#SBATCH --error=/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003/logs/%x-%j.err
set -euo pipefail
: "${ENDPOINT_ID:?ENDPOINT_ID required}"
N="${RELAX_NSTEP:-10}"
ROOT=/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003/endpoint_relax_47
RUN="$ROOT/results/${RUN_SUBDIR:-project_scratch_${ENDPOINT_ID}_${N}step}"
IN="$ROOT/inputs/$ENDPOINT_ID/relax_$ENDPOINT_ID.in"
BASE="$ROOT/qe_scratch"
JOB="$BASE/${ENDPOINT_ID}_${SLURM_JOB_ID}"
OUT="$JOB/qe_out"
mkdir -p "$RUN" "$OUT" "$JOB/tmp"
cp "$IN" "$RUN/input.relax.in"
cd "$RUN"
df -h "$JOB" > scratch_df_before.txt
df -ih "$JOB" > scratch_df_inode_before.txt
cp input.relax.in input.relax.runtime.in
awk -v d="$OUT" -v n="$N" '
BEGIN{s=sprintf("%c",39);q=sprintf("%c",34);c=0;di=0;ns=0}
/^[[:space:]]*&CONTROL/{c=1;print;next}
c&&/^[[:space:]]*disk_io[[:space:]]*=/{print "  disk_io = "s"none"s;di=1;next}
c&&/^[[:space:]]*nstep[[:space:]]*=/{print "  nstep = "n;ns=1;next}
c&&/^[[:space:]]*\/[[:space:]]*$/{if(!di)print "  disk_io = "s"none"s;if(!ns)print "  nstep = "n;c=0}
/^[[:space:]]*outdir[[:space:]]*=/{print "  outdir = "q d q;next}{print}' input.relax.runtime.in > x && mv x input.relax.runtime.in
export OMP_NUM_THREADS=11 OMP_STACKSIZE=2G TMPDIR="$JOB/tmp" APPTAINERENV_TMPDIR="$JOB/tmp"
export SINGULARITY_BIND="${SINGULARITY_BIND:-}"
source /projects/pclancy3_lab/singularity.sh
export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}$JOB:$JOB,$ROOT:$ROOT"
{
 echo "candidate=$ENDPOINT_ID job=$SLURM_JOB_ID scratch=$JOB outdir=$OUT"
 echo 'settings=disk_io=none,nstep='"$N"',outdir=project_filesystem'
 df -h "$JOB"; df -ih "$JOB"; sha256sum input.relax.in input.relax.runtime.in
} > runtime_provenance.txt
set +e
srun --export=ALL singularity exec --nv --cleanenv /projects/pclancy3_lab/qe75-h200-v2.sif \
 bash --noprofile --norc -c '
  export OPAL_PREFIX=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi
  export PATH=/opt/qe/bin:$OPAL_PREFIX/bin:$PATH
  export LD_LIBRARY_PATH=$OPAL_PREFIX/lib:${LD_LIBRARY_PATH:-}
  export UCX_MEMTYPE_CACHE=n UCX_TLS=self,shm,tcp OMPI_MCA_opal_cuda_support=false
  export HWLOC_HIDE_ERRORS=1 OMPI_MCA_hwloc_base_binding_policy=none
  export OMPI_MCA_rmaps_base_mapping_policy=slot CUDA_LAUNCH_BLOCKING=1
  "$OPAL_PREFIX/bin/mpirun" --bind-to none --map-by slot -np 1 pw.x -inp input.relax.runtime.in
' > scf.out 2> scf.err
rc=$?
grep 'JOB DONE' scf.out >/dev/null && done=1 || done=0
du -sh "$JOB" > scratch_usage_before_cleanup.txt 2>/dev/null || true
awk '/ATOMIC_POSITIONS/{p=1;n=0;next} p&&NF>=4&&$1~/^[A-Za-z]/{a[++n]=$0;next} p&&n>0{if(n==81){for(i=1;i<=81;i++)print a[i]}p=0} END{if(n==81)for(i=1;i<=81;i++)print a[i]}' scf.out > relaxed_last_step.xyz.tmp || true
test "$(wc -l < relaxed_last_step.xyz.tmp)" -eq 81 && mv relaxed_last_step.xyz.tmp relaxed_last_step.xyz || rm -f relaxed_last_step.xyz.tmp
echo "$rc" > exit_code.txt
grep -E 'JOB DONE|!    total energy|Forces acting on atoms|Total force|Error in routine|stopping|convergence NOT achieved' scf.out > gate_markers.txt || true
echo "rc=$rc job_done=$done" >> runtime_provenance.txt
if ((rc==0 && done==1)); then echo cleanup=removed >> runtime_provenance.txt; rm -rf "$JOB"; else echo cleanup=preserved >> runtime_provenance.txt; fi
exit "$rc"
