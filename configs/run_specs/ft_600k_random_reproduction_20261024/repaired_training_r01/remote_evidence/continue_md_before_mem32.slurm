#!/bin/bash -l
# 1 ns continuation for one FT-600K repair model after its smoke gate.
# A signal-safe checkpoint is resubmitted afterany without overwriting the run.
#SBATCH --job-name=ft600k-1ns
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/scratch16/pclancy3/yi/ft_600k_random_reproduction_1ns_20261024_r01/logs/%x-%A.out
#SBATCH --error=/scratch16/pclancy3/yi/ft_600k_random_reproduction_1ns_20261024_r01/logs/%x-%A.err

set -euo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export RF_CONDA_SITE="${RF_CONDA_SITE:-}"
export SINGULARITY_BIND="${SINGULARITY_BIND:-}"
source /etc/profile.d/modules.sh
set +u
source /data/apps/go.sh
source /data/apps/helpers/singularity.sh
module restore
module unload openmpi/3.1.6 >/dev/null 2>&1 || true
ml openmpi/4.1.6
source /data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh
conda activate mace
set -u

PY=/data/pclancy3/yi/.conda/envs/mace/bin/python
CONT_DRIVER=/scratch16/pclancy3/yi/mace_foundation_representatives_1ns_20261006/seedreplica_continue_1ns.py
INPUT=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/AI4MAT25/mlff-benchmark-cr-sb2te3/results/MD/3-multi_T-fine-tuning/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz
MODEL_ROOT=/scratch16/pclancy3/yi/ft_600k_random_reproduction_20261019_repair_r01/models
ROOT_BASE=/scratch16/pclancy3/yi/ft_600k_random_reproduction_1ns_20261024_r01
SEED="${FT_SEED:?FT_SEED was not exported by the smoke job}"
case "$SEED" in
  234|345) ;;
  *) echo "unsupported FT_SEED=$SEED" >&2; exit 2 ;;
esac
MODEL_NAME="ft_600k_random_reproduction_repair_r01_seed${SEED}"
MODEL="$MODEL_ROOT/$MODEL_NAME/${MODEL_NAME}_stagetwo_compiled.model"
ROOT="$ROOT_BASE/seed${SEED}"
RUN="$ROOT/continuation"

test -s "$PY"
test -s "$CONT_DRIVER"
test -s "$INPUT"
test -s "$MODEL"
test -s "$ROOT/smoke/status.json"

set +e
"$PY" "$CONT_DRIVER" --source-dir "$ROOT/smoke" --input "$INPUT" --model "$MODEL" \
  --model-label "FT-600K-r01-seed${SEED}" --training-seed "$SEED" --md-seed 1100 \
  --run-dir "$RUN" --target-step 1000000 --temperature-K 600 \
  --device cuda --write-interval 200 --checkpoint-interval 5000
rc=$?
set -e

if [ "$rc" -eq 99 ]; then
  echo "walltime/stop checkpoint saved; submitting non-overwriting continuation"
  NEXT_JOB=$(sbatch --parsable --dependency="afterany:${SLURM_JOB_ID}" \
    --export=ALL,FT_SEED="$SEED" --job-name="ft600k-1ns-${SEED}" "$0")
  echo "continuation_job_id=$NEXT_JOB"
  exit 0
fi
exit "$rc"
