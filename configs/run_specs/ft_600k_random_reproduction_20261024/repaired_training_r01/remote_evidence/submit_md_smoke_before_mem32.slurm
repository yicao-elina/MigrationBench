#!/bin/bash -l
# Conditional 20 ps smoke gate for the two verified FT-600K repair models.
# Each successful smoke task submits its own 1 ns continuation with afterok.
# The model gate job is supplied at submission time as an afterok dependency.
#SBATCH --job-name=ft600k-1ns-smoke-r01
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch16/pclancy3/yi/ft_600k_random_reproduction_1ns_20261024_r01/logs/%x-%A_%a.out
#SBATCH --error=/scratch16/pclancy3/yi/ft_600k_random_reproduction_1ns_20261024_r01/logs/%x-%A_%a.err

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
SMOKE=/scratch16/pclancy3/yi/mace_four_model_seedreplica_smoke_retry_20261004/seedreplica_smoke.py
CONT=/scratch16/pclancy3/yi/ft_600k_random_reproduction_1ns_20261024_r01/continue_ft600k_1ns_r01.slurm
INPUT=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/AI4MAT25/mlff-benchmark-cr-sb2te3/results/MD/3-multi_T-fine-tuning/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz
MODEL_ROOT=/scratch16/pclancy3/yi/ft_600k_random_reproduction_20261019_repair_r01/models
ROOT_BASE=/scratch16/pclancy3/yi/ft_600k_random_reproduction_1ns_20261024_r01
SEEDS=(234 345)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="ft_600k_random_reproduction_repair_r01_seed${SEED}"
MODEL="$MODEL_ROOT/$MODEL_NAME/${MODEL_NAME}_stagetwo_compiled.model"
ROOT="$ROOT_BASE/seed${SEED}"
RUN="$ROOT/smoke"

test -s "$SMOKE"
test -s "$INPUT"
test -s "$MODEL"
mkdir -p "$ROOT/logs"
test ! -e "$RUN" || { echo "refusing to overwrite existing smoke directory: $RUN" >&2; exit 43; }

{
  echo "campaign_id=ft600k_random_reproduction_1ns_20261024_r01"
  echo "model_name=$MODEL_NAME"
  echo "training_seed=$SEED"
  echo "md_velocity_seed=1100"
  echo "slurm_job_id=${SLURM_JOB_ID:-unknown}"
  echo "array_task_id=${SLURM_ARRAY_TASK_ID:-unknown}"
  echo "host=$(hostname)"
  sha256sum "$MODEL" "$INPUT" "$SMOKE"
} | tee "$ROOT/preflight.txt"

"$PY" "$SMOKE" --input "$INPUT" --model "$MODEL" \
  --model-label "FT-600K-r01-seed${SEED}" --training-seed "$SEED" --md-seed 1100 \
  --output "$RUN" --steps 20000 --start-step 200000 --temperature-K 600 --device cuda

"$PY" - "$RUN/status.json" <<'PY'
import json
import sys

status = json.load(open(sys.argv[1]))
if status.get("smoke_pass") is not True or status.get("reason") != "target_steps":
    raise SystemExit(f"smoke gate failed: {status}")
PY

echo "smoke gate passed; releasing 1 ns continuation for seed $SEED"
CONT_JOB=$(sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" \
  --export=ALL,FT_SEED="$SEED" --job-name="ft600k-1ns-${SEED}" "$CONT")
echo "continuation_job_id=$CONT_JOB"
