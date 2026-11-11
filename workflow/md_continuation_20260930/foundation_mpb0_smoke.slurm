#!/bin/bash -l
#SBATCH --job-name=foundation-mpb0-smoke
#SBATCH --account=pclancy3_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch16/pclancy3/yi/mace_foundation_mpb0_smoke_20261006/logs/%A.out
#SBATCH --error=/scratch16/pclancy3/yi/mace_foundation_mpb0_smoke_20261006/logs/%A.err

set -euo pipefail
export OMP_NUM_THREADS=4
PY=/data/pclancy3/yi/.conda/envs/mace/bin/python
ROOT=/scratch16/pclancy3/yi/mace_foundation_mpb0_smoke_20261006
INPUT=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/AI4MAT25/mlff-benchmark-cr-sb2te3/results/MD/3-multi_T-fine-tuning/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz
MODEL=/scratch16/pclancy3/yi/foundation_grouped_benchmark_20261001/models/mace-mp-0b3-medium.model
SCRIPT=/scratch16/pclancy3/yi/mace_foundation_mpb0_smoke_20261006/seedreplica_smoke.py
OUT="$ROOT/MP-0b3-medium"

mkdir -p "$ROOT/logs"
set +e
"$PY" "$SCRIPT" --input "$INPUT" --model "$MODEL" \
  --model-label Foundation-MP-0b3 --training-seed 0 --md-seed 1100 \
  --output "$OUT" --steps 20000 --start-step 200000 \
  --temperature-K 600 --device cuda
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
  exit "$rc"
fi
"$PY" - "$OUT/status.json" <<'PY'
import json, sys
status = json.load(open(sys.argv[1]))
if status.get("smoke_pass") is not True or status.get("reason") != "target_steps":
    raise SystemExit("smoke did not pass; refusing to release 1 ns continuation")
print("smoke gate passed")
PY
