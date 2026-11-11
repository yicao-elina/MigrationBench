#!/bin/bash -l
# Parameterized 20 ps grouped-model temperature smoke. Never reuse an output dir.
# Required exports: MODEL, MODEL_TAG, INPUT, TEMP, SEED, OUT_ROOT
# Rockfish submission requires SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
# and /cm/shared/apps/slurm/current/bin/sbatch.
#SBATCH --job-name=grp-md-smoke
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
: "${MODEL:?MODEL is required}"
: "${MODEL_TAG:?MODEL_TAG is required}"
: "${INPUT:?INPUT is required}"
: "${TEMP:?TEMP is required}"
: "${SEED:?SEED is required}"
: "${OUT_ROOT:?OUT_ROOT is required}"
START_STEP=${START_STEP:-0}
MACE_ENV=/data/pclancy3/yi/.conda/envs/mace
SCRIPT_ROOT=/scratch16/pclancy3/yi/AI4MAT25_MD_restart_20260926
OUT="$OUT_ROOT/${MODEL_TAG}/T${TEMP}_seed${SEED}"
mkdir -p "$OUT"
test ! -e "$OUT/segment_smoke20ps.xyz"
test ! -e "$OUT/status.json"
export PYTHONPATH=/data/pclancy3/yi/.python_pkgs:/data/pclancy3/yi/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$MACE_ENV/bin/python" - "$OUT/run_manifest.json" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

out = Path(sys.argv[1])
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

payload = {
    "input": os.path.abspath(os.environ["INPUT"]),
    "model": os.path.abspath(os.environ["MODEL"]),
    "model_tag": os.environ["MODEL_TAG"],
    "temperature_target_K": float(os.environ["TEMP"]),
    "seed": int(os.environ["SEED"]),
    "start_step": int(os.environ.get("START_STEP", "0")),
    "steps": 20000,
    "timestep_fs": 1.0,
    "friction": 0.01,
    "write_interval_steps": 200,
    "input_sha256": sha256(os.environ["INPUT"]),
    "model_sha256": sha256(os.environ["MODEL"]),
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
exec "$MACE_ENV/bin/python" "$SCRIPT_ROOT/restart_mace_md.py" \
  --input-xyz "$INPUT" --model-path "$MODEL" --output-dir "$OUT" \
  --segment smoke20ps --start-step "$START_STEP" --steps 20000 --interval 200 \
  --checkpoint-steps 2000 --temperature "$TEMP" --friction 0.01 \
  --timestep-fs 1.0 --device cuda --seed "$SEED"
