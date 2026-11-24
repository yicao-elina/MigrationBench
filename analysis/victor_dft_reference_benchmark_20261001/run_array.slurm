#!/bin/bash
# Isolated Victor benchmark; never points at an existing campaign result root.
#SBATCH --job-name=victor-mace-20261001
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-540%10
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -eo pipefail
RUN_ROOT="${RUN_ROOT:-/scratch16/pclancy3/yi/victor_dft_reference_mace_20261001}"
source /etc/profile.d/modules.sh
MACE_ENV="${MACE_ENV:-/data/pclancy3/yi/.conda/envs/mace}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH="$RUN_ROOT/python_pkgs:$RUN_ROOT/src:${PYTHONPATH:-}"
ROW=$(awk -F '\t' -v n="$SLURM_ARRAY_TASK_ID" 'NR==n+1 {print}' "$RUN_ROOT/run_matrix.tsv")
IFS=$'\t' read -r idx model_id model_label model_family seed model_path task_json task_id <<< "$ROW"
OUT="$RUN_ROOT/results/$model_id/$task_id"
mkdir -p "$OUT"
set +e
"$MACE_ENV/bin/python" "$RUN_ROOT/run_endpoint_relax.py" \
  --task-json "$RUN_ROOT/$task_json" --input-root "$RUN_ROOT/inputs" --out-dir "$OUT" \
  --model-label "$model_label" --model-path "$model_path" --device cpu --steps 500 --fmax 0.03
rc=$?
set -e
if [ ! -f "$OUT/mace_relax_summary.json" ]; then
  python - "$OUT" "$rc" "$task_id" "$model_id" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True)
(p/'failure.json').write_text(json.dumps({'task_id':sys.argv[3],'model_id':sys.argv[4], 'returncode':int(sys.argv[2]), 'scientific_success':False},indent=2)+'\n')
PY
fi
exit "$rc"
