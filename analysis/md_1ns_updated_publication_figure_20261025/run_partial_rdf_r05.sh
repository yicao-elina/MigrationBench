#!/usr/bin/env bash
set -euo pipefail

ROOT=/scratch16/pclancy3/yi/md15_partial_rdf_r05_run01
PROGRAM="$ROOT/compute_partial_rdf_r05"
MANIFEST="$ROOT/md15_inputs.tsv"
OUT="$ROOT/output"
SPLIT="$ROOT/manifests_corrected"
LOG="$ROOT/logs"
MAX_PARALLEL=4

if [[ "$(hostname)" != "login02" && "$(hostname)" != "login03" ]]; then
  echo "unexpected_host=$(hostname)" >&2
  exit 99
fi

mkdir -p "$OUT" "$SPLIT" "$LOG"

while IFS=$'\t' read -r model_id family min_step source_path; do
  [[ -z "${model_id:-}" || "$model_id" == \#* ]] && continue
  [[ -n "${MODEL_ONLY:-}" && "$model_id" != "$MODEL_ONLY" ]] && continue
  printf '%s\t%s\t%s\t%s\n' "$model_id" "$family" "$min_step" "$source_path" >> "$SPLIT/$model_id.tsv"
done < "$MANIFEST"

pids=()
for manifest in "$SPLIT"/*.tsv; do
  model_id=$(basename "$manifest" .tsv)
  [[ -n "${MODEL_ONLY:-}" && "$model_id" != "$MODEL_ONLY" ]] && continue
  while (( ${#pids[@]} >= MAX_PARALLEL )); do
    next=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then next+=("$pid"); fi
    done
    pids=("${next[@]}")
    ((${#pids[@]} < MAX_PARALLEL)) && break
    sleep 2
  done
  mkdir -p "$OUT/$model_id"
  "$PROGRAM" "$manifest" "$OUT/$model_id" 20 8 0.05 > "$LOG/$model_id.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done

echo "completed_models=$(find "$OUT" -mindepth 2 -maxdepth 2 -name '*.csv' | wc -l | tr -d ' ')"
