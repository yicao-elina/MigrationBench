#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=/scratch16/pclancy3/yi/grouped_A2_ft_1ns_md_20261025
PROGRAM="$WORK_ROOT/md15_work_r01/compute_md15"
LOCAL_MANIFEST=/tmp/md15_raw_r02/md15_inputs.tsv
ANALYSIS_ROOT="$WORK_ROOT/md15_analysis_r03"
PART_ROOT="$ANALYSIS_ROOT/parts"
LOG_ROOT="$WORK_ROOT/md15_work_r01/md15_r03_logs"
SPLIT_ROOT=/tmp/md15_r03_manifests
MAX_PARALLEL=4

if [[ "$(hostname)" != "login03" ]]; then
  echo "wrong_login_host=$(hostname)" >&2
  exit 99
fi

rm -rf "$ANALYSIS_ROOT" "$SPLIT_ROOT" "$LOG_ROOT"
mkdir -p "$PART_ROOT" "$SPLIT_ROOT" "$LOG_ROOT"

while IFS=$'\t' read -r model_id family min_step source_path; do
  [[ -z "${model_id:-}" || "$model_id" == \#* ]] && continue
  printf '%s\t%s\t%s\t%s\n' "$model_id" "$family" "$min_step" "$source_path" >> "$SPLIT_ROOT/$model_id.tsv"
done < "$LOCAL_MANIFEST"

pids=()
for manifest in "$SPLIT_ROOT"/*.tsv; do
  model_id=$(basename "$manifest" .tsv)
  while (( ${#pids[@]} >= MAX_PARALLEL )); do
    next=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then next+=("$pid"); fi
    done
    pids=("${next[@]}")
    ((${#pids[@]} < MAX_PARALLEL)) && break
    sleep 2
  done
  mkdir -p "$PART_ROOT/$model_id"
  "$PROGRAM" "$manifest" "$PART_ROOT/$model_id" 20 12 0.1 > "$LOG_ROOT/$model_id.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do wait "$pid"; done

mkdir -p "$ANALYSIS_ROOT/series" "$ANALYSIS_ROOT/msd" "$ANALYSIS_ROOT/vacf" "$ANALYSIS_ROOT/rdf"
first=1
: > "$ANALYSIS_ROOT/analysis_summary.tsv"
for manifest in "$SPLIT_ROOT"/*.tsv; do
  model_id=$(basename "$manifest" .tsv)
  for kind in series msd vacf rdf; do cp "$PART_ROOT/$model_id/$kind/"*.csv "$ANALYSIS_ROOT/$kind/"; done
  if (( first )); then
    cp "$PART_ROOT/$model_id/analysis_summary.tsv" "$ANALYSIS_ROOT/analysis_summary.tsv"
    first=0
  else
    tail -n +2 "$PART_ROOT/$model_id/analysis_summary.tsv" >> "$ANALYSIS_ROOT/analysis_summary.tsv"
  fi
done
echo "completed_models=$(find "$ANALYSIS_ROOT/series" -maxdepth 1 -type f -name '*.csv' | wc -l | tr -d ' ')"
