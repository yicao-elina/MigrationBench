#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=/scratch16/pclancy3/yi/grouped_A2_ft_1ns_md_20261025
ANALYSIS_ROOT=/scratch16/pclancy3/yi/grouped_A2_ft_1ns_md_20261025/md15_analysis_r02
SOURCE_MANIFEST=/scratch16/pclancy3/yi/grouped_A2_ft_1ns_md_20261025/md15_work_r01/md15_inputs.tsv
PROGRAM=/scratch16/pclancy3/yi/grouped_A2_ft_1ns_md_20261025/md15_work_r01/compute_md15
LOCAL_ROOT=/tmp/md15_raw_r02

if [[ "$(hostname)" != "login03" ]]; then
  echo "wrong_login_host=$(hostname)" >&2
  exit 99
fi

mkdir -p "$LOCAL_ROOT"
LOCAL_MANIFEST="$LOCAL_ROOT/md15_inputs.tsv"
: > "$LOCAL_MANIFEST"
while IFS=$'\t' read -r model_id family min_step source_path; do
  [[ -z "${model_id:-}" || "$model_id" == \#* ]] && continue
  model_dir="$LOCAL_ROOT/$model_id"
  mkdir -p "$model_dir"
  basename_path=$(basename "$source_path")
  local_path="$model_dir/$basename_path"
  if [[ ! -s "$local_path" ]]; then
    cp -- "$source_path" "$local_path"
  fi
  printf '%s\t%s\t%s\t%s\n' "$model_id" "$family" "$min_step" "$local_path" >> "$LOCAL_MANIFEST"
done < "$SOURCE_MANIFEST"

rm -rf "$ANALYSIS_ROOT"
"$PROGRAM" "$LOCAL_MANIFEST" "$ANALYSIS_ROOT" 20 12 0.1
