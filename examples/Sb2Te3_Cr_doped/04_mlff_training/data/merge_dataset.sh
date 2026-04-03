#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Usage:
#   ./merge_xyz_datasets_by_frames_ratio.sh OUTPUT_DIR folder1:ratio folder2:ratio ...
#
# Example:
#   ./merge_xyz_datasets_by_frames_ratio.sh merged_data \
#       data/folderA:0.2 data/folderB:0.05 data/folderC:0.1
#
# Description:
#   - Each input folder must contain train.xyz, valid.xyz, test.xyz
#   - Each folder has its own sampling ratio
#   - Sampling is done per FRAME (not by lines)
#   - Each frame (N_atoms + 2 lines) is preserved completely
#   - Output:
#       OUTPUT_DIR/merged_train.xyz
#       OUTPUT_DIR/merged_valid.xyz
#       OUTPUT_DIR/merged_test.xyz
# ============================================================

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 OUTPUT_DIR folder1:ratio [folder2:ratio ...]"
  exit 1
fi

OUTDIR="$1"
shift
mkdir -p "$OUTDIR"

# ============================================================
# Helper function: count frames in .xyz file
# ============================================================
count_frames() {
  local file="$1"
  if [ ! -s "$file" ]; then
    echo 0
    return
  fi
  local n_atoms
  n_atoms=$(head -n 1 "$file")
  local total_lines
  total_lines=$(wc -l < "$file")
  echo $(( total_lines / (n_atoms + 2) ))
}

# ============================================================
# Helper function: sample frames with ratio, preserving integrity
# ============================================================
sample_xyz_frames() {
  local infile="$1"
  local outfile="$2"
  local ratio="$3"

  if [ ! -s "$infile" ]; then
    echo "  [!] Empty file $infile"
    return
  fi

  local n_atoms
  n_atoms=$(head -n 1 "$infile")
  local lines_per_frame=$(( n_atoms + 2 ))
  local total_lines
  total_lines=$(wc -l < "$infile")
  local total_frames=$(( total_lines / lines_per_frame ))

  if (( total_frames == 0 )); then
    echo "  [!] File malformed: $infile"
    return
  fi

  # Integer number of frames to sample
  local sample_frames
  sample_frames=$(awk -v n="$total_frames" -v r="$ratio" 'BEGIN{printf "%d", n*r}')
  if (( sample_frames < 1 )); then
    sample_frames=1
  fi

  echo "  [+] $infile → sampling ${sample_frames}/${total_frames} frames"

  # Randomly sample distinct frame indices
  seq 0 $(( total_frames - 1 )) | shuf -n "$sample_frames" | sort -n | while read -r idx; do
    start=$(( idx * lines_per_frame + 1 ))
    end=$(( start + lines_per_frame - 1 ))
    sed -n "${start},${end}p" "$infile" >> "$outfile"
  done
}

# ============================================================
# Main merging loop
# ============================================================
for split in train valid test; do
  merged_file="$OUTDIR/merged_${split}.xyz"
  rm -f "$merged_file"
  touch "$merged_file"
  echo "--------------------------------------------"
  echo "Merging ${split}.xyz → $merged_file"

  for pair in "$@"; do
    dir="${pair%%:*}"
    ratio="${pair##*:}"
    file="$dir/${split}.xyz"

    if [ ! -f "$file" ]; then
      echo "  [!] Missing file: $file (skipped)"
      continue
    fi

    # Validate ratio numeric range
    if ! [[ "$ratio" =~ ^0(\.[0-9]+)?$|^1(\.0+)?$ ]]; then
      echo "  [!] Invalid ratio '$ratio' for $dir. Must be between 0–1."
      continue
    fi

    sample_xyz_frames "$file" "$merged_file" "$ratio"
  done
done

echo "--------------------------------------------"
echo "✅ All merged files written to: $OUTDIR"
echo "  - merged_train.xyz"
echo "  - merged_valid.xyz"
echo "  - merged_test.xyz"
