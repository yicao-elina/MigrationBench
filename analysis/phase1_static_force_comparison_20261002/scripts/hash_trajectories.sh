#!/bin/bash
set -euo pipefail
ROOT=/scratch16/pclancy3/yi/phase1_static_force_comparison_20261002
SRC=/scratch16/pclancy3/yi/mace_four_model_MD_continuation_20260930
sha256sum "$SRC/1-from-scratch/segment_from_000200000.extxyz" > "$ROOT/logs/hash_from_scratch.txt" &
sha256sum "$SRC/2-naive-fine-tuning/segment_from_000080000.extxyz" > "$ROOT/logs/hash_naive_ft.txt" &
sha256sum "$SRC/3-multi_T-fine-tuning/segment_from_000200000.extxyz" > "$ROOT/logs/hash_multi_t.txt" &
sha256sum "$SRC/foundation-omat/segment_from_000008000.extxyz" > "$ROOT/logs/hash_foundation.txt" &
wait
cat "$ROOT"/logs/hash_*.txt > "$ROOT/logs/trajectory_sha256.txt"
