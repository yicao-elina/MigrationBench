#!/usr/bin/env python3
"""Prepare exact-path and SHA256 check inputs for the WSe2 WFC preflight."""

from __future__ import annotations

import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "candidate_wfc_manifest.tsv"
PATHS = HERE / "wse2_wfc_candidate_paths.txt"
HASHES = HERE / "wse2_wfc_candidate_sha256_check.txt"

with SOURCE.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
if len(rows) != 8:
    raise SystemExit(f"Expected eight WFC candidates, got {len(rows)}")
PATHS.write_text("".join(f"{row['artifact_path']}\n" for row in rows), encoding="utf-8")
HASHES.write_text("".join(f"{row['pre_delete_sha256']}  {row['artifact_path']}\n" for row in rows), encoding="utf-8")
print(f"Prepared {len(rows)} exact candidate paths and SHA256 checks.")
