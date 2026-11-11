#!/usr/bin/env python3
"""Response Table R8 (tab:resp-c10-grouped-split): grouped 600 K partitions.

Frame counts and sha256 prefixes come from the zero-force training audit, which hashed the
grouped A2 train/valid/test files on Rockfish. Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import Evidence  # noqa: E402

SUMMARY = "paper/revision1/data_processed/current_release/zero_force_training_summary.json"
SPLITS = [("Train", "grouped_FT600K"), ("Validation", "grouped_FT600K_valid"), ("Test", "grouped_FT600K_test")]
PREFIX = 12  # caption: hashes abbreviated to the first 12 hex characters


def main() -> int:
    ev = Evidence(__file__, "tab:resp-c10-grouped-split")
    files = {f["dataset"]: f for f in json.loads(ev.input(SUMMARY).read_text())["files"]}
    rows = [[label, f"{files[key]['frames']:,}", f"\\texttt{{{files[key]['sha256'][:PREFIX]}}}"]
            for label, key in SPLITS]
    ev.compare_table(rows, "response_to_reviewers.tex")
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
