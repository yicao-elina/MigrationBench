#!/usr/bin/env python3
"""Response Table R11 (tab:resp-c13-shap-reproduction): SHAP campaign reproduction check.

Rows and columns are counted from the archived SHAP campaign outputs in the repo; the
maximum numerical difference is read from shap_reproduction_check.json (written by
tools/compare_reproduction.py against a rerun of run_shap_full_campaign.py in the
reference environment; the rerun CSVs themselves are not stored in the repo, so that
difference is read, not recomputed here). Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import Evidence, fmt_int, sha256  # noqa: E402

CHECK = "paper/revision1/data_processed/current_release/shap_reproduction_check.json"
RESULTS = "analysis/shap_full_campaign_20261101/results/"
RELEASE = "paper/revision1/data_processed/current_release/"
OUTPUTS = [  # (submitted row label, campaign output file)
    ("Per-seed metrics", "shap_seed_metrics.csv"),
    ("Family-feature aggregate", "shap_family_aggregate.csv"),
    ("Seed-feature importance", "shap_seed_feature_importance.csv"),
    ("Frame-aligned attributions", "shap_frame_aligned_full_campaign.csv"),
]


def main() -> int:
    ev = Evidence(__file__, "tab:resp-c13-shap-reproduction")
    check = json.loads(ev.input(CHECK).read_text())
    rows, shapes, copies = [], {}, {}
    for label, name in OUTPUTS:
        archived = ev.input(RESULTS + name)
        shape = list(pd.read_csv(archived).shape)
        entry = check[name]
        shapes[name] = {"archived": shape, "check_source": entry["shape_source"], "check_rerun": entry["shape_rerun"]}
        release_copy = Path(str(archived).replace(RESULTS, RELEASE))
        if release_copy.exists():
            copies[name] = sha256(ev.input(RELEASE + name)) == sha256(archived)
        passed = entry["column_names_match"] and entry["text_columns_match"]
        ev.check(f"{name}: column names and text columns match in the rerun", passed, entry)
        rows.append([label, fmt_int(shape[0]), fmt_int(shape[1]), f"{entry['max_abs_numeric_difference']:g}"])
    ev.check("archived output shapes equal source and rerun shapes in the check file",
             all(s["archived"] == s["check_source"] == s["check_rerun"] for s in shapes.values()), shapes)
    ev.check("current_release copies are byte-identical to the archived campaign outputs",
             all(copies.values()), copies)
    ev.compare_table(rows, "response_to_reviewers.tex")
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
