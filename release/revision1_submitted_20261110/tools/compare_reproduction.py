#!/usr/bin/env python3
"""Compare rerun CSV outputs with the supplied ones (writes Table R11 / R10 checks).

Reconstructs the comparison behind ``shap_reproduction_check.json`` (response
Table R11), whose original script was not archived. For each named CSV it
reports both shapes, whether column names match, the maximum absolute
difference over numeric columns, and whether text columns are identical.

Example (SHAP, reference env: release/revision1_camera_ready/requirements-current.txt):
    python3 compare_reproduction.py --source analysis/shap_full_campaign_20261101/results \
        --rerun <rerun-dir> --files shap_seed_metrics.csv shap_family_aggregate.csv \
        shap_seed_feature_importance.csv shap_frame_aligned_full_campaign.csv --output check.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def compare_csv(source: Path, rerun: Path) -> dict:
    left, right = pd.read_csv(source), pd.read_csv(rerun)
    result = {
        "shape_source": list(left.shape),
        "shape_rerun": list(right.shape),
        "column_names_match": list(left.columns) == list(right.columns),
    }
    if left.shape != right.shape or not result["column_names_match"]:
        return {**result, "max_abs_numeric_difference": None, "text_columns_match": False}
    numeric = left.select_dtypes("number").columns
    text = [column for column in left.columns if column not in numeric]
    difference = (left[numeric] - right[numeric]).abs().to_numpy()
    result["max_abs_numeric_difference"] = float(difference.max()) if difference.size else 0.0
    result["text_columns_match"] = bool(left[text].equals(right[text]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args()
    report, ok = {}, True
    for name in args.files:
        paths = [args.source / name, args.rerun / name]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            report[name], ok = {"missing": missing}, False
            continue
        report[name] = compare_csv(*paths)
        entry = report[name]
        ok &= bool(entry["column_names_match"] and entry["text_columns_match"]
                   and entry["max_abs_numeric_difference"] is not None
                   and entry["max_abs_numeric_difference"] <= args.tolerance)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
