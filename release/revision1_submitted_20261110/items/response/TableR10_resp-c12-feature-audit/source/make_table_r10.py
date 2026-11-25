#!/usr/bin/env python3
"""Response Table R10 (tab:resp-c12-feature-audit): feature-matrix placeholder audit.

Recomputed from the Fig. 6 feature matrix (fig6_current_input.npz, 'features'): shape,
the all-zero column(s), and the columns kept after removal; cross-checked against
current_analysis_audit.json. Columns 6151-6153 are also zero in every frame but are
retained by the analysis; this is recorded in the evidence, not in the table. Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import Evidence, fmt_int  # noqa: E402

FEATURES = "release/revision1_camera_ready/reproducibility/fig6_current_input.npz"
AUDIT = "paper/revision1/data_processed/current_release/current_analysis_audit.json"


def main() -> int:
    ev = Evidence(__file__, "tab:resp-c12-feature-audit")
    with np.load(ev.input(FEATURES)) as archive:
        features = archive["features"]
    audit = json.loads(ev.input(AUDIT).read_text())
    frames, columns = features.shape
    placeholder = columns - 1  # the placeholder removed by recompute_feature_metrics.py (raw[:, -1])
    zero_frames = int((features[:, placeholder] == 0).sum())
    retained = columns - 1
    zero_columns = np.flatnonzero(~features.any(axis=0)).tolist()
    ev.check("final column is zero in every frame", zero_frames == frames,
             {"placeholder": placeholder, "zero_frames": zero_frames, "frames": frames})
    ev.check("note: other all-zero columns are retained (not part of the submitted claim)", True,
             {"all_zero_columns": zero_columns, "retained_all_zero": [c for c in zero_columns if c != placeholder],
              "note": "columns besides the placeholder are zero in every frame but are kept by the analysis; "
                      "the table and current_analysis_audit.json only describe the final placeholder column"})
    ev.check("recomputed audit equals current_analysis_audit.json",
             [frames, columns] == audit["feature_shape_before"] and [frames, retained] == audit["feature_shape_after"]
             and placeholder == audit["removed_column"] and audit["removed_column_all_zero"],
             {"recomputed": {"shape": [frames, columns], "removed": placeholder, "retained": retained},
              "audit_json": {k: audit[k] for k in ("feature_shape_before", "feature_shape_after", "removed_column")},
              "note": "audit_json feature_sha256 (of fig6_full_campaign_features.npz) differs from the sha256 of "
                      "this npz; shapes and the removed column agree"})
    scope = "all" if zero_frames == frames else "only"
    rows = [
        ["Frames", fmt_int(frames)],
        ["Input feature columns", fmt_int(columns)],
        ["Zero-valued placeholder", f"final column (index {fmt_int(placeholder)}; zero-based)"],
        ["Placeholder values equal to zero", f"{scope} {fmt_int(zero_frames)} frames"],
        ["Columns retained after removal", fmt_int(retained)],
    ]
    ev.compare_table(rows, "response_to_reviewers.tex")
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
