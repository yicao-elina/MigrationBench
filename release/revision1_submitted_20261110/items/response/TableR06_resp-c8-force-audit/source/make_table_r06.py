#!/usr/bin/env python3
"""Response Table R6 (tab:resp-c8-force-audit): zero-force training-record audit.

Counts come from the frame-level manifest written by
release/revision1_camera_ready/reproducibility/build_zero_force_training_manifest.py
(one row per training frame) and are cross-checked against its summary JSON.
Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import Evidence, fmt_int  # noqa: E402

RELEASE = "paper/revision1/data_processed/current_release/"
MANIFEST = RELEASE + "zero_force_training_manifest.csv"
SUMMARY = RELEASE + "zero_force_training_summary.json"
ISOLATED = "isolated_atom_reference"     # zero_force_class values written by the audit script
UNCLASSIFIED = "unclassified_zero_force"
NEB_LINEAGE = "neb_exporter_lineage_present"  # source_exporter value for NEB-exported frames


def main() -> int:
    ev = Evidence(__file__, "tab:resp-c8-force-audit")
    frames = pd.read_csv(ev.input(MANIFEST), usecols=["dataset", "frame_index", "natoms", "all_zero_force",
                                                      "zero_force_class", "source_exporter"])
    summary = json.loads(ev.input(SUMMARY).read_text())
    zero = frames[frames["all_zero_force"]]
    counts = {
        "total": len(frames),
        "zero": len(zero),
        "isolated": int((zero["zero_force_class"] == ISOLATED).sum()),
        "unclassified": int((zero["zero_force_class"] == UNCLASSIFIED).sum()),
        "neb": int((zero["source_exporter"] == NEB_LINEAGE).sum()),
    }
    ev.check("manifest counts equal the summary JSON",
             (counts["total"], counts["zero"], counts["unclassified"]) ==
             (summary["total_frames"], summary["total_all_zero_force_frames"],
              summary["total_unclassified_zero_force_frames"]),
             {"manifest": counts, "summary": {k: v for k, v in summary.items() if k.startswith("total")}})
    ev.check("isolated-atom references are single-atom frames", bool((zero["natoms"] == 1).all()),
             zero[["dataset", "frame_index", "natoms"]].to_dict("records"))
    ev.check("zero-force frames per dataset equal the per-file summary",
             zero.groupby("dataset").size().to_dict() ==
             {f["dataset"]: f["all_zero_force_frames"] for f in summary["files"] if f["all_zero_force_frames"]},
             zero.groupby("dataset").size().to_dict())
    rows = [
        ["Total records inspected", fmt_int(counts["total"])],
        ["Records with identically zero forces", fmt_int(counts["zero"])],
        ["Zero-force isolated-atom energy references", fmt_int(counts["isolated"])],
        ["Unclassified zero-force records", fmt_int(counts["unclassified"])],
        ["NEB-derived zero-force records", fmt_int(counts["neb"])],
    ]
    ev.compare_table(rows, "response_to_reviewers.tex")
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
