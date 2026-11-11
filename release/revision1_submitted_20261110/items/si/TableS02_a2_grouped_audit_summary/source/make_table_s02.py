#!/usr/bin/env python3
"""SI Table S2 (tab:a2_grouped_audit_summary): legacy grouped G1 control.

G1 (audit path 1-4) absolute errors per training seed come from the grouped A2 seed comparison
and are cross-checked against the per-model NEB barrier summary. Train-frame counts come from the
zero-force training audit, which counted the grouped train files on Rockfish.
Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import Evidence, fmt_int, fmt_num, fmt_pm  # noqa: E402

A2 = "paper/response_to_reviewers/reviewer_a2_grouped_retraining/data"
SEEDS_CSV = f"{A2}/grouped_seed_comparison.csv"
NEB_CSV = f"{A2}/grouped_neb_barrier_summary.csv"
SUMMARY = "paper/revision1/data_processed/current_release/zero_force_training_summary.json"
G1_AUDIT_ID = "1-4"
DECIMALS = 2

# (row label, family in grouped_seed_comparison.csv, model_id prefix in the NEB summary,
#  training dataset in the zero-force summary, or None for the zero-shot checkpoint)
ROUTES = [
    ("FT-600K", "OMAT-FT", "ft600k_omatft", "grouped_FT600K"),
    ("Scratch", "Scratch", "scratch_grouped", "grouped_FT600K"),
    ("Scratch-5\\%", "Scratch-5%", "scratch5_grouped", "grouped_scratch5"),
    ("MPA0", "MPA0", "ft600k_mpa0", None),
]
ZERO_SHOT_FRAMES = "zero-shot"
ZERO_SHOT_SEEDS = "checkpoint"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    ev = Evidence(__file__, "tab:a2_grouped_audit_summary")
    seeds = read_csv(ev.input(SEEDS_CSV))
    neb = [r for r in read_csv(ev.input(NEB_CSV)) if r["pathway"] == G1_AUDIT_ID]
    frames = {f["dataset"]: f["frames"] for f in json.loads(ev.input(SUMMARY).read_text())["files"]}

    rows, mismatches = [], []
    for label, family, prefix, dataset in ROUTES:
        errors = [float(r["pathway_1_4_abs_error_eV"]) for r in seeds if r["family"] == family]
        neb_errors = sorted(float(r["absolute_error_ev"]) for r in neb if r["model_id"].startswith(prefix))
        if sorted(round(e, 6) for e in errors) != [round(e, 6) for e in neb_errors]:
            mismatches.append({"route": label, "seed_csv": errors, "neb_csv": neb_errors})
        if dataset is None:
            rows.append([label, ZERO_SHOT_FRAMES, fmt_num(errors[0], DECIMALS), ZERO_SHOT_SEEDS])
        else:
            cell = fmt_pm(statistics.mean(errors), statistics.stdev(errors), DECIMALS)
            rows.append([label, fmt_int(frames[dataset]), cell, str(len(errors))])
    ev.check("G1 errors in grouped_seed_comparison equal grouped_neb_barrier_summary (path 1-4)",
             not mismatches, mismatches or "all routes agree to 1e-6 eV")
    ev.compare_table(rows, "sn-article-SI.tex")
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
