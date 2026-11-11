#!/usr/bin/env python3
"""Fig. S2: canonical in-gap fixed-path errors (a) and legacy train-path overlap audit (b).

The compact plotting input source_data/grouped_retraining_overlap_data.json is first checked
against its upstream tables (Fig. 5 in-gap fixed-path absolute errors as supplied, and the two
overlap leakage audits); the generator is then re-run into rerun_output/ and compared with the
submitted SI asset, which it rendered directly.

Writes rerun_output/ and evidence/FigS2.json.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import REPO, Evidence, lint_script  # noqa: E402

ITEM = "release/revision1_submitted_20261110/items/si/FigS2_grouped_retraining_overlap/source"
GENERATOR = f"{ITEM}/plot_grouped_retraining_overlap.py"
PLOT_DATA = f"{ITEM}/source_data/grouped_retraining_overlap_data.json"
FIG5_VALUES = "paper/revision1/data_processed/current_release/fig5_plotted_values_as_supplied.tsv"
OVERLAP = "paper/response_to_reviewers/reviewer_a2_grouped_retraining/data/overlap_{}_leakage_audit.json"
OVERLAP_SPLITS = {"Full grouped": "full", "Scratch-5%": "scratch5"}
FINAL_ASSET = "paper/revision1/figures/grouped_retraining_overlap.pdf"
IN_GAP_FIXED = "migration_in_gap_fixed"
JSON_DECIMALS = 6  # the plotting JSON stores errors at 6 decimals


def upstream_panel_a(ev: Evidence) -> dict:
    with ev.input(FIG5_VALUES).open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle, delimiter="\t") if r["column_id"] == IN_GAP_FIXED]
    trained, foundation, reference = {}, {}, set()
    for row in rows:
        value = round(float(row["absolute_error_eV"]), JSON_DECIMALS)
        if " / " in row["row_id"]:
            family, seed = row["row_id"].split(" / ")
            trained.setdefault(family, {})[seed] = value
        else:
            foundation[row["row_id"]] = value
            reference.add(float(row["reference_value_eV"]))
    return {"trained": {f: [v for _, v in sorted(s.items())] for f, s in trained.items()},
            "foundation": foundation, "reference": reference}


def upstream_panel_b(ev: Evidence) -> dict:
    minimums, thresholds = {}, set()
    for label, key in OVERLAP_SPLITS.items():
        audit = json.loads(ev.input(OVERLAP.format(key)).read_text())
        train = audit["summaries"]["train"]
        minimums[label] = {"soap_cosine": train["soap_cosine_distance"]["min"],
                           "cr_local_rmsd_angstrom": train["cr_local_species_rmsd_A"]["min"]}
        thresholds.add((audit["thresholds"]["near_soap_cosine"], audit["thresholds"]["near_rmsd_A"]))
    return {"minimums": minimums, "thresholds": thresholds}


def check_plot_data(ev: Evidence) -> None:
    data = json.loads(ev.input(PLOT_DATA).read_text())
    a, b = data["panel_a"], data["panel_b"]
    up_a, up_b = upstream_panel_a(ev), upstream_panel_b(ev)
    ev.check("panel a trained errors equal fig5 in-gap fixed values (6 dp)",
             a["trained_models"] == {f: up_a["trained"][f] for f in a["trained_models"]}
             and set(a["trained_models"]) == set(up_a["trained"]), {"json": a["trained_models"], "upstream": up_a["trained"]})
    ev.check("panel a foundation errors equal fig5 in-gap fixed values (6 dp)",
             a["foundation_models"] == up_a["foundation"], {"json": a["foundation_models"], "upstream": up_a["foundation"]})
    ev.check("panel a DFT reference equals fig5 reference", up_a["reference"] == {a["reference_barrier_eV"]},
             sorted(up_a["reference"]))
    ev.check("panel b minima equal overlap audits", b["minimums"] == up_b["minimums"], up_b["minimums"])
    ev.check("panel b thresholds equal overlap audits",
             up_b["thresholds"] == {(b["thresholds"]["soap_cosine"], b["thresholds"]["cr_local_rmsd_angstrom"])},
             sorted(up_b["thresholds"]))


def main() -> int:
    ev = Evidence(__file__, "FigS2")
    ev.input(GENERATOR)
    findings = lint_script(REPO / GENERATOR)
    ev.check(f"hardcode lint {GENERATOR}", not findings, findings or "no findings")
    check_plot_data(ev)

    stem = ev.output("grouped_retraining_overlap_palette_updated")
    run = subprocess.run([sys.executable, str(REPO / GENERATOR)], cwd=REPO, capture_output=True, text=True,
                         env={**os.environ, "GROUPED_OVERLAP_OUT": str(stem)})
    if ev.check("generator exits 0", run.returncode == 0, run.stderr[-2000:] or run.stdout.strip()):
        ev.input(FINAL_ASSET)
        ev.compare_pdf(stem.with_suffix(".pdf"), FINAL_ASSET)
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
