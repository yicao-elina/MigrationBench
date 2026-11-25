#!/usr/bin/env python3
"""Response Table R3 (tab:resp-a3-path-errors): seed-resolved fixed-image errors by path.

Legacy rows: mean +/- sample SD of absolute_error_ev over the three grouped seeds per family
(same data and arithmetic as SI Table S7). The canonical within-QL row is the Scratch-600K
fixed-path error from the Fig. 5 family statistics (same as SI Table S4).
Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import Evidence, fmt_pm  # noqa: E402

NEB_CSV = "paper/response_to_reviewers/reviewer_a2_grouped_retraining/data/grouped_neb_barrier_summary.csv"
REGISTRY = "paper/revision1/data_processed/path_registry.csv"
FAMILY_STATS = "paper/revision1/data_processed/current_release/fig5_family_statistics.csv"
MODEL_PREFIX = {"Scratch": "scratch_grouped", "FT-600K": "ft600k_omatft", "Scratch-5\\%": "scratch5_"}
SHORT_CLASS = {"gap-to-QL/into-QL": "gap-to-QL", "within-gap": "within-gap"}
LAYOUT = [("Q1", ["Scratch", "FT-600K"]), ("Q2", ["Scratch", "FT-600K"]), ("Q3", ["Scratch", "FT-600K"]),
          ("G1", ["Scratch", "FT-600K", "Scratch-5\\%"]), ("G2", ["Scratch", "FT-600K"])]
CANONICAL = ("Within-QL", "Scratch-600K", "migration_within_ql_fixed")   # label, family, column_id
DECIMALS = 2


def read(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    ev = Evidence(__file__, "tab:resp-a3-path-errors")
    registry = {r["name"]: r for r in read(ev.input(REGISTRY))}
    neb = read(ev.input(NEB_CSV))
    stats = read(ev.input(FAMILY_STATS))

    rows, seed_counts = [], {}
    for name, models in LAYOUT:
        entry = registry[name]
        for index, model in enumerate(models):
            errors = [float(r["absolute_error_ev"]) for r in neb
                      if r["pathway"] == entry["audit_id"] and r["model_id"].startswith(MODEL_PREFIX[model])]
            seed_counts[f"{name}/{model}"] = len(errors)
            label = f"{name} ({SHORT_CLASS[entry['class']]})" if index == 0 else ""
            rows.append([label, model, fmt_pm(statistics.mean(errors), statistics.stdev(errors), DECIMALS)])
    label, family, column = CANONICAL
    canonical = next(r for r in stats if r["family"] == family and r["column_id"] == column)
    seed_counts[f"{label}/{family}"] = int(canonical["n"])
    rows.append([label, family, fmt_pm(canonical["error_mean"], canonical["error_sd"], DECIMALS)])
    ev.check("three training seeds behind every row (caption)", set(seed_counts.values()) == {3}, seed_counts)
    ev.compare_table(rows, "response_to_reviewers.tex")
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
