#!/usr/bin/env python3
"""Response Table R4 (tab:resp-a4-scope): scope of the evidence across systems.

The table is qualitative except for three data-backed cells: the non-Cr dopants with
fixed-path profiles (Fig. S3 source data), the number of dopant--host systems and the
number of endpoint relaxations (endpoint perturbation campaign summary).
Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import Evidence, fmt_int  # noqa: E402

ENDPOINTS = "paper/revision1/appendix/data/endpoint_level_summary.csv"
FIXED_PATH = ("release/revision1_submitted_20261110/items/si/FigS3_four_dopant_fixed_path/source/source_data/"
              "foundation_model_class_profiles_4x3_R03.tsv")
PRIMARY_DOPANT = "Cr"
PRIMARY_SYSTEM = "Cr@Sb2Te3"
NUMBER_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight",
                9: "Nine", 10: "Ten"}


def main() -> int:
    ev = Evidence(__file__, "tab:resp-a4-scope")
    endpoints = pd.read_csv(ev.input(ENDPOINTS))
    profiles = pd.read_csv(ev.input(FIXED_PATH), sep="\t")

    elements = list(dict.fromkeys(profiles["element"]))
    others = [e for e in elements if e != PRIMARY_DOPANT]
    systems = sorted(endpoints["system"].unique())
    relaxations = len(endpoints)

    ev.check("fixed-path profiles include the primary dopant plus the supplementary ones",
             PRIMARY_DOPANT in elements, {"elements": elements, "supplementary": others})
    ev.check("endpoint campaign covers the primary Cr@Sb2Te3 system", PRIMARY_SYSTEM in systems,
             {"systems": systems, "rows_per_system": endpoints["system"].value_counts().to_dict()})
    ev.check("every endpoint task_id is unique", endpoints["task_id"].is_unique,
             {"rows": relaxations, "unique_task_ids": int(endpoints["task_id"].nunique())})

    rows = [
        ["Cr-doped Sb$_2$Te$_3$", "Full grouped, path, and MD comparisons", "Primary-system conclusions"],
        [f"{'/'.join(others)} in Sb$_2$Te$_3$", "Supplied fixed-path profiles", "Qualitative pathway diagnostics"],
        [f"{NUMBER_WORDS[len(systems)]} dopant--host systems", f"{fmt_int(relaxations)} endpoint relaxations",
         "Local structural perturbation response"],
    ]
    ev.compare_table(rows, "response_to_reviewers.tex")
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
