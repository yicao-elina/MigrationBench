#!/usr/bin/env python3
"""Response Table R5 (tab:resp-a6-shap-fidelity): SHAP surrogate five-fold R^2 by family.

Mean and sample standard deviation (ddof=1) over training seeds of r2_5fold_cv in the
per-seed SHAP metrics table; seeds are listed from the same table. The submitted spreads
are typeset at 1-2 significant figures; ddof=0 would not reproduce them (e.g. FT-600K 2.3e-3).
Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import Evidence, fmt_sci  # noqa: E402

METRICS = "paper/revision1/data_processed/current_release/shap_seed_metrics.csv"
FAMILIES = ["FT-600K", "FT-MultiT", "Scratch-600K", "Scratch-MultiT"]  # table row order
MEAN_DECIMALS = 2
SD_DIGITS = 2  # generated spread keeps 2 significant figures; compared at the submitted precision


def main() -> int:
    ev = Evidence(__file__, "tab:resp-a6-shap-fidelity")
    metrics = pd.read_csv(ev.input(METRICS))
    ev.check("table families are exactly the families in the metrics file",
             sorted(FAMILIES) == sorted(metrics["model_family"].unique()),
             {"data": sorted(metrics["model_family"].unique())})
    rows = []
    for family in FAMILIES:
        group = metrics[metrics["model_family"] == family].sort_values("seed")
        r2 = group["r2_5fold_cv"]
        seeds = ", ".join(str(s) for s in group["seed"])
        rows.append([family, seeds, f"${r2.mean():.{MEAN_DECIMALS}f}\\pm{fmt_sci(r2.std(ddof=1), SD_DIGITS).strip('$')}$"])
    ev.compare_table(rows, "response_to_reviewers.tex")
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
