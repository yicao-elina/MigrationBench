#!/usr/bin/env python3
"""Create compact machine-readable and paper-table MACE diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "mace_results" / "mace_reconstructed.json"
OUT_CSV = ROOT / "mace_results" / "mace_summary.csv"
OUT_MD = ROOT / "mace_results" / "MACE_RESULTS.md"


def main() -> None:
    data = json.loads(SOURCE.read_text())
    rows = data["rows"]
    fields = ["case", "model", "energy_eV", "max_force_eV_A", "mean_force_eV_A", "input_sha256", "model_sha256", "scope"]
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    lines = [
        "# MACE reconstruction diagnostics",
        "",
        "These are four-model single-point evaluations on the same reconstructed 81-atom local cells used by the additive Phase-3 DFT diagnostic.",
        "They are not full-cell validation and absolute energy ranking across models is not scientifically interpreted.",
        "",
        "| Case | Foundation-OMAT max force | From-scratch max force | Naive-FT max force | Multi-T max force |",
        "|---|---:|---:|---:|---:|",
    ]
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[row["model"]] = row
    order = ["foundation-omat", "from-scratch", "naive-fine-tuning", "multi_T-fine-tuning"]
    for case in sorted(by_case):
        vals = [by_case[case][model]["max_force_eV_A"] for model in order]
        lines.append(f"| {case} | " + " | ".join(f"{value:.4f}" for value in vals) + " |")
    lines += ["", "The full per-atom force arrays, energy values, model hashes, and input hashes are in `mace_reconstructed.json` and `mace_summary.csv`.", ""]
    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
