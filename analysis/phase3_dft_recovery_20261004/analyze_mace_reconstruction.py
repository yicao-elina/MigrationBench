#!/usr/bin/env python3
"""Summarize within-model energy and force changes across reconstructed cases."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "mace_results" / "mace_reconstructed.json"
OUT = ROOT / "mace_results" / "mace_within_model_comparison.csv"


def main() -> None:
    rows = json.loads(DATA.read_text())["rows"]
    normal = {r["model"]: r for r in rows if r["case"] == "normal_461800_initial"}
    out = []
    for row in rows:
        ref = normal[row["model"]]
        item = {
            "case": row["case"],
            "model": row["model"],
            "delta_energy_eV_within_model": row["energy_eV"] - ref["energy_eV"],
            "max_force_eV_A": row["max_force_eV_A"],
            "normal_max_force_eV_A": ref["max_force_eV_A"],
            "max_force_ratio_to_normal": row["max_force_eV_A"] / ref["max_force_eV_A"],
            "input_sha256": row["input_sha256"],
            "model_sha256": row["model_sha256"],
            "scope": row["scope"],
        }
        out.append(item)
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    for item in out:
        print(item["case"], item["model"], f"dE={item['delta_energy_eV_within_model']:.6f}", f"F_ratio={item['max_force_ratio_to_normal']:.3f}")


if __name__ == "__main__":
    main()
