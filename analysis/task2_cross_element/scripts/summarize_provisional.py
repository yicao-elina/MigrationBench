#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = list(csv.DictReader((ROOT / "results/provisional_mace_profiles.csv").open()))
    audit = {r["path_id"]: r for r in json.loads((ROOT / "results/provisional_neb_iteration_audit.json").read_text())["records"]}
    groups = defaultdict(list)
    for row in rows:
        groups[(row["path_id"], row["variant"])].append(row)
    out_rows = []
    for (path_id, variant), group in sorted(groups.items()):
        rel = [float(r["relative_energy_eV"]) for r in group]
        forces = [float(r["max_force_eV_A"]) for r in group]
        d = audit[path_id]
        out_rows.append({"path_id": path_id, "element": group[0]["element"], "variant": variant, "n_images": len(group), "max_relative_energy_eV": max(rel), "mean_relative_energy_eV": sum(rel) / len(rel), "max_force_eV_A": max(forces), "all_mace_values_finite": all(r["finite"] == "True" for r in group), "last_two_completed_neb_iteration": d["last_completed_iteration"], "last_two_max_abs_dft_image_energy_delta_eV": d["max_abs_image_energy_delta_eV"], "last_two_mean_abs_dft_image_energy_delta_eV": d["mean_abs_image_energy_delta_eV"], "last_two_barrier_delta_eV": d["barrier_delta_eV"], "screen_delta_lt_0p1_eV": bool(d["max_abs_image_energy_delta_eV"] is not None and d["max_abs_image_energy_delta_eV"] < 0.1), "scientific_success": False, "interpretation": "provisional_non_equilibrium_snapshot_only"})
    out = ROOT / "results/provisional_mace_summary.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0])); writer.writeheader(); writer.writerows(out_rows)
    (ROOT / "results/provisional_mace_summary.json").write_text(json.dumps({"schema_version": "task2-provisional-summary-1.0", "rows": out_rows, "scientific_success_policy": "always_false"}, indent=2) + "\n")
    print(json.dumps({"rows": len(out_rows), "screen_delta_lt_0p1_eV": sum(x["screen_delta_lt_0p1_eV"] for x in out_rows), "scientific_success": 0}, indent=2))


if __name__ == "__main__":
    main()
