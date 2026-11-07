#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np


def f(rows, key):
    return np.asarray([float(r[key]) for r in rows], dtype=float)


def classify(rows):
    checks = {}
    numeric = ["potential_energy_eV", "kinetic_energy_eV", "total_energy_eV", "temperature_K",
               "max_force_eV_A", "max_velocity_A_ps", "max_momentum_amu_A_ps", "min_distance_A"]
    checks["finite"] = all(np.isfinite(f(rows, k)).all() for k in numeric)
    checks["no_close_contact_lt_1p5A"] = bool(np.nanmin(f(rows, "min_distance_A")) >= 1.5)
    # Thermal velocities of Te at 600 K can be several hundred A/ps.  The
    # thresholds below therefore target catastrophic growth, not ordinary
    # thermal motion or a single migration event.
    checks["no_catastrophic_velocity_gt_1000A_ps"] = bool(np.nanmax(f(rows, "max_velocity_A_ps")) < 1000.0)
    checks["no_catastrophic_force_gt_100eV_A"] = bool(np.nanmax(f(rows, "max_force_eV_A")) < 100.0)
    checks["no_catastrophic_unwrapped_gt_50A"] = bool(np.nanmax(f(rows, "max_unwrapped_displacement_A")) < 50.0)
    checks["no_catastrophic_gt5A_count_gt_100"] = bool(np.nanmax(f(rows, "n_atoms_unwrapped_displacement_gt5A")) < 100)
    e = f(rows, "total_energy_eV")
    checks["bounded_total_energy"] = bool(np.nanmax(e) - np.nanmin(e) < 250.0)
    t = f(rows, "temperature_K")
    checks["temperature_finite_and_bounded"] = bool(np.nanmin(t) > 0 and np.nanmax(t) < 2000)
    catastrophic = all(checks.values())
    if not catastrophic:
        label = "runaway_or_nonfinite"
    elif np.nanmax(f(rows, "max_unwrapped_displacement_A")) > 5 or np.nanmax(f(rows, "n_atoms_unwrapped_displacement_gt5A")) > 0:
        label = "bounded_with_structural_shift"
    else:
        label = "stable_short_handoff"
    return checks, label


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); args = ap.parse_args()
    root = Path(args.root); summaries = []
    for p in sorted(root.glob("snapshot_*/model_*/metrics.csv")):
        with p.open() as fh: rows = list(csv.DictReader(fh))
        checks, label = classify(rows)
        inp = json.loads((p.parent / "handoff_input.json").read_text())
        energy = f(rows, "total_energy_eV"); temp = f(rows, "temperature_K")
        summary = {
            "snapshot": p.parent.parent.name, "model": p.parent.name.removeprefix("model_"),
            "metrics": str(p), "n_frames": len(rows), "start_step": int(rows[0]["absolute_step"]),
            "end_step": int(rows[-1]["absolute_step"]), "duration_ps": float(float(rows[-1]["time_ps"]) - float(rows[0]["time_ps"])),
            "initial_temperature_K": float(temp[0]), "temperature_mean_K": float(np.mean(temp)),
            "temperature_min_K": float(np.min(temp)), "temperature_max_K": float(np.max(temp)),
            "total_energy_min_eV": float(np.min(energy)), "total_energy_max_eV": float(np.max(energy)),
            "total_energy_span_eV": float(np.max(energy) - np.min(energy)),
            "max_force_eV_A": float(np.max(f(rows, "max_force_eV_A"))),
            "max_velocity_A_ps": float(np.max(f(rows, "max_velocity_A_ps"))),
            "min_distance_A": float(np.min(f(rows, "min_distance_A"))),
            "max_unwrapped_displacement_A": float(np.max(f(rows, "max_unwrapped_displacement_A"))),
            "max_atoms_unwrapped_gt5A": int(np.max(f(rows, "n_atoms_unwrapped_displacement_gt5A"))),
            "input_sha256": inp["input_sha256"], "model_sha256": inp["model_sha256"],
            "source_step": inp["source_step"], "initial_positions_sha256": inp["positions_sha256"],
            "initial_momenta_sha256": inp["momenta_sha256"], "initial_cell_sha256": inp["cell_sha256"],
            "checks": checks, "classification": label,
        }
        summaries.append(summary)
    (root / "stability_summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    with (root / "stability_summary.csv").open("w", newline="") as fh:
        cols = ["snapshot", "model", "n_frames", "start_step", "end_step", "duration_ps", "classification",
                "temperature_mean_K", "temperature_min_K", "temperature_max_K", "total_energy_span_eV",
                "max_force_eV_A", "max_velocity_A_ps", "min_distance_A", "max_unwrapped_displacement_A",
                "max_atoms_unwrapped_gt5A", "source_step", "initial_positions_sha256", "initial_momenta_sha256"]
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for s in summaries: w.writerow({k: s[k] for k in cols})
    print(json.dumps({"runs": len(summaries), "classifications": {x: sum(s["classification"] == x for s in summaries) for x in sorted(set(s["classification"] for s in summaries))}}, indent=2))


if __name__ == "__main__": main()
