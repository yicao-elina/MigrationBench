#!/usr/bin/env python3
"""Audit accepted Phase-2 handoff metrics without touching raw artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def num(rows, key):
    return np.asarray([float(r[key]) for r in rows], dtype=float)


def first_gate(rows):
    gates = {
        "force_gt_100_eV_A": lambda r: float(r["max_force_eV_A"]) >= 100.0,
        "velocity_gt_1000_A_ps": lambda r: float(r["max_velocity_A_ps"]) >= 1000.0,
        "min_distance_lt_1p5_A": lambda r: float(r["min_distance_A"]) < 1.5,
        "temperature_gt_2000_K": lambda r: float(r["temperature_K"]) >= 2000.0,
        "unwrapped_displacement_gt_50_A": lambda r: float(r["max_unwrapped_displacement_A"]) >= 50.0,
        "n_atoms_gt5_A_ge_100": lambda r: float(r["n_atoms_unwrapped_displacement_gt5A"]) >= 100.0,
    }
    crossings = {}
    for name, predicate in gates.items():
        row = next((r for r in rows if predicate(r)), None)
        crossings[name] = None if row is None else {
            "step": int(row["absolute_step"]), "time_ps": float(row["time_ps"])
        }
    emin = emax = float(rows[0]["total_energy_eV"])
    energy = None
    for row in rows:
        value = float(row["total_energy_eV"])
        emin, emax = min(emin, value), max(emax, value)
        if energy is None and emax - emin >= 250.0:
            energy = {"step": int(row["absolute_step"]), "time_ps": float(row["time_ps"])}
    crossings["total_energy_span_ge_250_eV"] = energy
    available = [(v["step"], k, v["time_ps"]) for k, v in crossings.items() if v is not None]
    if not available:
        return crossings, None
    step, gate, time_ps = min(available)
    return crossings, {"step": step, "time_ps": time_ps, "gate": gate}


def classify(rows):
    finite_fields = ["potential_energy_eV", "kinetic_energy_eV", "total_energy_eV", "temperature_K",
                     "max_force_eV_A", "max_velocity_A_ps", "max_momentum_amu_A_ps", "min_distance_A",
                     "max_unwrapped_displacement_A", "rdf_l1_vs_initial"]
    finite = all(np.isfinite(num(rows, key)).all() for key in finite_fields)
    gate_ok = finite and all([
        np.nanmin(num(rows, "min_distance_A")) >= 1.5,
        np.nanmax(num(rows, "max_velocity_A_ps")) < 1000.0,
        np.nanmax(num(rows, "max_force_eV_A")) < 100.0,
        np.nanmax(num(rows, "max_unwrapped_displacement_A")) < 50.0,
        np.nanmax(num(rows, "n_atoms_unwrapped_displacement_gt5A")) < 100.0,
        np.nanmax(num(rows, "temperature_K")) < 2000.0,
        np.nanmax(num(rows, "total_energy_eV")) - np.nanmin(num(rows, "total_energy_eV")) < 250.0,
    ])
    if not gate_ok:
        label = "runaway_or_nonfinite"
    elif np.nanmax(num(rows, "max_unwrapped_displacement_A")) > 5.0 or np.nanmax(num(rows, "n_atoms_unwrapped_displacement_gt5A")) > 0:
        label = "bounded_with_structural_shift"
    else:
        label = "stable_short_handoff"
    return finite, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root, output = Path(args.root), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    frame_rows = []
    for metrics in sorted(root.glob("snapshot_*/model_*/metrics.csv")):
        rows = list(csv.DictReader(metrics.open()))
        snapshot = metrics.parent.parent.name.removeprefix("snapshot_")
        model = metrics.parent.name.removeprefix("model_")
        crossings, first = first_gate(rows)
        finite, label = classify(rows)
        source = json.loads((metrics.parent / "handoff_input.json").read_text())
        summary = {
            "snapshot": snapshot, "model": model, "n_frames": len(rows),
            "start_step": int(rows[0]["absolute_step"]), "end_step": int(rows[-1]["absolute_step"]),
            "classification": label, "finite_required_metrics": finite,
            "first_divergence": first, "gate_crossings": crossings,
            "max_force_eV_A": float(np.nanmax(num(rows, "max_force_eV_A"))),
            "max_velocity_A_ps": float(np.nanmax(num(rows, "max_velocity_A_ps"))),
            "min_distance_A": float(np.nanmin(num(rows, "min_distance_A"))),
            "max_temperature_K": float(np.nanmax(num(rows, "temperature_K"))),
            "max_unwrapped_displacement_A": float(np.nanmax(num(rows, "max_unwrapped_displacement_A"))),
            "max_atoms_unwrapped_displacement_gt5A": int(np.nanmax(num(rows, "n_atoms_unwrapped_displacement_gt5A"))),
            "total_energy_span_eV": float(np.nanmax(num(rows, "total_energy_eV")) - np.nanmin(num(rows, "total_energy_eV"))),
            "input_sha256": source["input_sha256"], "positions_sha256": source["positions_sha256"],
            "momenta_sha256": source["momenta_sha256"], "cell_sha256": source["cell_sha256"],
            "model_sha256": source["model_sha256"],
        }
        summaries.append(summary)
        for row in rows:
            frame = {"snapshot": snapshot, "model": model, **row}
            frame_rows.append(frame)
    (output / "first_divergence_by_run.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    cols = ["snapshot", "model", "n_frames", "start_step", "end_step", "classification", "finite_required_metrics",
            "first_divergence_step", "first_divergence_time_ps", "first_divergence_gate", "max_force_eV_A",
            "max_velocity_A_ps", "min_distance_A", "max_temperature_K", "max_unwrapped_displacement_A",
            "max_atoms_unwrapped_displacement_gt5A", "total_energy_span_eV", "input_sha256", "positions_sha256",
            "momenta_sha256", "cell_sha256", "model_sha256"]
    with (output / "first_divergence_by_run.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for s in summaries:
            first = s["first_divergence"] or {}
            w.writerow({**{k: s.get(k, "") for k in cols},
                        "first_divergence_step": first.get("step", ""),
                        "first_divergence_time_ps": first.get("time_ps", ""),
                        "first_divergence_gate": first.get("gate", "")})
    frame_cols = list(frame_rows[0].keys())
    with (output / "frame_aligned_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=frame_cols); w.writeheader(); w.writerows(frame_rows)
    # All 202 rows are retained; the explicit row count prevents a plot from
    # silently becoming an endpoint-only summary.
    report = {
        "runs": len(summaries), "frames": len(frame_rows),
        "classifications": {label: sum(s["classification"] == label for s in summaries)
                             for label in sorted({s["classification"] for s in summaries})},
        "first_divergence_steps": {f"{s['snapshot']}/{s['model']}": (s["first_divergence"] or {}).get("step")
                                   for s in summaries},
        "input_hash_consistency_by_snapshot": {},
    }
    for snap in sorted({s["snapshot"] for s in summaries}):
        group = [s for s in summaries if s["snapshot"] == snap]
        report["input_hash_consistency_by_snapshot"][snap] = {
            field: len({s[field] for s in group}) == 1
            for field in ["input_sha256", "positions_sha256", "momenta_sha256", "cell_sha256"]
        }
    (output / "audit_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
