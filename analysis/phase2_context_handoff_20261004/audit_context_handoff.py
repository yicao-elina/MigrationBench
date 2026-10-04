#!/usr/bin/env python3
"""Audit the additive 460000/462200 context handoff outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from ase.io import read


ROOT = Path(__file__).resolve().parent
MODELS = ["foundation-omat", "from-scratch", "naive-fine-tuning", "multi_T-fine-tuning"]
SNAPS = [460000, 462200]
SOURCE = {
    snap: ROOT.parent / "phase2_handoff_md_20261002" / "remote_artifacts" /
    "snapshots" / "from_scratch_source" / f"snapshot_{snap}.extxyz"
    for snap in SNAPS
}
FINITE_FIELDS = [
    "potential_energy_eV", "kinetic_energy_eV", "total_energy_eV", "temperature_K",
    "max_force_eV_A", "max_velocity_A_ps", "max_momentum_amu_A_ps", "min_distance_A",
    "max_unwrapped_displacement_A", "rdf_l1_vs_initial",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def first_gate(rows: list[dict]) -> tuple[dict, dict | None]:
    predicates = {
        "force_gt_100_eV_A": lambda r: float(r["max_force_eV_A"]) >= 100.0,
        "velocity_gt_1000_A_ps": lambda r: float(r["max_velocity_A_ps"]) >= 1000.0,
        "min_distance_lt_1p5_A": lambda r: float(r["min_distance_A"]) < 1.5,
        "temperature_gt_2000_K": lambda r: float(r["temperature_K"]) >= 2000.0,
        "unwrapped_displacement_gt_50_A": lambda r: float(r["max_unwrapped_displacement_A"]) >= 50.0,
        "n_atoms_gt5_A_ge_100": lambda r: float(r["n_atoms_unwrapped_displacement_gt5A"]) >= 100.0,
    }
    crossings = {}
    for name, predicate in predicates.items():
        row = next((r for r in rows if predicate(r)), None)
        crossings[name] = None if row is None else {
            "step": int(row["absolute_step"]), "time_ps": float(row["time_ps"])
        }
    energy = np.asarray([float(r["total_energy_eV"]) for r in rows])
    span_row = next((rows[i] for i in range(len(rows))
                     if np.max(energy[:i + 1]) - np.min(energy[:i + 1]) >= 250.0), None)
    crossings["total_energy_span_ge_250_eV"] = None if span_row is None else {
        "step": int(span_row["absolute_step"]), "time_ps": float(span_row["time_ps"])
    }
    available = [(v["step"], k, v["time_ps"]) for k, v in crossings.items() if v is not None]
    if not available:
        return crossings, None
    step, gate, time_ps = min(available)
    return crossings, {"step": step, "time_ps": time_ps, "gate": gate}


def verify_first_frame(path: Path, source_path: Path, handoff: dict) -> dict:
    """Check serialized first-frame state against the exact local source frame."""
    first = read(path, index=0)
    source = read(source_path, index=-1)
    step = int(first.info.get("absolute_step", first.info.get("step", -1)))
    return {
        "source_file_sha256_matches_handoff": sha256(source_path) == handoff["input_sha256"],
        "natoms": len(first),
        "natoms_match": len(first) == int(handoff["natoms"]) == len(source),
        "species_match": first.get_chemical_symbols() == source.get_chemical_symbols(),
        "positions_match_tol_1e-6_A": bool(np.allclose(first.positions, source.positions, atol=1e-6, rtol=0.0)),
        "momenta_match_tol_1e-6": bool(np.allclose(first.get_momenta(), source.get_momenta(), atol=1e-6, rtol=0.0)),
        "cell_match_tol_1e-8_A": bool(np.allclose(first.cell.array, source.cell.array, atol=1e-8, rtol=0.0)),
        "pbc_match": bool(np.array_equal(first.pbc, source.pbc)),
        "first_step": step,
        "first_step_matches_source_step": step == int(handoff["source_step"]),
    }


def audit_run(path: Path, snap: int) -> dict:
    metrics = path / "metrics.csv"
    completion = path / "completion.json"
    handoff_file = path / "handoff_input.json"
    first_frame = path / "first_frame.extxyz"
    if not all(p.exists() for p in (metrics, completion, handoff_file, first_frame)):
        return {"status": "MISSING_ARTIFACT", "path": str(path)}
    handoff = json.loads(handoff_file.read_text())
    completion_data = json.loads(completion.read_text())
    rows = list(csv.DictReader(metrics.open()))
    finite = all(math.isfinite(float(row[key])) for row in rows for key in FINITE_FIELDS)
    values = {key: np.asarray([float(row[key]) for row in rows]) for key in FINITE_FIELDS}
    energy_span = float(np.max(values["total_energy_eV"]) - np.min(values["total_energy_eV"]))
    gates_ok = finite and all([
        np.min(values["min_distance_A"]) >= 1.5,
        np.max(values["max_velocity_A_ps"]) < 1000.0,
        np.max(values["max_force_eV_A"]) < 100.0,
        np.max(values["max_unwrapped_displacement_A"]) < 50.0,
        np.max([float(row["n_atoms_unwrapped_displacement_gt5A"]) for row in rows]) < 100.0,
        np.max(values["temperature_K"]) < 2000.0,
        energy_span < 250.0,
    ])
    classification = "runaway_or_nonfinite" if not gates_ok else (
        "bounded_with_structural_shift"
        if np.max(values["max_unwrapped_displacement_A"]) > 5.0
        or np.max([float(row["n_atoms_unwrapped_displacement_gt5A"]) for row in rows]) > 0
        else "stable_short_handoff"
    )
    crossings, first = first_gate(rows)
    frame = verify_first_frame(first_frame, SOURCE[snap], handoff)
    frame["all_initial_state_checks"] = all([
        frame["source_file_sha256_matches_handoff"], frame["natoms_match"], frame["species_match"],
        frame["positions_match_tol_1e-6_A"], frame["momenta_match_tol_1e-6"],
        frame["cell_match_tol_1e-8_A"], frame["pbc_match"], frame["first_step_matches_source_step"],
    ])
    return {
        "status": "COMPLETE", "snapshot": snap,
        "model": path.name.removeprefix("model_"), "frames": len(rows), "expected_frames": 202,
        "frame_count_ok": len(rows) == 202, "start_step": int(rows[0]["absolute_step"]),
        "end_step": int(rows[-1]["absolute_step"]), "expected_end_step": snap + 2000,
        "completion_completed": bool(completion_data.get("completed")),
        "completion_last_step": completion_data.get("last_step"),
        "completion_ok": bool(completion_data.get("completed")) and completion_data.get("last_step") == snap + 2000,
        "finite_required_metrics": finite, "classification": classification,
        "first_gate": first, "gate_crossings": crossings, "total_energy_span_eV": energy_span,
        "input_sha256": handoff["input_sha256"], "model_sha256": handoff["model_sha256"],
        "positions_sha256": handoff["positions_sha256"], "momenta_sha256": handoff["momenta_sha256"],
        "cell_sha256": handoff["cell_sha256"], "metrics_sha256": sha256(metrics), "first_frame": frame,
    }


def main() -> None:
    rows = [audit_run(ROOT / "runs" / f"snapshot_{snap}" / f"model_{model}", snap)
            for snap in SNAPS for model in MODELS]
    frame_rows = []
    for snap in SNAPS:
        for model in MODELS:
            metrics = ROOT / "runs" / f"snapshot_{snap}" / f"model_{model}" / "metrics.csv"
            with metrics.open() as fh:
                frame_rows.extend({"snapshot": snap, "model": model, **row}
                                   for row in csv.DictReader(fh))
    summary = {
        "runs": len(rows), "all_complete": all(row.get("status") == "COMPLETE" for row in rows),
        "all_frame_counts_ok": all(row.get("frame_count_ok", False) for row in rows),
        "all_completions_ok": all(row.get("completion_ok", False) for row in rows),
        "all_initial_state_checks_ok": all(row.get("first_frame", {}).get("all_initial_state_checks", False) for row in rows),
        "input_hash_consistency_by_snapshot": {
            str(snap): {field: len({row[field] for row in rows if row.get("snapshot") == snap}) == 1
                        for field in ("input_sha256", "positions_sha256", "momenta_sha256", "cell_sha256")}
            for snap in SNAPS
        },
        "classifications": {label: sum(row.get("classification") == label for row in rows)
                            for label in sorted({row.get("classification") for row in rows})},
        "runs_detail": rows,
    }
    (ROOT / "audit_results.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if frame_rows:
        with (ROOT / "frame_aligned_metrics.csv").open("w", newline="") as fh:
            fields = ["snapshot", "model"] + [key for key in frame_rows[0] if key not in {"snapshot", "model"}]
            writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(frame_rows)
    with (ROOT / "audit_results.csv").open("w", newline="") as fh:
        fields = ["snapshot", "model", "frames", "start_step", "end_step", "classification",
                  "finite_required_metrics", "frame_count_ok", "completion_ok", "all_initial_state_checks",
                  "total_energy_span_eV", "first_gate_step", "first_gate"]
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader()
        for row in rows:
            first = row.get("first_gate") or {}
            writer.writerow({"snapshot": row.get("snapshot"), "model": row.get("model"), "frames": row.get("frames"),
                             "start_step": row.get("start_step"), "end_step": row.get("end_step"),
                             "classification": row.get("classification"), "finite_required_metrics": row.get("finite_required_metrics"),
                             "frame_count_ok": row.get("frame_count_ok"), "completion_ok": row.get("completion_ok"),
                             "all_initial_state_checks": row.get("first_frame", {}).get("all_initial_state_checks"),
                             "total_energy_span_eV": row.get("total_energy_span_eV"),
                             "first_gate_step": first.get("step"), "first_gate": first.get("gate")})
    print(json.dumps({key: summary[key] for key in ("runs", "all_complete", "all_frame_counts_ok",
                                                     "all_completions_ok", "all_initial_state_checks_ok",
                                                     "classifications")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
