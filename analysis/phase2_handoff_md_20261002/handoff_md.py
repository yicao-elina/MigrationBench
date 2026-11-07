#!/usr/bin/env python3
"""Short-range, provenance-preserving MACE handoff MD.

The input frame is treated as immutable initial state: positions, momenta,
cell, PBC, species, and instantaneous kinetic temperature are recorded before
the calculator is attached.  No velocity resampling or coordinate wrapping is
performed by this driver.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np
from ase import units
from ase.geometry import find_mic
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.neighborlist import neighbor_list
from mace.calculators import MACECalculator


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(a: np.ndarray) -> bool:
    return bool(np.isfinite(a).all())


def pbc_displacement(atoms, reference: np.ndarray) -> np.ndarray:
    delta = atoms.positions - reference
    mic, _ = find_mic(delta, cell=atoms.cell, pbc=atoms.pbc)
    return np.asarray(mic)


def neighbor_metrics(atoms, dopant="Cr"):
    # A 6 A cutoff is sufficient for the requested short-range pair and RDF
    # diagnostics while avoiding an O(N^2) dense distance matrix.
    i, j, d = neighbor_list("ijd", atoms, 6.0, self_interaction=False)
    symbols = np.asarray(atoms.get_chemical_symbols())
    mask = i < j
    i, j, d = i[mask], j[mask], d[mask]
    if len(d):
        min_dist = float(np.min(d))
    else:
        min_dist = math.nan
    dop = np.where(symbols == dopant)[0]
    key = {}
    for host in ("Sb", "Te", "Cr"):
        vals = d[((symbols[i] == dopant) & (symbols[j] == host)) |
                 ((symbols[j] == dopant) & (symbols[i] == host))]
        key[f"min_{dopant}_{host}_A"] = float(np.min(vals)) if len(vals) else math.nan
        key[f"mean_{dopant}_{host}_A"] = float(np.mean(vals)) if len(vals) else math.nan
        cutoff = {"Sb": 3.6, "Te": 3.4, "Cr": 3.5}[host]
        key[f"coord_{dopant}_{host}_lt_{cutoff:.1f}A"] = int(np.count_nonzero(vals < cutoff))
    # A compact all-pair RDF histogram; normalized only by the shell-volume
    # convention, so it is suitable for comparing recovery to the initial frame.
    bins = np.arange(0.0, 6.0001, 0.1)
    hist, _ = np.histogram(d, bins=bins)
    return min_dist, key, hist.astype(int).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--start-step", type=int, required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--interval", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=None,
                    help="Langevin target K; default is 600 K")
    ap.add_argument("--friction", type=float, default=0.01)
    ap.add_argument("--timestep-fs", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    inp = Path(args.input).resolve()
    model = Path(args.model).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    atoms = read(inp, index=-1)
    if len(atoms) != 2050 or not bool(np.asarray(atoms.pbc).all()):
        raise RuntimeError(f"unexpected input: natoms={len(atoms)} pbc={atoms.pbc}")
    if not atoms.has("momenta"):
        raise RuntimeError("input does not contain momenta")
    initial_positions = atoms.positions.copy()
    initial_momenta = atoms.get_momenta().copy()
    initial_cell = atoms.cell.array.copy()
    initial_pbc = np.asarray(atoms.pbc, dtype=bool).copy()
    initial_temp = float(atoms.get_temperature())
    target_temp = float(args.temperature if args.temperature is not None else 600.0)
    input_sha = sha256(inp)
    model_sha = sha256(model)
    source_meta = {
        "input": str(inp), "input_sha256": input_sha,
        "model": str(model), "model_sha256": model_sha,
        "source_step": int(args.start_step), "natoms": len(atoms),
        "species_sha256": hashlib.sha256("".join(atoms.get_chemical_symbols()).encode()).hexdigest(),
        "positions_sha256": hashlib.sha256(initial_positions.tobytes()).hexdigest(),
        "momenta_sha256": hashlib.sha256(initial_momenta.tobytes()).hexdigest(),
        "cell_sha256": hashlib.sha256(initial_cell.tobytes()).hexdigest(),
        "cell_A": initial_cell.tolist(), "pbc": initial_pbc.tolist(),
        "initial_kinetic_temperature_K": initial_temp,
        "timestep_fs": float(args.timestep_fs), "target_temperature_K": target_temp,
        "friction_1_per_fs": float(args.friction), "device": args.device,
        "steps": int(args.steps), "write_interval_steps": int(args.interval),
        "python": sys.version, "platform": platform.platform(),
    }
    (out / "handoff_input.json").write_text(json.dumps(source_meta, indent=2, sort_keys=True) + "\n")
    traj = out / "trajectory.extxyz"
    metrics_path = out / "metrics.csv"
    fields = ["absolute_step", "time_ps", "potential_energy_eV", "kinetic_energy_eV",
              "total_energy_eV", "temperature_K", "max_force_eV_A", "max_velocity_A_ps",
              "max_momentum_amu_A_ps", "min_distance_A", "max_mic_displacement_A",
              "n_atoms_mic_displacement_gt5A", "max_unwrapped_displacement_A",
              "n_atoms_unwrapped_displacement_gt5A", "rdf_l1_vs_initial"]
    min0, pair0, rdf0 = neighbor_metrics(atoms)
    fields += list(pair0.keys())
    unwrapped = initial_positions.copy()
    previous = initial_positions.copy()
    previous_rdf = np.asarray(rdf0, dtype=float)
    atoms.calc = MACECalculator(model_paths=str(model), device=args.device, default_dtype="float64")
    dyn = Langevin(atoms, timestep=args.timestep_fs * units.fs,
                   temperature_K=target_temp, friction=args.friction)
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        def record():
            nonlocal unwrapped, previous, previous_rdf
            step = int(args.start_step + dyn.nsteps)
            positions = atoms.positions.copy()
            delta_mic, _ = find_mic(positions - previous, cell=atoms.cell, pbc=atoms.pbc)
            unwrapped = unwrapped + np.asarray(delta_mic)
            previous = positions
            mic_from_initial = pbc_displacement(atoms, initial_positions)
            unwrapped_disp = unwrapped - initial_positions
            min_dist, pair, rdf = neighbor_metrics(atoms)
            rdf_arr = np.asarray(rdf, dtype=float)
            denom = max(float(np.sum(previous_rdf)), 1.0)
            row = {
                "absolute_step": step, "time_ps": step / 1000.0,
                "potential_energy_eV": float(atoms.get_potential_energy()),
                "kinetic_energy_eV": float(atoms.get_kinetic_energy()),
                "total_energy_eV": float(atoms.get_total_energy()),
                "temperature_K": float(atoms.get_temperature()),
                "max_force_eV_A": float(np.abs(atoms.get_forces()).max()),
                "max_velocity_A_ps": float(np.linalg.norm(atoms.get_velocities(), axis=1).max() * 1000.0),
                "max_momentum_amu_A_ps": float(np.linalg.norm(atoms.get_momenta(), axis=1).max()),
                "min_distance_A": min_dist,
                "max_mic_displacement_A": float(np.linalg.norm(mic_from_initial, axis=1).max()),
                "n_atoms_mic_displacement_gt5A": int(np.count_nonzero(np.linalg.norm(mic_from_initial, axis=1) > 5.0)),
                "max_unwrapped_displacement_A": float(np.linalg.norm(unwrapped_disp, axis=1).max()),
                "n_atoms_unwrapped_displacement_gt5A": int(np.count_nonzero(np.linalg.norm(unwrapped_disp, axis=1) > 5.0)),
                "rdf_l1_vs_initial": float(np.sum(np.abs(rdf_arr - previous_rdf)) / denom),
            }
            row.update(pair)
            required_finite = ["potential_energy_eV", "kinetic_energy_eV", "total_energy_eV",
                               "temperature_K", "max_force_eV_A", "max_velocity_A_ps",
                               "max_momentum_amu_A_ps", "min_distance_A", "max_mic_displacement_A",
                               "max_unwrapped_displacement_A", "rdf_l1_vs_initial"]
            # A NaN in a named pair metric means that no such pair occurs
            # within the 6 A neighbor-list cutoff; it is not a non-finite MD
            # state and is retained as an explicit missing pair observation.
            if not all(np.isfinite(float(row[k])) for k in required_finite):
                raise FloatingPointError(f"non-finite metric at step {step}: {row}")
            writer.writerow(row); f.flush()
            atoms.info.update({"absolute_step": step, "time_fs": float(step), "handoff_source_step": args.start_step})
            write(traj, atoms, format="extxyz", append=traj.exists())

        # Record the exact initial state before the first integration step.
        record()
        dyn.attach(record, interval=args.interval)
        dyn.run(args.steps)
    (out / "completion.json").write_text(json.dumps({
        "completed": True, "last_step": int(args.start_step + dyn.nsteps),
        "trajectory": str(traj), "metrics": str(metrics_path),
        "handoff_input": str(out / "handoff_input.json")
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
