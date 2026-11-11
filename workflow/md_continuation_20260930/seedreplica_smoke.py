#!/usr/bin/env python
"""Run one isolated 20 ps seed-replica MACE-MD smoke test.

The starting frame, momenta, cell, and species are read unchanged from the
common 200 ps state. Training seed and MD random seed are recorded separately.
This script writes only to its caller-provided output directory.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from mace.calculators import MACECalculator


class SmokeStop(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def temperature(atoms):
    p = atoms.get_momenta()
    m = atoms.get_masses()[:, None]
    return float(np.sum(p * p / m) / (3.0 * len(atoms) * units.kB))


def min_distance(atoms):
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    return float(np.nanmin(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--training-seed", type=int, required=True)
    ap.add_argument("--md-seed", type=int, default=1100)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--start-step", type=int, default=200000)
    ap.add_argument("--temperature-K", type=float, default=600.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    input_path, model_path = Path(args.input), Path(args.model)
    if not input_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"missing input/model: {input_path} {model_path}")

    atoms = read(str(input_path), index=-1, format="extxyz")
    if len(atoms) != 2050 or not all(atoms.pbc):
        raise RuntimeError("starting state failed 2050-atom/PBC check")
    if atoms.get_momenta() is None or not np.isfinite(atoms.get_momenta()).all():
        raise RuntimeError("starting state has no finite momenta")

    manifest = {
        "model_label": args.model_label,
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "training_seed": args.training_seed,
        "md_velocity_seed": args.md_seed,
        "start_step": args.start_step,
        "steps_requested": args.steps,
        "timestep_fs": 1.0,
        "temperature_target_K": args.temperature_K,
        "friction": 0.01,
        "write_interval_steps": 200,
        "natoms": len(atoms),
        "pbc": [bool(x) for x in atoms.pbc],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    atoms.calc = MACECalculator(
        model_paths=str(model_path), device=args.device, default_dtype="float64"
    )
    dyn = Langevin(
        atoms,
        timestep=1.0 * units.fs,
        temperature_K=args.temperature_K,
        friction=0.01,
        rng=np.random.default_rng(args.md_seed),
    )
    segment = out / "segment_smoke20ps.extxyz"
    rows = []
    state = {"last_step": args.start_step, "reason": "target_steps"}

    def record():
        local_step = int(dyn.nsteps)
        absolute_step = args.start_step + local_step
        forces = atoms.get_forces()
        energy = float(atoms.get_potential_energy())
        temp = temperature(atoms)
        dmin = min_distance(atoms) if local_step % 1000 == 0 else np.nan
        atoms.info.update({"step": absolute_step, "time_fs": float(absolute_step)})
        write(str(segment), atoms, format="extxyz", append=segment.exists())
        row = {
            "step": absolute_step,
            "time_ps": absolute_step * 0.001,
            "energy_eV": energy,
            "temperature_K": temp,
            "max_force_eV_A": float(np.abs(forces).max()),
            "min_distance_A": float(dmin),
        }
        rows.append(row)
        state["last_step"] = absolute_step
        bad = (
            not np.isfinite(atoms.positions).all()
            or not np.isfinite(forces).all()
            or not np.isfinite(energy)
            or not np.isfinite(temp)
            or temp > 5000.0
            or row["max_force_eV_A"] > 100.0
            or (np.isfinite(dmin) and dmin < 0.5)
        )
        if bad:
            state["reason"] = "scientific_threshold"
            raise SmokeStop

    record()
    try:
        dyn.attach(record, interval=200)
        dyn.run(args.steps)
    except SmokeStop:
        pass
    finally:
        write(str(out / "latest_restart.extxyz"), atoms, format="extxyz", append=False)

    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    result = {
        **state,
        "model_label": args.model_label,
        "training_seed": args.training_seed,
        "md_velocity_seed": args.md_seed,
        "smoke_pass": state["reason"] == "target_steps",
        "frames_written": len(rows),
    }
    (out / "status.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
