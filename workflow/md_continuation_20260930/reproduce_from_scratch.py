#!/usr/bin/env python
"""Short, isolated reproduction of the from-scratch instability.

Starts from the exact stored 200 ps endpoint and never touches the production
run directory.  Metrics are written before the process exits on a threshold.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from mace.calculators import MACECalculator


def temp(atoms):
    p = atoms.get_momenta(); m = atoms.get_masses()[:, None]
    return float(2.0 * (0.5 * np.sum(p * p / m)) / (3.0 * len(atoms) * units.kB))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    atoms = read(args.input, index=-1)
    if len(atoms) != 2050 or not all(atoms.pbc):
        raise RuntimeError("input structure failed 2050/PBC check")
    atoms.calc = MACECalculator(model_paths=args.model, device=args.device, default_dtype="float64")
    dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=600.0,
                   friction=0.01, rng=np.random.default_rng(args.seed))
    rows = []
    stop = {"reason": "target_steps", "step": 0}
    def record():
        step = int(dyn.nsteps)
        f = atoms.get_forces()
        d = atoms.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        e = atoms.get_potential_energy()
        row = {"step": step, "time_ps": step * 0.001, "energy_eV": float(e),
               "temperature_K": temp(atoms), "max_force_eV_A": float(np.abs(f).max()),
               "min_distance_A": float(d.min())}
        rows.append(row)
        if row["temperature_K"] > 5000 or row["max_force_eV_A"] > 100 or row["min_distance_A"] < 0.5:
            stop.update(reason="instability_threshold", step=step)
            raise SystemExit(99)
    dyn.attach(record, interval=200)
    try:
        dyn.run(args.steps)
    except SystemExit:
        pass
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (out / "result.json").write_text(json.dumps({"seed": args.seed, "steps_requested": args.steps,
        "stop": stop, "input": args.input, "model": args.model}, indent=2) + "\n")
    write(out / "last.extxyz", atoms, format="extxyz")
    print(json.dumps(stop))


if __name__ == "__main__":
    main()
