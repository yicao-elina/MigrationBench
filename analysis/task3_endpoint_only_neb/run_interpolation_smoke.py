#!/usr/bin/env python3
"""Create and validate endpoint-only linear and IDPP initializations.

This is a geometry preflight only. It does not evaluate a model and does not
turn an unaccepted DFT path into a MACE result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import ase
from ase import Atoms
from ase.io import read, write
from ase.mep.neb import idpp_interpolate


def min_pair_distance(atoms: Atoms) -> float:
    d = atoms.get_all_distances(mic=True)
    d[np.diag_indices_from(d)] = np.inf
    return float(np.min(d))


def mic_linear(initial: Atoms, final: Atoms, n_images: int) -> list[Atoms]:
    cell = np.asarray(initial.cell.array)
    inv = np.linalg.inv(cell)
    f0, f1 = initial.positions @ inv, final.positions @ inv
    df = f1 - f0
    df -= np.round(df)
    images = []
    for i in range(n_images):
        t = i / (n_images - 1)
        a = initial.copy()
        a.set_cell(cell)
        a.set_pbc((True, True, True))
        a.positions = (f0 + t * df) @ cell
        images.append(a)
    return images


def max_mic_step(a: Atoms, b: Atoms) -> float:
    d = b.positions - a.positions
    frac = d @ np.linalg.inv(a.cell.array)
    frac -= np.round(frac)
    return float(np.max(np.linalg.norm(frac @ a.cell.array, axis=1)))


def profile(images: list[Atoms], method: str, path_id: str, dft_eligible: bool) -> dict:
    sep = [min_pair_distance(a) for a in images]
    steps = [max_mic_step(a, b) for a, b in zip(images[:-1], images[1:])]
    finite = all(np.isfinite(a.positions).all() and np.isfinite(a.cell.array).all() for a in images)
    no_overlap = bool(min(sep) > 0.70)
    return {
        "path_id": path_id, "initialization": method, "n_images": len(images),
        "nat": len(images[0]), "min_pair_distance_A": min(sep), "max_adjacent_step_A": max(steps),
        "path_length_A": sum(float(np.linalg.norm((b.positions - a.positions) @ np.linalg.inv(a.cell.array) - np.round((b.positions - a.positions) @ np.linalg.inv(a.cell.array)) @ a.cell.array, axis=1).max()) for a, b in zip(images[:-1], images[1:])),
        "finite_geometry": finite, "no_overlap_or_penetration": no_overlap,
        "geometry_smoke_pass": bool(finite and no_overlap),
        "dft_gate_eligible": dft_eligible,
        "scientific_success": False,
        "interpretation": "geometry_preflight_only; no MACE energies or forces",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("analysis/task3_endpoint_only_neb"))
    ap.add_argument("--n-images", type=int, default=10)
    ap.add_argument("--idpp-steps", type=int, default=2, help="Short preflight iteration count; formal runs need an explicit convergence budget")
    ap.add_argument("--idpp-path", default="2L-Cu__neb-gap-diffuse", help="Representative path for the bounded IDPP smoke")
    args = ap.parse_args()
    root = args.root.resolve()
    rows = []
    manifest = {r["path_id"]: r for r in csv.DictReader((root / "dft_gate_manifest.csv").open())}
    manifest_sha = hashlib.sha256((root / "dft_gate_manifest.csv").read_bytes()).hexdigest()
    environment = f"python={sys.version.split()[0]};ase={ase.__version__}"
    for path_id, rec in manifest.items():
        initial = read(root / "inputs" / f"{path_id}__initial.extxyz")
        final = read(root / "inputs" / f"{path_id}__final.extxyz")
        if len(initial) != len(final):
            raise ValueError(f"atom count mismatch for {path_id}")
        base = root / "interpolations" / path_id
        base.mkdir(parents=True, exist_ok=True)
        linear = mic_linear(initial, final, args.n_images)
        write(base / "linear.extxyz", linear, format="extxyz")
        linear_row = profile(linear, "linear", path_id, rec["formal_mace_eligible"] == "True")
        linear_row["output_sha256"] = hashlib.sha256((base / "linear.extxyz").read_bytes()).hexdigest()
        linear_row.update({"input_sha256": f"{rec['initial_endpoint_sha256']};{rec['final_endpoint_sha256']}", "manifest_sha256": manifest_sha, "model_sha256": "", "environment": environment, "failure_reason": rec["failure_reason"]})
        rows.append(linear_row)

        if path_id == args.idpp_path:
            idpp = mic_linear(initial, final, args.n_images)
            idpp_interpolate(idpp, traj=str(base / "idpp.traj"), log=str(base / "idpp.log"), fmax=0.10, mic=True, steps=args.idpp_steps)
            write(base / "idpp.extxyz", idpp, format="extxyz")
            idpp_row = profile(idpp, "idpp", path_id, rec["formal_mace_eligible"] == "True")
            idpp_row["output_sha256"] = hashlib.sha256((base / "idpp.extxyz").read_bytes()).hexdigest()
            idpp_row.update({"input_sha256": f"{rec['initial_endpoint_sha256']};{rec['final_endpoint_sha256']}", "manifest_sha256": manifest_sha, "model_sha256": "", "environment": environment, "failure_reason": rec["failure_reason"]})
            rows.append(idpp_row)
    with (root / "initialization_smoke.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(json.dumps({"rows": len(rows), "geometry_smoke_pass": sum(r["geometry_smoke_pass"] for r in rows), "formal_mace_rows": sum(r["dft_gate_eligible"] for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
