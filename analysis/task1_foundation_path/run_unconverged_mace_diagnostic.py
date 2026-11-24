#!/usr/bin/env python3
"""MACE diagnostics on selected, explicitly unconverged DFT NEB snapshots.

This is intentionally a separate diagnostic tier. It never labels the DFT
snapshots as converged references and never compares absolute total energies
between models.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import socket
import sys
import traceback
from pathlib import Path

import numpy as np
from ase import Atoms, __version__ as ase_version
from ase.calculators.calculator import Calculator
from ase.io import read, write
from ase.optimize import FIRE
from ase.io.trajectory import Trajectory


ROOT = Path("/scratch4/pclancy3/joshitha/vdW-sratch/sb2te3")
MODEL = Path("/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/X-FORCE/xforce/models/Foundation-MACE-matpes-pbe-omat-ft.model")
PATHS = {
    "2L-Cu__neb-QLs": ROOT / "2L-Cu/NEB/neb-QLs",
    "2L-Cu__neb-gap-diffuse": ROOT / "2L-Cu/NEB/neb-gap-diffuse",
    "2L-Cu__neb-gap2QL": ROOT / "2L-Cu/NEB/neb-gap2QL",
    "2L-Li__neb-QLs": ROOT / "2L-Li/NEB/neb-QLs",
    "2L-Li__neb-gap-diffuse": ROOT / "2L-Li/NEB/neb-gap-diffuse",
    "2L-Li__neb-gap2Ql": ROOT / "2L-Li/NEB/neb-gap2Ql",
    "2L-Na__neb-gap-diffuse": ROOT / "2L-Na/NEB/neb-gap-diffuse",
    "2L-Na__neb-gap2QL": ROOT / "2L-Na/NEB/neb-gap2QL",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def cell_from_input(path: Path) -> np.ndarray:
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().upper().startswith("CELL_PARAMETERS")), None)
    if start is None:
        raise ValueError(f"no CELL_PARAMETERS in {path}")
    cell = []
    for line in lines[start + 1:]:
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            cell.append([float(x.replace("D", "E").replace("d", "e")) for x in fields[:3]])
        except ValueError:
            continue
        if len(cell) == 3:
            return np.array(cell, dtype=float)
    raise ValueError(f"incomplete CELL_PARAMETERS in {path}")


def xyz_path(directory: Path) -> Path:
    hits = sorted(directory.glob("*.xyz"))
    if not hits:
        raise FileNotFoundError(f"no xyz trajectory in {directory}")
    return hits[0]


def dopant_indices(atoms: Atoms) -> np.ndarray:
    return np.array([i for i, s in enumerate(atoms.get_chemical_symbols()) if s not in {"Sb", "Te"}], dtype=int)


def finite_geometry(atoms: Atoms, force: np.ndarray | None = None) -> tuple[bool, float, float]:
    pos_ok = bool(np.isfinite(atoms.positions).all())
    d = atoms.get_all_distances(mic=True) if pos_ok else np.full((len(atoms), len(atoms)), np.nan)
    d[d == 0] = np.nan
    min_d = float(np.nanmin(d)) if np.isfinite(d).any() else np.nan
    fmax = float(np.max(np.linalg.norm(force, axis=1))) if force is not None and np.isfinite(force).all() else np.nan
    return pos_ok and np.isfinite(min_d), min_d, fmax


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--paths", nargs="+", choices=sorted(PATHS), default=sorted(PATHS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default=str(MODEL))
    ap.add_argument("--fmax", type=float, default=0.03)
    ap.add_argument("--max-steps", type=int, default=500)
    args = ap.parse_args()
    out = Path(args.out).resolve(); (out / "trajectories").mkdir(parents=True, exist_ok=True); (out / "final_structures").mkdir(parents=True, exist_ok=True)
    from mace.calculators import MACECalculator
    model_path = Path(args.model)
    calc = MACECalculator(model_paths=str(model_path), device=args.device, default_dtype="float64")
    runtime = {"python": sys.version.split()[0], "ase": ase_version, "torch": __import__("torch").__version__, "mace_torch": getattr(__import__("mace"), "__version__", "unknown"), "hostname": socket.gethostname(), "platform": platform.platform(), "model_path": str(model_path), "model_sha256": sha256(model_path), "diagnostic_status": "unconverged_dft_snapshots"}
    (out / "runtime_manifest.json").write_text(json.dumps(runtime, indent=2, sort_keys=True))
    sp_rows, relax_rows, failures = [], [], []
    perturbations = {"base_000": np.array([0.0, 0.0, 0.0]), "dx020": np.array([0.20, 0.0, 0.0]), "tilt020": np.array([-0.10, 0.14, 0.10])}
    for path_id in args.paths:
        directory = PATHS[path_id]; cell = cell_from_input(directory / "neb.in"); frames = read(xyz_path(directory), index=":")
        if not isinstance(frames, list): frames = [frames]
        energies = []
        for image_index, atoms0 in enumerate(frames):
            atoms = atoms0.copy(); atoms.set_cell(cell); atoms.set_pbc([True, True, True]); atoms.calc = calc
            row = {"model_id": "legacy_foundation", "model_label": "Foundation", "model_sha256": runtime["model_sha256"], "path_id": path_id, "image_index": image_index, "n_atoms": len(atoms), "input_trajectory": str(xyz_path(directory)), "status": "success", "scientific_success": False, "diagnostic_only": True, "failure_reason": "unconverged DFT snapshot; not a reference"}
            try:
                e = float(atoms.get_potential_energy()); f = np.asarray(atoms.get_forces(), float); ok, mind, fmax = finite_geometry(atoms, f); energies.append(e)
                row.update({"energy_eV": e, "relative_energy_eV": np.nan, "max_force_eV_A": fmax, "min_distance_A": mind, "finite": ok})
                if not ok: row.update({"status": "invalid_geometry_or_force", "failure_reason": "nonfinite geometry/force or invalid distance"})
            except Exception as exc:
                row.update({"status": "runtime_failure", "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=6)}); failures.append(row.copy())
            sp_rows.append(row)
        if energies:
            emin = min(energies)
            for row in sp_rows[-len(energies):]: row["relative_energy_eV"] = energies[row["image_index"]] - emin
        for image_index, atoms0 in enumerate(frames):
            for label, delta in perturbations.items():
                atoms = atoms0.copy(); atoms.set_cell(cell); atoms.set_pbc([True, True, True]); idx = dopant_indices(atoms)
                base = {"model_id": "legacy_foundation", "model_label": "Foundation", "model_sha256": runtime["model_sha256"], "path_id": path_id, "image_index": image_index, "perturbation": label, "dopant_indices": json.dumps(idx.tolist()), "diagnostic_only": True, "scientific_success": False, "status": "success", "failure_reason": "unconverged DFT snapshot; MACE-only relaxation diagnostic"}
                if len(idx) != 1:
                    base.update({"status": "identity_anomaly", "failure_reason": f"expected one dynamic dopant, found {len(idx)}"}); relax_rows.append(base); failures.append(base); continue
                atoms.positions[idx[0]] += delta; atoms.calc = calc
                traj_path = out / "trajectories" / f"{path_id}__image-{image_index:02d}__{label}.traj"
                final_path = out / "final_structures" / f"{path_id}__image-{image_index:02d}__{label}.extxyz"
                try:
                    initial = atoms.copy(); initial.set_cell(cell); initial.set_pbc([True, True, True]); initial.calc = calc
                    e0 = float(initial.get_potential_energy()); pos0 = initial.positions.copy(); host = np.delete(np.arange(len(initial)), idx[0])
                    opt = FIRE(atoms, trajectory=str(traj_path), logfile=None); opt.run(fmax=args.fmax, steps=args.max_steps)
                    ef = float(atoms.get_potential_energy()); ff = np.asarray(atoms.get_forces(), float); ok, mind, fmax = finite_geometry(atoms, ff); disp = np.linalg.norm(atoms.positions - pos0, axis=1)
                    write(final_path, atoms, format="extxyz")
                    base.update({"trajectory": str(traj_path), "final_structure": str(final_path), "steps": int(opt.nsteps), "converged": bool(opt.converged()), "initial_energy_eV": e0, "final_energy_eV": ef, "relative_energy_change_eV": ef - e0, "dopant_displacement_A": float(disp[idx[0]]), "whole_rmsd_A": float(np.sqrt(np.mean(disp ** 2))), "host_rmsd_A": float(np.sqrt(np.mean(disp[host] ** 2))), "max_displacement_A": float(np.max(disp)), "residual_force_eV_A": fmax, "min_distance_A": mind, "runaway": bool(not ok or np.max(disp) > 5.0), "identity_anomaly": False})
                    if not ok: base["status"] = "invalid_final_geometry"
                except Exception as exc:
                    base.update({"status": "runtime_failure", "failure_reason": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=6)}); failures.append(base)
                relax_rows.append(base)
    def dump(name, rows):
        fields = sorted({k for r in rows for k in r})
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    dump("foundation_path_single_point_diagnostic.csv", sp_rows); dump("foundation_perturb_relax_summary_diagnostic.csv", relax_rows); dump("failure_retry_ledger_diagnostic.csv", failures)
    (out / "coverage_summary.json").write_text(json.dumps({"paths_requested": args.paths, "single_point_rows": len(sp_rows), "relaxation_rows": len(relax_rows), "failures": len(failures), "diagnostic_only": True}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
