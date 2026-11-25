#!/usr/bin/env python3
"""Materialize non-equilibrium DFT-path samples and the two historical perturbations."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write


ROOT = Path(__file__).resolve().parent
SNAP = ROOT / "remote_snapshot"
OUT = ROOT / "exploratory_energy_audit" / "path_sampling"
PERTURBATIONS = {
    "base": np.array([0.0, 0.0, 0.0]),
    "perturb_a_dx020": np.array([0.20, 0.0, 0.0]),
    "perturb_b_tilt020": np.array([-0.10, 0.14, 0.10]),
}
HARTREE_TO_EV = 27.211386245988


def parse_path(path: Path, nat: int) -> tuple[list[np.ndarray], list[float]]:
    text = path.read_text(errors="replace")
    blocks = re.split(r"\nImage:\s*", text)[1:]
    positions, energies = [], []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        try: energy = float(lines[1].split()[0]) * HARTREE_TO_EV
        except ValueError: continue
        rows = []
        for line in lines[2:]:
            vals = line.split()
            if len(vals) >= 6:
                try: rows.append([float(x.replace("D", "E")) for x in vals[:6]])
                except ValueError: pass
            if len(rows) == nat: break
        if len(rows) == nat and np.isfinite(rows).all():
            positions.append(np.array(rows)[:, :3] * 0.529177210903)
            energies.append(energy)
    return positions, energies


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame_rows = []
    audit = list(csv.DictReader((ROOT / "exploratory_energy_audit" / "scf_neb_iteration_energy_audit.csv").open()))
    gate = {r["path_id"]: r for r in csv.DictReader((ROOT / "dft_gate_manifest.csv").open())}
    eligible = [r for r in audit if r["exploratory_energy_stable"] == "True"]
    for rec in eligible:
        path_id = rec["path_id"]
        initial = read(ROOT / "inputs" / f"{path_id}__initial.extxyz")
        parts = path_id.split("__", 1)
        remote_dir = gate[path_id]["remote_neb_dir"]
        rel = remote_dir.removeprefix("/scratch4/pclancy3/joshitha/vdW-sratch/sb2te3/")
        path_file = SNAP / rel / "sb2te3.path"
        positions, energies = parse_path(path_file, len(initial))
        if len(positions) != 10:
            raise ValueError(f"{path_id}: expected 10 finite path images, found {len(positions)}")
        for variant, vector in PERTURBATIONS.items():
            frames = []
            for i, (pos, energy) in enumerate(zip(positions, energies), start=1):
                atoms = Atoms(symbols=initial.get_chemical_symbols(), positions=pos, cell=initial.cell.array, pbc=True)
                dopant = next(j for j, s in enumerate(atoms.get_chemical_symbols()) if s not in {"Sb", "Te"})
                if variant != "base": atoms.positions[dopant] += vector
                frames.append(atoms)
                frame_rows.append({
                    "sample_id": f"{path_id}__image{i:02d}__{variant}", "path_id": path_id, "image_index": i,
                    "variant": variant, "dopant_element": atoms[dopant].symbol, "dopant_atom_index": dopant,
                    "perturbation_vector_A": json.dumps(vector.tolist()), "dft_path_energy_eV": energy,
                    "dft_path_source": str(path_file), "dft_path_sha256": gate[path_id]["dft_path_sha256"],
                    "initial_endpoint_sha256": gate[path_id]["initial_endpoint_sha256"], "final_endpoint_sha256": gate[path_id]["final_endpoint_sha256"],
                    "dft_energy_status": "exploratory_non_equilibrium_path_sample", "scientific_success": False,
                })
            target = OUT / f"{path_id}__{variant}.extxyz"
            write(target, frames, format="extxyz", write_results=False)
    with (OUT / "path_sampling_manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(frame_rows[0])); w.writeheader(); w.writerows(frame_rows)
    summary = {"n_eligible_paths": len(eligible), "n_frames": len(frame_rows), "variants": list(PERTURBATIONS), "n_images_per_path": 10, "dft_gate_status": "not_fully_converged; exploratory_energy_stability_only", "sampling_sha256": hashlib.sha256((OUT / "path_sampling_manifest.csv").read_bytes()).hexdigest()}
    (OUT / "sampling_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
