"""Materialize all ten saved QE path snapshots for same-geometry diagnostics."""
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
OUT = Path("/tmp/migrationbench_all_dft_base_inputs_20261003")
HARTREE_TO_EV = 27.211386245988
BOHR_TO_A = 0.529177210903


def parse_path(path: Path, nat: int) -> tuple[list[np.ndarray], list[float]]:
    blocks = re.split(r"\nImage:\s*", path.read_text(errors="replace"))[1:]
    positions, energies = [], []
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        try:
            energy = float(lines[1].split()[0]) * HARTREE_TO_EV
        except ValueError:
            continue
        rows = []
        for line in lines[2:]:
            vals = line.split()
            if len(vals) >= 6:
                try:
                    rows.append([float(x.replace("D", "E")) for x in vals[:6]])
                except ValueError:
                    pass
            if len(rows) == nat:
                break
        if len(rows) == nat and np.isfinite(rows).all():
            positions.append(np.asarray(rows)[:, :3] * BOHR_TO_A)
            energies.append(energy)
    return positions, energies


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gate = {r["path_id"]: r for r in csv.DictReader((ROOT / "dft_gate_manifest.csv").open())}
    rows = []
    for path_id, rec in gate.items():
        initial = read(ROOT / "inputs" / f"{path_id}__initial.extxyz")
        rel = rec["remote_neb_dir"].removeprefix("/scratch4/pclancy3/joshitha/vdW-sratch/sb2te3/")
        path_file = SNAP / rel / "sb2te3.path"
        positions, energies = parse_path(path_file, len(initial))
        if len(positions) != 10:
            raise ValueError(f"{path_id}: expected 10 finite path images, found {len(positions)}")
        frames = [Atoms(symbols=initial.get_chemical_symbols(), positions=pos,
                        cell=initial.cell.array, pbc=True) for pos in positions]
        target = OUT / f"{path_id}__base.extxyz"
        write(target, frames, format="extxyz", write_results=False)
        rows.append({"path_id": path_id, "image_count": len(frames), "input": str(target),
                     "input_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                     "dft_path_sha256": rec["dft_path_sha256"], "dft_path_source": str(path_file),
                     "dft_energy_units_converted": "Hartree_to_eV", "coordinate_units_converted": "Bohr_to_Angstrom"})
    with (OUT / "all_neb_base_input_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    manifest = {"n_paths": len(rows), "n_images_per_path": 10, "n_frames": len(rows) * 10,
                "energy_conversion": HARTREE_TO_EV, "coordinate_conversion": BOHR_TO_A,
                "scientific_scope": "all ten saved NEB path snapshots; including two paths that fail the exploratory energy-table stability screen"}
    (OUT / "all_neb_base_input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
