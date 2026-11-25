#!/usr/bin/env python3
"""Evaluate MACE-MH-1 on the canonical Cr fixed-path campaign.

This is an additive diagnostic supplement for the four-row Foundation-model
ensemble figure.  It deliberately evaluates only the materialized fixed-path
inputs used by the Cr campaign and does not run a relaxed NEB.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import read
from mace.calculators import MACECalculator


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis" / "neb_cross_element_triptychs_20261024"
DATA_OUT = OUT / "data" / "foundation_class_4x3_R02"
INPUT_DIR = (
    REPO
    / "analysis"
    / "task3_endpoint_only_neb"
    / "neb_fixed_vs_self_matrix_20261019"
    / "inputs"
    / "fixed_r01"
)
MODEL_PATH = Path("/Users/alina/Downloads/mace-foundation-models/mace-mh-1.model")
OUTPUT = DATA_OUT / "Cr_mace_mh1_fixed_path_profiles_R02.tsv"
MODEL_ID = "mace-mh-1"
PATHS = [
    ("within_gap", "cr_canonical_in_gap", "in-gap diffusion"),
    ("gap_to_ql", "cr_canonical_gap_to_ql", "gap→QL"),
    ("within_ql", "cr_canonical_within_ql", "within-QL"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    model_hash = sha256(MODEL_PATH)
    calc = MACECalculator(
        model_paths=[str(MODEL_PATH)],
        device="cpu",
        default_dtype="float64",
        head="omat_pbe",
    )
    rows: list[dict[str, object]] = []
    input_hashes: dict[str, str] = {}
    for path_key, path_id, path_class in PATHS:
        input_path = INPUT_DIR / f"{path_id}.extxyz"
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        frames = read(str(input_path), index=":")
        if len(frames) != 10 or any(len(atoms) != 81 for atoms in frames):
            raise ValueError(f"Expected 10 81-atom frames: {input_path}")
        if any(not np.isfinite(atoms.positions).all() for atoms in frames):
            raise ValueError(f"Non-finite geometry: {input_path}")
        input_hashes[path_id] = sha256(input_path)
        energies: list[float] = []
        fmax_values: list[float] = []
        for image_index, atoms in enumerate(frames):
            atoms.calc = calc
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            fmax = float(np.abs(forces).max())
            if not np.isfinite(energy) or not np.isfinite(fmax):
                raise ValueError(f"Non-finite MACE-MH-1 result: {input_path} image {image_index}")
            energies.append(energy)
            fmax_values.append(fmax)
        baseline = energies[0]
        for image_index, (energy, fmax) in enumerate(zip(energies, fmax_values)):
            rows.append(
                {
                    "element": "Cr",
                    "path_key": path_key,
                    "path_id": path_id,
                    "model_id": MODEL_ID,
                    "model_class": "MACE-MH",
                    "image_index": image_index + 1,
                    "s": image_index / 9.0,
                    "energy_eV": energy,
                    "relative_energy_eV": energy - baseline,
                    "max_force_eV_A": fmax,
                    "head": "omat_pbe",
                    "model_sha256": model_hash,
                    "input_sha256": input_hashes[path_id],
                    "input_path": str(input_path),
                    "evaluation_status": "FINITE_DIAGNOSTIC_LOCAL_MACE",
                    "profile_source": str(input_path),
                    "source_record": "MACE-MH-1_LOCAL_FIXED_PATH_R02",
                    "path_class": path_class,
                }
            )
    if len(rows) != 30:
        raise ValueError(f"Expected 30 rows, found {len(rows)}")
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "artifact": str(OUTPUT),
        "status": "FINITE_DIAGNOSTIC_LOCAL_MACE",
        "model_id": MODEL_ID,
        "model_path": str(MODEL_PATH),
        "model_sha256": model_hash,
        "calculator_head": "omat_pbe",
        "device": "cpu",
        "dtype": "float64",
        "input_paths": {path_id: str(INPUT_DIR / f"{path_id}.extxyz") for _, path_id, _ in PATHS},
        "input_sha256": input_hashes,
        "rows": len(rows),
        "images_per_path": 10,
        "atoms_per_image": 81,
        "operator": "fixed-path single-point evaluation; no relaxation",
        "scientific_boundary": "diagnostic local Foundation-model profile; not a converged DFT ground truth and not a self-relaxed NEB barrier",
    }
    (DATA_OUT / "Cr_mace_mh1_fixed_path_manifest_R02.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "rows": len(rows), "model_sha256": model_hash}, indent=2))


if __name__ == "__main__":
    main()
