#!/usr/bin/env python3
"""Evaluate one registry model on the 47 immutable base source geometries.

This is intentionally a static single-point calculation.  It is not a
relaxation, does not overwrite an earlier campaign, and does not claim a DFT
error: the structures are MACE source inputs rather than locally available DFT
final geometries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read
from mace.calculators import MACECalculator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cr_index(atoms) -> int:
    indices = [index for index, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == "Cr"]
    if len(indices) != 1:
        raise ValueError(f"expected one Cr, received {len(indices)}")
    return indices[0]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inputs-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-index", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    registry = pd.read_csv(args.registry, sep="\t")
    manifest = pd.read_csv(args.manifest, sep="\t")
    if not 0 <= args.model_index < len(registry):
        raise IndexError(f"model index {args.model_index} outside registry of {len(registry)} models")
    model = registry.iloc[args.model_index].to_dict()
    model_id = model["analysis_model_id"]
    model_path = Path(model["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    observed_hash = sha256(model_path)
    if observed_hash != model["model_sha256"]:
        raise RuntimeError(
            f"model hash mismatch for {model_id}: expected {model['model_sha256']}, observed {observed_hash}"
        )
    destination = args.output_root / model_id
    destination.mkdir(parents=True, exist_ok=False)
    calculator = MACECalculator(model_paths=str(model_path), device=args.device, default_dtype="float64")
    rows = []
    forces = {}
    for task in manifest.sort_values("endpoint").to_dict("records"):
        endpoint = task["endpoint"]
        source = args.inputs_root / task["input_extxyz"]
        record = {
            "analysis_model_id": model_id,
            "display_label": model["display_label"],
            "analysis_family": model["analysis_family"],
            "seed_or_member": model["seed_or_member"],
            "model_path": str(model_path),
            "model_sha256_expected": model["model_sha256"],
            "model_sha256_observed": observed_hash,
            "endpoint": endpoint,
            "base_task_id": task["base_task_id"],
            "input_extxyz": task["input_extxyz"],
            "input_extxyz_sha256_expected": task["input_extxyz_sha256"],
            "input_extxyz_sha256_observed": sha256(source),
            "calculation_kind": "static_source_geometry_single_point",
            "static_geometry_scope": "MACE base input, not DFT final geometry",
            "status": "FAILURE",
            "energy_eV": np.nan,
            "fmax_eV_A": np.nan,
            "cr_force_x_eV_A": np.nan,
            "cr_force_y_eV_A": np.nan,
            "cr_force_z_eV_A": np.nan,
            "cr_force_norm_eV_A": np.nan,
            "error": "",
        }
        try:
            if record["input_extxyz_sha256_expected"] != record["input_extxyz_sha256_observed"]:
                raise RuntimeError("input hash mismatch")
            atoms = read(source)
            atoms.calc = calculator
            energy = float(atoms.get_potential_energy())
            atomic_forces = np.asarray(atoms.get_forces(), dtype=float)
            cr_force = atomic_forces[cr_index(atoms)]
            record.update(
                {
                    "status": "SUCCESS",
                    "energy_eV": energy,
                    "fmax_eV_A": float(np.linalg.norm(atomic_forces, axis=1).max()),
                    "cr_force_x_eV_A": float(cr_force[0]),
                    "cr_force_y_eV_A": float(cr_force[1]),
                    "cr_force_z_eV_A": float(cr_force[2]),
                    "cr_force_norm_eV_A": float(np.linalg.norm(cr_force)),
                }
            )
            forces[endpoint] = atomic_forces
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback_last_line"] = traceback.format_exc().strip().splitlines()[-1]
        rows.append(record)
    results = pd.DataFrame(rows)
    results.to_csv(destination / "source_geometry_predictions.tsv", sep="\t", index=False, float_format="%.12g")
    np.savez_compressed(destination / "source_geometry_forces.npz", **forces)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "calculation_kind": "static_source_geometry_single_point",
        "static_geometry_scope": "MACE base input, not DFT final geometry",
        "model": model,
        "model_sha256_observed": observed_hash,
        "registry_sha256": sha256(args.registry),
        "manifest_sha256": sha256(args.manifest),
        "n_requested": int(len(manifest)),
        "n_success": int(results["status"].eq("SUCCESS").sum()),
        "n_failure": int(results["status"].ne("SUCCESS").sum()),
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    if provenance["n_failure"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
