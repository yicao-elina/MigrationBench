#!/usr/bin/env python3
"""Evaluate one trained artifact on the fixed 2050-atom input before MD/NEB."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from ase.io import read
from mace.calculators import MACECalculator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--family", required=True)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--output-root", type=Path, help="training root for this attempt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    family = cfg["families"][args.family]
    foundation = cfg["foundation_models"][family["foundation_model"]]
    output_root = (args.output_root or Path(cfg["remote_root"])).resolve()
    root = output_root / family["output_dir_template"].format(seed=args.training_seed)
    manifest_path = root / "training_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing training manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    model = Path(manifest["artifact_path"])
    input_path = Path(cfg["md"]["input"])
    if not model.is_file() or sha256(model) != manifest["artifact_sha256"]:
        raise SystemExit("trained artifact missing or hash changed")
    if not input_path.is_file() or sha256(input_path) != cfg["md"]["input_sha256"]:
        raise SystemExit("common MD input missing or hash changed")
    atoms = read(str(input_path), index=int(cfg["md"].get("input_frame", -1)), format="extxyz")
    if len(atoms) != int(cfg["md"]["required_natoms"]):
        raise SystemExit("input failed atom-count gate")
    if tuple(bool(value) for value in atoms.pbc) != tuple(cfg["md"]["required_pbc"]):
        raise SystemExit("input failed PBC gate")
    if not np.isfinite(atoms.get_positions()).all() or not np.isfinite(atoms.get_cell().array).all():
        raise SystemExit("input failed finite geometry gate")

    kwargs = {"model_paths": str(model), "device": args.device, "default_dtype": "float64", "model_type": foundation["model_type"]}
    atoms.calc = MACECalculator(**kwargs)
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    stress = np.asarray(atoms.get_stress(), dtype=float)
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    result = {
        "schema_version": "1.0",
        "campaign_id": cfg["campaign_id"],
        "family": args.family,
        "training_seed": args.training_seed,
        "foundation_name": foundation["canonical_name"],
        "model_type": foundation["model_type"],
        "model": str(model),
        "model_sha256": sha256(model),
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "natoms": len(atoms),
        "symbols_hash": hashlib.sha256(" ".join(atoms.get_chemical_symbols()).encode()).hexdigest(),
        "pbc": [bool(value) for value in atoms.pbc],
        "energy_eV": energy,
        "max_force_eV_A": float(np.abs(forces).max()),
        "stress_finite": bool(np.isfinite(stress).all()),
        "force_finite": bool(np.isfinite(forces).all()),
        "energy_finite": bool(np.isfinite(energy)),
        "min_distance_A": float(np.nanmin(distances)),
        "runtime_device": args.device,
        "output_root": str(output_root),
    }
    result["pass"] = bool(
        result["force_finite"]
        and result["energy_finite"]
        and result["stress_finite"]
        and np.isfinite(result["min_distance_A"])
        and result["min_distance_A"] >= float(cfg["md"].get("min_distance_A", 0.5))
    )
    output = root / "model_gate.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite model gate: {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(42)


if __name__ == "__main__":
    main()
