#!/usr/bin/env python3
"""Evaluate one trained artifact on the fixed MD input before release to MD."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from ase.io import read
from mace.calculators import MACECalculator


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--family", required=True)
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    md = cfg["md"]
    manifest_path = Path(cfg["remote_root"]) / "training" / args.family / ("seed%d" % args.training_seed) / "training_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("missing training manifest: %s" % manifest_path)
    manifest = json.loads(manifest_path.read_text())
    model = Path(manifest["artifact"])
    input_path = Path(md["input"])
    atoms = read(str(input_path), index=int(md.get("input_frame", -1)), format="extxyz")
    if len(atoms) != int(md["required_natoms"]) or tuple(bool(x) for x in atoms.pbc) != tuple(md["required_pbc"]):
        raise SystemExit("input failed atom/PBC gate")
    atoms.calc = MACECalculator(model_paths=str(model), device=args.device, default_dtype="float64")
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    result = {
        "family": args.family, "training_seed": args.training_seed,
        "model": str(model), "model_sha256": sha256(model),
        "input": str(input_path), "input_sha256": md["input_sha256"],
        "natoms": len(atoms), "pbc": [bool(x) for x in atoms.pbc],
        "energy_eV": energy, "max_force_eV_A": float(np.abs(forces).max()),
        "force_finite": bool(np.isfinite(forces).all()), "energy_finite": bool(np.isfinite(energy)),
        "min_distance_A": float(np.nanmin(distances)),
    }
    result["pass"] = bool(result["force_finite"] and result["energy_finite"] and np.isfinite(result["min_distance_A"]) and result["min_distance_A"] >= float(md.get("min_distance_A", 0.5)))
    out = manifest_path.parent / "static_model_gate.json"
    if out.exists():
        raise SystemExit("refusing to overwrite static gate: %s" % out)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(42)


if __name__ == "__main__":
    main()
