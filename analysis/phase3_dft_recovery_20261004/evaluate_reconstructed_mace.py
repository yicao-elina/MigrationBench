#!/usr/bin/env python3
"""Evaluate all four canonical MACE models on the reconstructed QE inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ase.io import read
from mace.calculators import MACECalculator


MODELS = {
    "foundation-omat": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/finetuned_MACE_compiled.model",
    "from-scratch": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE_models_l1_0802/mace_l1_0802_compiled.model",
    "naive-fine-tuning": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_600K/finetuned_MACE_multihead0804_compiled.model",
    "multi_T-fine-tuning": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_Multi_T/finetuned_MACE_multihead0804.model",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    rows = []
    for case_dir in sorted(p for p in args.input_root.iterdir() if p.is_dir()):
        input_path = case_dir / "scf.in"
        atoms = read(input_path, format="espresso-in")
        for model_name, model_path_str in MODELS.items():
            model_path = Path(model_path_str)
            calc = MACECalculator(model_paths=str(model_path), device=args.device, default_dtype="float64")
            atoms.calc = calc
            energy = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            row = {
                "case": case_dir.name,
                "model": model_name,
                "input_sha256": sha256(input_path),
                "model_path": str(model_path),
                "model_sha256": sha256(model_path),
                "nat": len(atoms),
                "energy_eV": energy,
                "max_force_eV_A": float((forces * forces).sum(axis=1).max() ** 0.5),
                "mean_force_eV_A": float((forces * forces).sum(axis=1).mean() ** 0.5),
                "forces_eV_A": forces.tolist(),
                "scope": "RECONSTRUCTION / REPRODUCED_SCOPE_LIMITED",
            }
            rows.append(row)
            print(case_dir.name, model_name, row["energy_eV"], row["max_force_eV_A"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
