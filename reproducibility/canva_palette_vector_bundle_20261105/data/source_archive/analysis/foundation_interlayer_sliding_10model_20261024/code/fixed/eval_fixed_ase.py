#!/usr/bin/env python3
"""Evaluate one retained ten-image path with an explicit MACE checkpoint."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
from ase.io import read


def parse_qe(path: Path) -> dict[int, float]:
    text = path.read_text()
    blocks = []
    for section in re.split(r"image\s+energy \(eV\)\s+error \(eV/A\)\s+frozen", text)[1:]:
        rows = []
        for line in section.strip().splitlines():
            parts = line.split()
            if len(parts) < 4:
                if rows:
                    break
                continue
            try:
                image = int(parts[0])
                energy = float(parts[1])
                float(parts[2])
                rows.append((image - 1, energy))
            except ValueError:
                if rows:
                    break
        if rows:
            blocks.append(rows)
    if not blocks:
        raise ValueError(f"no QE image-energy block found in {path}")
    result = dict(blocks[-1])
    if sorted(result) != list(range(10)):
        raise ValueError(f"expected ten QE energies, found indices {sorted(result)}")
    return result


def build_calculator(model: str, head: str | None):
    # The CPU TorchScript workaround is required for embedded model buffers.
    import torch

    original = torch.jit.load

    def load_on_cpu(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return original(*args, **kwargs)

    torch.jit.load = load_on_cpu
    from mace.calculators import MACECalculator

    kwargs = {"model_paths": model, "device": "cpu", "default_dtype": "float64"}
    if head:
        kwargs["head"] = head
    try:
        return MACECalculator(**kwargs)
    except TypeError:
        kwargs["model_path"] = kwargs.pop("model_paths")
        return MACECalculator(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--head", default="")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--neb-out", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    images = read(args.xyz, index=":")
    if len(images) != 10:
        raise ValueError(f"expected 10 images, found {len(images)}")
    counts = [len(atoms) for atoms in images]
    if len(set(counts)) != 1:
        raise ValueError(f"inconsistent atom counts: {counts}")
    qe = parse_qe(args.neb_out)
    calc = build_calculator(args.model, args.head or None)
    rows = []
    for image_index, atoms in enumerate(images):
        atoms.calc = calc
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        max_force = float(np.max(np.linalg.norm(forces, axis=1)))
        values = (energy, max_force, qe[image_index], energy - qe[image_index])
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"nonfinite result at image {image_index}: {values}")
        rows.append({
            "model_name": args.model_label,
            "image_index": image_index,
            "predicted_energy_eV": f"{energy:.16g}",
            "max_force_eV_A": f"{max_force:.16g}",
            "qe_reference_energy_eV": f"{qe[image_index]:.16g}",
            "error_vs_qe_reference_eV": f"{energy - qe[image_index]:.16g}",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} finite rows to {args.output}")


if __name__ == "__main__":
    main()
