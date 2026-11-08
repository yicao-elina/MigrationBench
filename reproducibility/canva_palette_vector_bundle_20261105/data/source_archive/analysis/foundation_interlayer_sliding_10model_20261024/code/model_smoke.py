#!/usr/bin/env python3
"""Compute-node model/head smoke: one energy and finite force vector."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from ase.io import read


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--head", default="")
    parser.add_argument("--endpoint", type=Path, required=True)
    args = parser.parse_args()
    import torch

    original = torch.jit.load

    def load_on_cpu(*items, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return original(*items, **kwargs)

    torch.jit.load = load_on_cpu
    from mace.calculators import MACECalculator

    kwargs = {"model_paths": args.model, "device": "cpu", "default_dtype": "float64"}
    if args.head:
        kwargs["head"] = args.head
    try:
        calc = MACECalculator(**kwargs)
    except TypeError:
        kwargs["model_path"] = kwargs.pop("model_paths")
        calc = MACECalculator(**kwargs)
    atoms = read(args.endpoint)
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    max_force = float(np.linalg.norm(forces, axis=1).max())
    if not all(math.isfinite(value) for value in (energy, max_force)):
        raise RuntimeError(f"nonfinite smoke output: {energy}, {max_force}")
    print(f"MODEL_SMOKE_PASS model={args.model} head={args.head or 'default'} atoms={len(atoms)} energy={energy:.12g} max_force={max_force:.12g}")


if __name__ == "__main__":
    main()
