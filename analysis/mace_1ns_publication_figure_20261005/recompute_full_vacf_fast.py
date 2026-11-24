#!/usr/bin/env python3
"""Compute full-trajectory Cr VACF efficiently from remote extxyz inputs."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import iread


def read_cr_velocities(paths: list[Path]) -> np.ndarray:
    frames = []
    species = None
    for path in paths:
        for atoms in iread(str(path), format="extxyz"):
            current_species = tuple(atoms.get_chemical_symbols())
            if species is None:
                species = current_species
            if current_species != species:
                raise ValueError("species changed across trajectory")
            mask = np.asarray(current_species) == "Cr"
            frames.append(atoms.get_momenta()[mask] / atoms.get_masses()[mask, None])
    velocity = np.asarray(frames, dtype=float)
    return velocity - velocity.mean(axis=1, keepdims=True)


def vacf_fft(velocity: np.ndarray) -> np.ndarray:
    n = len(velocity)
    series = velocity.reshape(n, -1)
    f = np.fft.rfft(series, n=2 * n, axis=0)
    ac = np.fft.irfft(f * np.conjugate(f), n=2 * n, axis=0)[:n].real
    return np.sum(ac, axis=1) / (np.arange(n, 0, -1) * series.shape[1])


def main() -> None:
    root, model, output = sys.argv[1:]
    run = Path(root) / model
    manifest = json.loads((run / "run_manifest.json").read_text())
    original = Path(manifest["input"])
    segment = run / Path(manifest["segment"]).name
    vacf = vacf_fft(read_cr_velocities([original, segment]))
    vacf /= vacf[0] if vacf[0] else 1.0
    with Path(output).open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["lag_ps", "vacf_normalized"])
        writer.writerows((i * 0.2, float(value)) for i, value in enumerate(vacf))
    print(f"{model}: frames={len(vacf)} last_lag_ps={(len(vacf)-1)*0.2:.1f}", flush=True)


if __name__ == "__main__":
    main()
