#!/usr/bin/env python3
"""Compute a full-trajectory Cr MSD efficiently from remote extxyz inputs."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import iread


def read_cr_positions(paths: list[Path]) -> np.ndarray:
    frames = []
    species = None
    cell = None
    for path in paths:
        for atoms in iread(str(path), format="extxyz"):
            current_species = tuple(atoms.get_chemical_symbols())
            if species is None:
                species = current_species
                cell = np.asarray(atoms.cell.array, dtype=float)
            if current_species != species:
                raise ValueError("species changed across trajectory")
            frames.append(atoms.positions[np.asarray(current_species) == "Cr"].copy())
    xyz = np.asarray(frames, dtype=float)
    frac = np.einsum("nij,jk->nik", xyz, np.linalg.inv(cell))
    unwrapped = np.empty_like(frac)
    unwrapped[0] = frac[0]
    for i in range(1, len(frac)):
        delta = frac[i] - frac[i - 1]
        delta -= np.rint(delta)
        unwrapped[i] = unwrapped[i - 1] + delta
    return unwrapped @ cell


def msd_fft(pos: np.ndarray) -> np.ndarray:
    n = len(pos)
    series = pos.reshape(n, -1)
    f = np.fft.rfft(series, n=2 * n, axis=0)
    ac = np.fft.irfft(f * np.conjugate(f), n=2 * n, axis=0)[:n].real
    sq = series * series
    cs = np.concatenate((np.zeros((1, series.shape[1])), np.cumsum(sq, axis=0)))
    lag = np.arange(n)
    count = n - lag
    sum0 = cs[n - lag]
    sumlag = cs[n] - cs[lag]
    return np.mean((sum0 + sumlag - 2.0 * ac) / count[:, None], axis=1)


def main() -> None:
    root, model, output = sys.argv[1:]
    run = Path(root) / model
    manifest = json.loads((run / "run_manifest.json").read_text())
    original = Path(manifest["input"])
    segment = run / Path(manifest["segment"]).name
    pos = read_cr_positions([original, segment])
    msd = msd_fft(pos)
    with Path(output).open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["lag_ps", "msd_A2"])
        writer.writerows((i * 0.2, float(value)) for i, value in enumerate(msd[1:], start=1))
    print(f"{model}: frames={len(pos)} last_lag_ps={(len(pos)-1)*0.2:.1f}", flush=True)


if __name__ == "__main__":
    main()
