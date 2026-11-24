#!/usr/bin/env python3
"""Streaming coordinate-only MSD convergence analysis for 2050-atom extxyz."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STEP_RE = re.compile(r"(?:absolute_step|step)=(-?\d+(?:\.\d+)?)")
LATTICE_RE = re.compile(r'Lattice="([^"]+)"')


def frames(path: Path, dopant: str = "Cr"):
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        frame_index = 0
        while True:
            line = fh.readline()
            if not line:
                break
            n = int(line.strip())
            header = fh.readline().strip()
            lattice = LATTICE_RE.search(header)
            if lattice is None:
                raise ValueError(f"Missing Lattice in {path} frame {frame_index}")
            cell = np.asarray([float(x) for x in lattice.group(1).split()], dtype=float).reshape(3, 3)
            match = STEP_RE.search(header)
            step = int(float(match.group(1))) if match else frame_index * 200
            cr = []
            for _ in range(n):
                fields = fh.readline().split()
                if len(fields) < 4:
                    raise ValueError(f"Malformed atom line in {path} frame {frame_index}")
                if fields[0] == dopant:
                    cr.append([float(fields[1]), float(fields[2]), float(fields[3])])
            if not cr:
                raise ValueError(f"No {dopant} atoms in {path} frame {frame_index}")
            yield step, cell, np.asarray(cr, dtype=float)
            frame_index += 1


def read(paths: list[Path], dopant: str):
    steps, cells, coords = [], [], []
    for path in paths:
        for step, cell, cr in frames(path, dopant):
            steps.append(step); cells.append(cell); coords.append(cr)
    order = np.argsort(np.asarray(steps), kind="stable")
    steps = np.asarray(steps, dtype=int)[order]
    coords = np.asarray(coords, dtype=float)[order]
    cells = np.asarray(cells, dtype=float)[order]
    keep = np.r_[True, np.diff(steps) > 0]
    return steps[keep], coords[keep], cells[0]


def unwrap(coords: np.ndarray, cell: np.ndarray) -> np.ndarray:
    frac = np.einsum("nij,jk->nik", coords, np.linalg.inv(cell))
    out = np.empty_like(frac); out[0] = frac[0]
    for i in range(1, len(frac)):
        d = frac[i] - frac[i - 1]; d -= np.rint(d); out[i] = out[i - 1] + d
    return np.einsum("nij,jk->nik", out, cell)


def fit(x, y):
    if len(x) < 5 or np.ptp(x) <= 0:
        return {"n": int(len(x)), "slope_A2_per_ps": None, "Dxy_cm2_s": None, "r2": None}
    slope, intercept = np.polyfit(x, y, 1); pred = slope * x + intercept
    ssr = float(np.sum((y - pred) ** 2)); sst = float(np.sum((y - y.mean()) ** 2))
    return {"n": int(len(x)), "slope_A2_per_ps": float(slope),
            "Dxy_cm2_s": float(slope / 4.0 * 1e-4),
            "r2": float(1 - ssr / sst) if sst > 0 else None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--original", type=Path, required=True); ap.add_argument("--segment", type=Path, required=True); ap.add_argument("--out", type=Path, required=True); ap.add_argument("--name", required=True); ap.add_argument("--start-step", type=int, default=200000); ap.add_argument("--block-ps", type=float, default=50.0); a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    steps, coords, cell = read([a.original, a.segment], "Cr")
    mask = steps >= a.start_step
    steps, coords = steps[mask], coords[mask]
    t = (steps - steps[0]) * 0.001
    pos = unwrap(coords, cell).mean(axis=1)[:, :2]
    msd = np.sum((pos - pos[0]) ** 2, axis=1)
    cumulative = []
    for window in (50.0, 100.0, 200.0, 500.0):
        s = t <= window + 1e-9; row = {"window_ps": window, "actual_end_ps": float(t[s][-1]) if s.any() else 0.0}; row.update(fit(t[s], msd[s])); cumulative.append(row)
    blocks = []
    for i in range(int(np.floor(t[-1] / a.block_ps))):
        lo, hi = i * a.block_ps, (i + 1) * a.block_ps; s = (t >= lo - 1e-9) & (t <= hi + 1e-9); local_t = t[s] - lo; local_msd = np.sum((pos[s] - pos[s][0]) ** 2, axis=1); row = {"block_index": i + 1, "start_ps": lo, "end_ps": hi}; row.update(fit(local_t, local_msd)); blocks.append(row)
    summary = {"model": a.name, "original": str(a.original), "segment": str(a.segment), "dopant": "Cr", "dopant_count": int(coords.shape[1]), "natoms": 2050, "pbc": [True, True, True], "frames": int(len(steps)), "start_step": int(steps[0]), "last_step": int(steps[-1]), "duration_ps": float(t[-1]), "time_axis_fallback_for_legacy_original": True, "cumulative_windows": cumulative, "blocks_50ps": blocks}
    (a.out / "time_convergence.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, rows in (("cumulative_windows.csv", cumulative), ("blocks_50ps.csv", blocks)):
        with (a.out / name).open("w", newline="") as f:
            fields = list(rows[0]) if rows else ["empty"]; w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.linewidth": 1.4})
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True); ax[0].plot(t, msd, color="#245a9a", lw=1.7)
    ax[0].set(xlabel="Time after 200 ps (ps)", ylabel="MSD$_{xy}$ (Å$^2$)")
    if blocks:
        ax[1].plot([f"{int(x['start_ps'])}–{int(x['end_ps'])}" for x in blocks], [x["Dxy_cm2_s"] for x in blocks], marker="o", color="#b24d35", lw=1.5); ax[1].tick_params(axis="x", rotation=55)
    ax[1].set(xlabel="Chronological 50 ps block", ylabel="D$_{xy}$ (cm$^2$ s$^{-1}$)")
    for x in ax: x.tick_params(width=1.2, length=4)
    for ext in ("png", "pdf", "svg"): fig.savefig(a.out / f"time_convergence.{ext}", dpi=300 if ext == "png" else None)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
