#!/usr/bin/env python3
"""Time-convergence analysis for 2D dopant MSD in an ASE/extxyz trajectory.

The analysis concatenates an original trajectory and an optional continuation
segment, removes duplicate boundary frames, unwraps periodic coordinates, and
reports both cumulative-window fits and chronological 50 ps block fits. The
diffusion conversion is D_xy = slope / 4, with A^2/ps converted to cm^2/s.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import iread


def read_frames(paths: list[Path], dopant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[bool, bool, bool]]:
    positions: list[np.ndarray] = []
    steps: list[int] = []
    energies: list[float] = []
    force_max: list[float] = []
    species0: tuple[str, ...] | None = None
    cell0: np.ndarray | None = None
    pbc0: tuple[bool, bool, bool] | None = None
    dopant_idx: np.ndarray | None = None
    fallback_index = 0
    fallback_used = False
    for path in paths:
        for frame in iread(str(path), format="extxyz"):
            species = tuple(frame.get_chemical_symbols())
            if species0 is None:
                species0 = species
                cell0 = np.asarray(frame.cell.array, dtype=float)
                pbc0 = tuple(bool(x) for x in frame.pbc)
                dopant_idx = np.flatnonzero(np.asarray(species) == dopant)
                if len(dopant_idx) == 0:
                    raise ValueError(f"No dopant species {dopant!r} in {path}")
            if species != species0:
                raise ValueError(f"Species identity changed in {path}")
            if len(frame) != 2050 or tuple(bool(x) for x in frame.pbc) != pbc0:
                raise ValueError(f"Unexpected natoms/PBC in {path}: {len(frame)}/{frame.pbc}")
            if not np.isfinite(frame.positions).all():
                raise ValueError(f"Non-finite positions in {path}")
            if frame.has("momenta") and not np.isfinite(frame.get_momenta()).all():
                raise ValueError(f"Non-finite momenta in {path}")
            raw_step = frame.info.get("absolute_step", frame.info.get("step"))
            if raw_step is None:
                # The legacy from-scratch extxyz omits step/time metadata but
                # was saved every 200 MD steps. Record this reconstruction in
                # the output rather than silently treating frames as 1 fs.
                raw_step = fallback_index * 200
                fallback_used = True
            steps.append(int(raw_step))
            positions.append(np.asarray(frame.positions[dopant_idx], dtype=float))
            energies.append(float(frame.info.get("energy", np.nan)))
            forces = frame.get_forces()
            force_max.append(float(np.abs(forces).max()) if np.isfinite(forces).all() else np.inf)
            fallback_index += 1
    if not positions or cell0 is None or pbc0 is None:
        raise ValueError("No frames found")
    order = np.argsort(np.asarray(steps), kind="stable")
    steps_a = np.asarray(steps, dtype=int)[order]
    pos_a = np.asarray(positions, dtype=float)[order]
    energy_a = np.asarray(energies, dtype=float)[order]
    force_a = np.asarray(force_max, dtype=float)[order]
    keep = np.r_[True, np.diff(steps_a) > 0]
    return pos_a[keep], steps_a[keep], energy_a[keep], force_a[keep], pbc0, fallback_used


def unwrap_positions(pos: np.ndarray, cell: np.ndarray) -> np.ndarray:
    frac = np.einsum("nij,jk->nik", pos, np.linalg.inv(cell))
    unwrapped = np.empty_like(frac)
    unwrapped[0] = frac[0]
    for i in range(1, len(frac)):
        delta = frac[i] - frac[i - 1]
        delta -= np.rint(delta)
        unwrapped[i] = unwrapped[i - 1] + delta
    return np.einsum("nij,jk->nik", unwrapped, cell)


def fit(x: np.ndarray, y: np.ndarray) -> dict[str, float | int | None]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5 or np.ptp(x[mask]) <= 0:
        return {"n": int(mask.sum()), "slope_A2_per_ps": None, "Dxy_cm2_s": None, "r2": None}
    xx, yy = x[mask], y[mask]
    slope, intercept = np.polyfit(xx, yy, 1)
    pred = slope * xx + intercept
    ss_res = float(np.sum((yy - pred) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"n": int(mask.sum()), "slope_A2_per_ps": float(slope),
            "Dxy_cm2_s": float(slope / 4.0 * 1e-4), "r2": float(r2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--segment", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--dopant", default="Cr")
    ap.add_argument("--analysis-start-step", type=int, default=200000)
    ap.add_argument("--block-ps", type=float, default=50.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Use cell from the first frame; the campaign is a fixed-cell slab.
    first = next(iread(str(args.original), format="extxyz"))
    cell = np.asarray(first.cell.array, dtype=float)
    pos, steps, energies, forces, pbc, fallback_used = read_frames([args.original, args.segment], args.dopant)
    times = (steps - steps[0]) * 0.001
    mask = steps >= args.analysis_start_step
    if mask.sum() < 10:
        raise ValueError(f"Only {mask.sum()} frames at/after analysis start")
    pos = unwrap_positions(pos, cell)
    t = times[mask] - times[mask][0]
    r = pos[mask][:, :, :2]
    # Center over the dopant atoms, retaining the 2D migration signal.
    r = r.mean(axis=1)
    dr = r - r[0]
    msd = np.sum(dr * dr, axis=1)

    cumulative = []
    for window in (50.0, 100.0, 200.0, 500.0):
        sel = t <= window + 1e-9
        if sel.sum() >= 5:
            row = {"window_ps": window, "actual_end_ps": float(t[sel][-1])}
            row.update(fit(t[sel], msd[sel]))
        else:
            row = {"window_ps": window, "actual_end_ps": float(t[-1]), "n": int(sel.sum()),
                   "slope_A2_per_ps": None, "Dxy_cm2_s": None, "r2": None}
        cumulative.append(row)

    blocks = []
    block_count = int(np.floor(t[-1] / args.block_ps))
    for i in range(block_count):
        lo, hi = i * args.block_ps, (i + 1) * args.block_ps
        sel = (t >= lo - 1e-9) & (t <= hi + 1e-9)
        local_t = t[sel] - lo
        local_msd = np.sum((r[sel] - r[sel][0]) ** 2, axis=1)
        row = {"block_index": i + 1, "start_ps": lo, "end_ps": hi}
        row.update(fit(local_t, local_msd))
        blocks.append(row)

    summary = {
        "model": args.name,
        "original": str(args.original),
        "segment": str(args.segment),
        "dopant": args.dopant,
        "dopant_count": int(r.shape[1]),
        "natoms": 2050,
        "pbc": list(pbc),
        "frames_total_after_dedup": int(len(steps)),
        "analysis_start_step": args.analysis_start_step,
        "time_axis_reconstructed_for_missing_metadata": bool(fallback_used),
        "analysis_duration_ps": float(t[-1]),
        "finite_energy_fraction": float(np.isfinite(energies).mean()),
        "finite_force_fraction": float(np.isfinite(forces).mean()),
        "max_force_eV_A": float(np.nanmax(forces)),
        "cumulative_windows": cumulative,
        "blocks_50ps": blocks,
    }
    (args.out / "time_convergence.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out / "cumulative_windows.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cumulative[0].keys()); writer.writeheader(); writer.writerows(cumulative)
    with (args.out / "blocks_50ps.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=blocks[0].keys() if blocks else ["block_index"]); writer.writeheader(); writer.writerows(blocks)

    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.linewidth": 1.4})
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), constrained_layout=True)
    axes[0].plot(t, msd, color="#245a9a", lw=1.7, label="unwrapped MSD")
    for row in cumulative:
        if row["actual_end_ps"] == row["window_ps"] and row["slope_A2_per_ps"] is not None:
            end = float(row["actual_end_ps"]); slope = float(row["slope_A2_per_ps"])
            axes[0].plot([0, end], [0, slope * end], lw=1.0, label=f"{int(end)} ps fit")
    axes[0].set(xlabel="Time after 200 ps (ps)", ylabel="MSD$_{xy}$ (Å$^2$)")
    axes[0].legend(frameon=True, fontsize=8)
    if blocks:
        x = [f"{int(row['start_ps'])}–{int(row['end_ps'])}" for row in blocks]
        y = [row["Dxy_cm2_s"] for row in blocks]
        axes[1].plot(x, y, marker="o", color="#b24d35", lw=1.5)
        axes[1].tick_params(axis="x", rotation=55)
    axes[1].set(xlabel="Chronological 50 ps block", ylabel="D$_{xy}$ (cm$^2$ s$^{-1}$)")
    for ax in axes:
        ax.tick_params(width=1.2, length=4)
    fig.savefig(args.out / "time_convergence.png", dpi=300)
    fig.savefig(args.out / "time_convergence.pdf")
    fig.savefig(args.out / "time_convergence.svg")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
