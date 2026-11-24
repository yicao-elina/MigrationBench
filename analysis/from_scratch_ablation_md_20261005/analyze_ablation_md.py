#!/usr/bin/env python
"""Audit 20 ps ablation segments and make Atlas-style screening figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import iread


def unwrap(frac: np.ndarray) -> np.ndarray:
    out = np.empty_like(frac)
    out[0] = frac[0]
    for i in range(1, len(frac)):
        delta = frac[i] - frac[i - 1]
        delta -= np.rint(delta)
        out[i] = out[i - 1] + delta
    return out


def fit(x: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5 or np.ptp(x[mask]) <= 0:
        return {"n": int(mask.sum()), "slope_A2_per_ps": None, "Dxy_cm2_s": None, "r2": None}
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    pred = slope * x[mask] + intercept
    ss_tot = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    r2 = None if ss_tot == 0 else 1.0 - float(np.sum((y[mask] - pred) ** 2)) / ss_tot
    return {
        "n": int(mask.sum()),
        "slope_A2_per_ps": float(slope),
        "Dxy_cm2_s": float(slope / 4.0 * 1.0e-4),
        "r2": r2,
    }


def load(run: Path) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((run / "run_manifest.json").read_text())
    status = json.loads((run / "status.json").read_text())
    frames = list(iread(run / "trajectory.extxyz", format="extxyz"))
    if not frames:
        raise RuntimeError(f"{run}: empty trajectory")
    first = frames[0]
    species = tuple(first.get_chemical_symbols())
    cell = np.asarray(first.cell.array)
    positions = []
    velocities = []
    energies = []
    temperatures = []
    forces = []
    steps = []
    for index, atoms in enumerate(frames):
        if len(atoms) != 2050 or tuple(bool(x) for x in atoms.pbc) != (True, True, True):
            raise RuntimeError(f"{run}: identity/PBC failure at frame {index}")
        if tuple(atoms.get_chemical_symbols()) != species:
            raise RuntimeError(f"{run}: species identity changed at frame {index}")
        if not np.allclose(atoms.cell.array, cell, rtol=0, atol=1e-8):
            raise RuntimeError(f"{run}: cell changed at frame {index}")
        positions.append(atoms.positions[np.asarray(species) == "Cr"])
        velocities.append(atoms.get_momenta()[np.asarray(species) == "Cr"] / atoms.get_masses()[np.asarray(species) == "Cr", None])
        energies.append(float(atoms.info.get("potential_energy_eV", np.nan)))
        temperatures.append(float(atoms.info.get("temperature_K", np.nan)))
        forces.append(float(atoms.info.get("max_force_eV_A", np.nan)))
        steps.append(int(atoms.info.get("step", index * 200)))
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    steps = np.asarray(steps)
    fractional = np.einsum("nij,jk->nik", positions, np.linalg.inv(cell))
    unwrapped = unwrap(fractional) @ cell
    return manifest, status, unwrapped, velocities, np.column_stack((steps, energies, temperatures, forces))


def analyze_run(run: Path, out: Path) -> dict:
    manifest, status, pos, vel, scalar = load(run)
    steps = scalar[:, 0].astype(int)
    time_ps = (steps - steps[0]) * manifest["timestep_fs"] * 1.0e-3
    n = len(pos)
    max_lag = max(1, min(n - 1, 50))
    lag_ps = np.arange(1, max_lag + 1) * manifest["write_interval_steps"] * manifest["timestep_fs"] * 1.0e-3
    msd = []
    for lag in range(1, max_lag + 1):
        displacement = pos[lag:] - pos[:-lag]
        msd.append(float(np.mean(np.sum(displacement[:, :, :2] ** 2, axis=2))))
    msd = np.asarray(msd)
    fit_window = fit(lag_ps, msd)
    centered_v = vel - vel.mean(axis=1, keepdims=True)
    vacf = np.asarray([np.mean(centered_v[:-k] * centered_v[k:]) if k else np.mean(centered_v * centered_v) for k in range(min(n, 50))])
    vacf = vacf / vacf[0] if len(vacf) and vacf[0] else vacf
    displacements = pos[1:, :, :2] - pos[:-1, :, :2]
    hop_threshold_A = 1.5
    candidate_hops = int(np.sum(np.linalg.norm(displacements, axis=2) > hop_threshold_A))
    blocks = []
    for index in range(4):
        lo, hi = 5.0 * index, 5.0 * (index + 1)
        mask = (time_ps >= lo) & (time_ps <= hi)
        block_msd = np.mean(np.sum((pos[mask] - pos[mask][0])[:, :, :2] ** 2, axis=2), axis=1) if mask.sum() else np.array([])
        block_t = time_ps[mask] - lo if mask.sum() else np.array([])
        blocks.append({"block": index + 1, "start_ps": lo, "end_ps": hi, **fit(block_t, block_msd)})
    finite = bool(np.isfinite(pos).all() and np.isfinite(vel).all() and np.isfinite(scalar[:, 1:]).all())
    complete_grid = bool(status.get("status") == "complete" and steps[-1] == manifest["steps_requested"] and np.all(np.diff(steps) == manifest["write_interval_steps"]))
    result = {
        "run": str(run),
        "family": manifest["family"],
        "training_seed": manifest["training_seed"],
        "md_seed": manifest["md_seed"],
        "model_sha256": manifest["model_sha256"],
        "input_sha256": manifest["input_sha256"],
        "natoms": 2050,
        "pbc": manifest["pbc"],
        "cell_A": manifest["cell_A"],
        "status": status,
        "frames": int(n),
        "start_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "duration_ps": float(time_ps[-1]),
        "finite": finite,
        "complete_step_grid": complete_grid,
        "temperature_K": {"min": float(np.nanmin(scalar[:, 2])), "mean": float(np.nanmean(scalar[:, 2])), "max": float(np.nanmax(scalar[:, 2]))},
        "potential_energy_eV": {"min": float(np.nanmin(scalar[:, 1])), "mean": float(np.nanmean(scalar[:, 1])), "max": float(np.nanmax(scalar[:, 1]))},
        "max_force_eV_A": {"max": float(np.nanmax(scalar[:, 3])), "mean": float(np.nanmean(scalar[:, 3]))},
        "msd_xy_A2": {"lag_ps": lag_ps.tolist(), "values": msd.tolist()},
        "dxy_screening": fit_window,
        "blocks_5ps": blocks,
        "vacf": {"lag_ps": (np.arange(len(vacf)) * manifest["write_interval_steps"] * manifest["timestep_fs"] * 1.0e-3).tolist(), "values": vacf.tolist()},
        "hop_count": candidate_hops,
        "hop_definition": "candidate lateral Cr displacement > 1.5 A between saved frames; site mapping required for formal hops",
        "hfacc_status": "unavailable_no_heat_flux_recorded",
        "scientific_gate": "screening_only" if finite and complete_grid else "blocked",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def make_figures(results: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.linewidth": 1.5, "axes.grid": False})
    colors = {"baseline": "#245a9a", "ablation_correlation_1": "#b24d35", "ablation_correlation_3": "#5c8a52", "ablation_lmax_0": "#8b5a9b", "ablation_lmax_2": "#c58a2b"}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    for row in results:
        if row["scientific_gate"] == "blocked":
            continue
        label = f"{row['family']} seed{row['training_seed']}"
        x = np.asarray(row["msd_xy_A2"]["lag_ps"])
        y = np.asarray(row["msd_xy_A2"]["values"])
        axes[0].plot(x, y, lw=1.5, color=colors.get(row["family"], "#333333"), alpha=0.75, label=label)
        blocks = [b for b in row["blocks_5ps"] if b.get("Dxy_cm2_s") is not None]
        if blocks:
            axes[1].plot([b["block"] for b in blocks], [b["Dxy_cm2_s"] for b in blocks], marker="o", lw=1.3, color=colors.get(row["family"], "#333333"), alpha=0.75)
    axes[0].set(xlabel="Lag time (ps)", ylabel="Cr MSD$_{xy}$ (A$^2$)")
    axes[1].set(xlabel="5 ps block", ylabel="Screening D$_{xy}$ (cm$^2$ s$^{-1}$)")
    axes[0].legend(fontsize=6, frameon=True, ncol=2)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(output / f"screening_msd_dxy.{ext}", dpi=240 if ext == "png" else None)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = sorted(p for p in args.campaign_root.glob("runs/*") if (p / "run_manifest.json").exists() and (p / "trajectory.extxyz").exists())
    results = []
    for run in runs:
        try:
            results.append(analyze_run(run, args.output / "per_run" / run.name))
        except Exception as exc:
            results.append({"run": str(run), "scientific_gate": "blocked", "error": repr(exc)})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "screening_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    fields = ["family", "training_seed", "md_seed", "scientific_gate", "frames", "last_step", "duration_ps", "finite", "complete_step_grid", "hop_count", "dxy_screening"]
    with (args.output / "screening_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({key: json.dumps(row.get(key)) if key == "dxy_screening" else row.get(key) for key in fields})
    good = [row for row in results if row.get("scientific_gate") != "blocked"]
    if good:
        make_figures(good, args.output / "figures")
    print(json.dumps({"runs": len(results), "nonblocked": len(good)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
