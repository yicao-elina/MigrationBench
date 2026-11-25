#!/usr/bin/env python3
"""Unified, provenance-first post-processing for MigrationBench MACE MD.

The script reads a completed or partial extxyz trajectory and never edits the
campaign directory.  It writes auditable CSV/JSON products for
thermodynamics, Cr MSD/Dxy, VACF, and (when requested) the paper's
stress-based heat-flux/HFACF convention.  A trajectory without recorded
stress is not silently assigned a pressure or thermal conductivity: use
``--recompute-stress`` with the exact model to make those fields explicit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from ase import units
from ase.io import iread


PRESSURE_EV_A3_TO_GPA = 160.21766208
KB_EV_K = 8.617333262145e-5


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(value: float) -> bool:
    return bool(np.isfinite(value))


def temperature(atoms) -> float:
    return float(atoms.get_temperature())


def scalar_energy(atoms) -> float:
    for key in ("potential_energy_eV", "energy", "pot_energy_eV"):
        if key in atoms.info:
            return float(atoms.info[key])
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        return float("nan")


def max_force(atoms) -> float:
    if "max_force_eV_A" in atoms.info:
        return float(atoms.info["max_force_eV_A"])
    try:
        forces = np.asarray(atoms.get_forces())
    except Exception:
        return float("nan")
    return float(np.max(np.abs(forces))) if np.isfinite(forces).all() else float("nan")


def minimum_distance(atoms) -> float:
    if "min_distance_A" in atoms.info:
        return float(atoms.info["min_distance_A"])
    # The runner already samples this quantity.  Avoid an O(N^2) fallback in
    # long post-processing unless the source trajectory contains no value.
    return float("nan")


def stress_tensor(atoms) -> np.ndarray | None:
    try:
        s = np.asarray(atoms.get_stress(), dtype=float)
    except Exception:
        return None
    if s.shape == (6,):
        # ASE Voigt order: xx, yy, zz, yz, xz, xy.
        return np.array([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]])
    if s.shape == (3, 3):
        return s
    return None


def simplified_heat_flux(atoms, velocities: np.ndarray, stress: np.ndarray) -> np.ndarray:
    """Reproduce the original repository's documented simplified convention.

    This is not presented as a rigorous many-body per-atom heat-current
    implementation.  The exact formula and its caveat are recorded in the
    output manifest so that reference HFACF/κ values remain identifiable.
    """
    masses = atoms.get_masses()
    kinetic_energy = 0.5 * masses * np.sum(velocities * velocities, axis=1)
    kinetic_flux = np.sum(kinetic_energy[:, None] * velocities, axis=0)
    volume = float(atoms.get_volume())
    potential_flux = -np.dot(stress, np.mean(velocities, axis=0)) * volume
    return (kinetic_flux + potential_flux) / volume


def unwrap_fractional(frac: np.ndarray) -> np.ndarray:
    out = np.empty_like(frac)
    out[0] = frac[0]
    for i in range(1, len(frac)):
        delta = frac[i] - frac[i - 1]
        delta -= np.rint(delta)
        out[i] = out[i - 1] + delta
    return out


def autocorr_fft(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    centered = values - values.mean(axis=0)
    n = len(centered)
    if n < 2:
        return np.full((n, values.shape[1]), np.nan)
    fft = np.fft.fft(centered, n=2 * n - 1, axis=0)
    corr = np.fft.ifft(np.abs(fft) ** 2, axis=0).real[:n]
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = corr / corr[0]
    return corr


def multi_origin_msd(pos: np.ndarray, max_lag: int, origin_stride: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_lag = min(max_lag, len(pos) - 1)
    lags = np.arange(1, max_lag + 1, dtype=int)
    msd3 = np.full(len(lags), np.nan)
    msdxy = np.full(len(lags), np.nan)
    for index, lag in enumerate(lags):
        origins = np.arange(0, len(pos) - lag, origin_stride, dtype=int)
        if len(origins) == 0:
            continue
        displacement = pos[origins + lag] - pos[origins]
        msd3[index] = float(np.mean(np.sum(displacement * displacement, axis=2)))
        msdxy[index] = float(np.mean(np.sum(displacement[:, :, :2] ** 2, axis=2)))
    return lags, msd3, msdxy


def fit_window(time_ps: np.ndarray, values: np.ndarray, lo: float, hi: float, dimensions: float = 4.0) -> dict:
    mask = np.isfinite(values) & (time_ps >= lo) & (time_ps <= hi)
    if int(mask.sum()) < 5:
        return {"n": int(mask.sum()), "slope_A2_per_ps": None, "D_cm2_s": None, "window_ps": [lo, hi]}
    slope, intercept = np.polyfit(time_ps[mask], values[mask], 1)
    prediction = slope * time_ps[mask] + intercept
    ss_total = float(np.sum((values[mask] - np.mean(values[mask])) ** 2))
    r2 = None if ss_total == 0 else 1.0 - float(np.sum((values[mask] - prediction) ** 2)) / ss_total
    return {
        "n": int(mask.sum()),
        "slope_A2_per_ps": float(slope),
        "intercept_A2": float(intercept),
        "D_cm2_s": float(slope / dimensions * 1.0e-4),
        "r2": r2,
        "window_ps": [lo, hi],
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--recompute-stress", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-lag-frames", type=int, default=4000)
    parser.add_argument("--origin-stride", type=int, default=10)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    status = json.loads(args.status.read_text())
    if args.recompute_stress and args.model is None:
        parser.error("--recompute-stress requires --model")

    calculator = None
    stress_mode = "unavailable_no_stress_recorded"
    if args.recompute_stress:
        from mace.calculators import MACECalculator

        calculator = MACECalculator(model_paths=str(args.model), device=args.device, default_dtype="float64")
        stress_mode = "recomputed_stress_simplified_heat_flux_v1"

    positions, velocities, steps = [], [], []
    energies, kinetic_energies, temperatures, forces, min_distances = [], [], [], [], []
    pressures, heat_flux = [], []
    species = None
    cell = None
    pbc = None
    n_stress = 0
    n_frames = 0

    for atoms in iread(str(args.trajectory), format="extxyz"):
        n_frames += 1
        symbols = tuple(atoms.get_chemical_symbols())
        if species is None:
            species = symbols
            cell = np.asarray(atoms.cell.array, dtype=float)
            pbc = tuple(bool(x) for x in atoms.pbc)
        if symbols != species:
            raise RuntimeError(f"species identity changed at frame {n_frames - 1}")
        if len(atoms) != 2050 or pbc != (True, True, True):
            raise RuntimeError(f"unexpected identity/PBC at frame {n_frames - 1}: {len(atoms)}/{pbc}")
        if not np.allclose(atoms.cell.array, cell, rtol=0, atol=1e-8):
            raise RuntimeError(f"cell changed at frame {n_frames - 1}")
        if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.get_momenta()).all():
            raise RuntimeError(f"non-finite geometry/momenta at frame {n_frames - 1}")

        if calculator is not None:
            atoms.calc = calculator
        mask = np.asarray(symbols) == "Cr"
        mass = atoms.get_masses()[mask, None]
        positions.append(atoms.positions[mask].copy())
        velocities.append(atoms.get_momenta()[mask] / mass)
        step = int(atoms.info.get("step", (n_frames - 1) * int(manifest.get("write_interval_steps", 200))))
        steps.append(step)
        energies.append(scalar_energy(atoms))
        kinetic_energies.append(float(atoms.get_kinetic_energy()))
        temperatures.append(temperature(atoms))
        forces.append(max_force(atoms))
        min_distances.append(minimum_distance(atoms))

        stress = stress_tensor(atoms)
        if stress is not None and np.isfinite(stress).all():
            n_stress += 1
            pressures.append(float(-np.trace(stress) / 3.0 * PRESSURE_EV_A3_TO_GPA))
            if calculator is None:
                stress_mode = "trajectory_recorded_stress_simplified_heat_flux_v1"
            heat_flux.append(simplified_heat_flux(atoms, atoms.get_momenta() / atoms.get_masses()[:, None], stress))
        else:
            pressures.append(float("nan"))
            heat_flux.append(np.full(3, np.nan))

    if n_frames < 2:
        raise RuntimeError("trajectory has fewer than two frames")
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    steps = np.asarray(steps, dtype=int)
    energies = np.asarray(energies, dtype=float)
    kinetic_energies = np.asarray(kinetic_energies, dtype=float)
    temperatures = np.asarray(temperatures, dtype=float)
    forces = np.asarray(forces, dtype=float)
    min_distances = np.asarray(min_distances, dtype=float)
    pressures = np.asarray(pressures, dtype=float)
    heat_flux = np.asarray(heat_flux, dtype=float)

    keep = np.r_[True, np.diff(steps) > 0]
    positions, velocities, steps = positions[keep], velocities[keep], steps[keep]
    energies, kinetic_energies, temperatures, forces = energies[keep], kinetic_energies[keep], temperatures[keep], forces[keep]
    min_distances, pressures, heat_flux = min_distances[keep], pressures[keep], heat_flux[keep]

    frac = np.einsum("nij,jk->nik", positions, np.linalg.inv(cell))
    unwrapped = unwrap_fractional(frac) @ cell
    timestep_fs = float(manifest.get("timestep_fs", 1.0))
    write_interval = int(manifest.get("write_interval_steps", 200))
    time_ps = steps * timestep_fs * 1.0e-3
    relative_time_ps = (steps - steps[0]) * timestep_fs * 1.0e-3
    frame_ps = write_interval * timestep_fs * 1.0e-3

    lag, msd3, msdxy = multi_origin_msd(unwrapped, args.max_lag_frames, args.origin_stride)
    lag_ps = lag * frame_ps
    centered = velocities - velocities.mean(axis=1, keepdims=True)
    vacf_raw = np.asarray([
        np.mean(centered * centered) if k == 0 else np.mean(centered[:-k] * centered[k:])
        for k in range(min(len(centered), args.max_lag_frames))
    ])
    vacf = vacf_raw / vacf_raw[0] if len(vacf_raw) and finite(vacf_raw[0]) and vacf_raw[0] != 0 else np.full_like(vacf_raw, np.nan)

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_csv(
        out / "timeseries.csv",
        ["step", "time_ps", "relative_time_ps", "potential_energy_eV", "kinetic_energy_eV", "total_energy_eV", "temperature_K", "max_force_eV_A", "min_distance_A", "pressure_GPa"],
        [
            {
                "step": int(step),
                "time_ps": float(t),
                "relative_time_ps": float(rt),
                "potential_energy_eV": float(e),
                "kinetic_energy_eV": float(ke),
                "total_energy_eV": float(e + ke),
                "temperature_K": float(temp),
                "max_force_eV_A": float(force),
                "min_distance_A": float(dist),
                "pressure_GPa": float(pressure),
            }
            for step, t, rt, e, ke, temp, force, dist, pressure in zip(
                steps,
                time_ps,
                relative_time_ps,
                energies,
                kinetic_energies,
                temperatures,
                forces,
                min_distances,
                pressures,
            )
        ],
    )

    write_csv(out / "msd.csv", ["lag_ps", "msd_3d_A2", "msd_xy_A2"], [{"lag_ps": float(t), "msd_3d_A2": float(a), "msd_xy_A2": float(b)} for t, a, b in zip(lag_ps, msd3, msdxy)])
    write_csv(out / "vacf.csv", ["lag_ps", "vacf_normalized"], [{"lag_ps": float(i * frame_ps), "vacf_normalized": float(value)} for i, value in enumerate(vacf)])

    hfacf = np.full(len(heat_flux), np.nan)
    kappa = np.full(len(heat_flux), np.nan)
    hfacf_available = bool(len(heat_flux) and np.isfinite(heat_flux).all())
    if hfacf_available:
        hfacf_components = autocorr_fft(heat_flux)
        hfacf = np.mean(hfacf_components, axis=1)
        j2 = float(np.mean(heat_flux * heat_flux))
        volume = float(np.linalg.det(cell))
        prefactor = volume / (KB_EV_K * float(np.nanmean(temperatures)) ** 2) / 3.0
        # 200 fs is the actual saved-frame interval; conversion follows the
        # original script's eV/fs/A to W/m/K factor.
        integral = np.concatenate([[0.0], np.cumsum((hfacf[:-1] + hfacf[1:]) * 0.5 * frame_ps * 1000.0)]) * j2
        kappa = prefactor * integral * 1.60218e6
    write_csv(out / "hfacf.csv", ["lag_ps", "hfacf_normalized"], [{"lag_ps": float(i * frame_ps), "hfacf_normalized": float(value)} for i, value in enumerate(hfacf)])
    write_csv(out / "kappa.csv", ["time_ps", "kappa_W_mK"], [{"time_ps": float(t), "kappa_W_mK": float(value)} for t, value in zip(time_ps - time_ps[0], kappa)])

    finite_gate = bool(np.isfinite(positions).all() and np.isfinite(velocities).all() and np.isfinite(energies).all() and np.isfinite(temperatures).all() and np.isfinite(forces).all())
    complete = status.get("status") == "complete" and int(status.get("last_step", -1)) >= int(manifest.get("steps_requested", 10**9))
    result = {
        "trajectory": str(args.trajectory),
        "trajectory_sha256": sha256(args.trajectory),
        "manifest": str(args.manifest),
        "status": status,
        "model": str(args.model) if args.model else manifest.get("model"),
        "model_sha256": sha256(args.model) if args.model and args.model.exists() else manifest.get("model_sha256"),
        "frames": int(len(steps)),
        "start_step": int(steps[0]),
        "last_step": int(steps[-1]),
        "duration_ps": float((steps[-1] - steps[0]) * timestep_fs * 1.0e-3),
        "natoms": int(len(species)),
        "pbc": list(pbc),
        "cell_A": np.asarray(cell).tolist(),
        "finite_gate": finite_gate,
        "complete_step_gate": complete,
        "scientific_gate": "screening_or_production_complete" if finite_gate and complete else "blocked_or_incomplete",
        "temperature_K": {"min": float(np.nanmin(temperatures)), "mean": float(np.nanmean(temperatures)), "max": float(np.nanmax(temperatures))},
        "max_force_eV_A": {"max": float(np.nanmax(forces)), "mean": float(np.nanmean(forces))},
        "pressure_GPa": {"finite_frames": int(np.isfinite(pressures).sum()), "mean": float(np.nanmean(pressures)) if np.isfinite(pressures).any() else None},
        "msd": {
            "definition": "Cr multi-origin MSD; unwrapped fixed-cell coordinates; origin_stride=%d" % args.origin_stride,
            "Dxy_200_500_ps": fit_window(lag_ps, msdxy, 200.0, 500.0, 4.0),
            "Dxy_500_900_ps": fit_window(lag_ps, msdxy, 500.0, 900.0, 4.0),
            "D3D_200_500_ps": fit_window(lag_ps, msd3, 200.0, 500.0, 6.0),
            "D3D_500_900_ps": fit_window(lag_ps, msd3, 500.0, 900.0, 6.0),
        },
        "vacf": {"definition": "Cr velocity VACF with per-frame center-of-mass velocity removed", "frames": int(len(vacf)), "initial": float(vacf[0]) if len(vacf) else None},
        "hop_count": {"status": "candidate_only_site_mapping_required", "threshold_A": 1.5, "events_between_saved_frames": int(np.sum(np.linalg.norm(positions[1:] - positions[:-1], axis=2) > 1.5))},
        "heat_flux": {"status": "available" if hfacf_available else stress_mode, "stress_frames": int(n_stress), "formula": "repository_simplified_stress_based_heat_flux_v1", "kappa_final_W_mK": float(kappa[-1]) if hfacf_available else None},
        "uncertainty": {"status": "single_replica_until_all_selected_runs_complete", "required": "block and replica uncertainty; 1 ns does not imply transport convergence"},
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "postprocess_manifest.json").write_text(json.dumps({"schema": "migrationbench_unified_md_postprocess_v1", "source_script": str(Path(__file__).resolve()), "source_script_sha256": sha256(Path(__file__).resolve()), "stress_mode": stress_mode, "recompute_stress": bool(args.recompute_stress), "device": args.device}, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
