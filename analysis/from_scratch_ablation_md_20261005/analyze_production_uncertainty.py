#!/usr/bin/env python3
"""Audit production-window, block, and family-replica uncertainty.

The script is intentionally read-only with respect to MD campaign roots.  It
expects completed per-run post-processing directories containing metrics.json,
timeseries.csv and the trajectory path recorded in metrics.json.  A completed
1 ns run is accepted as a trajectory artifact, not as proof of transport
convergence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from ase.io import iread


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_run_artifact(
    run_dir: Path,
    recorded: str | None,
    filename: str,
    runs_root: Path | None,
) -> Path:
    """Resolve a source artifact without depending on one directory layout."""
    candidates = []
    if recorded:
        candidates.append(Path(recorded))
    if runs_root is not None:
        candidates.append(runs_root / run_dir.name / filename)
    # Backward-compatible fallback for the original remote layout.
    candidates.append(run_dir.parent.parent.parent / "runs" / run_dir.name / filename)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{filename} unavailable for {run_dir}; searched: {searched}")


def unwrap(frac: np.ndarray) -> np.ndarray:
    out = np.empty_like(frac)
    out[0] = frac[0]
    for i in range(1, len(frac)):
        delta = frac[i] - frac[i - 1]
        delta -= np.rint(delta)
        out[i] = out[i - 1] + delta
    return out


def multi_origin_msd(pos: np.ndarray, max_lag: int, origin_stride: int = 10) -> tuple[np.ndarray, np.ndarray]:
    max_lag = min(max_lag, len(pos) - 1)
    lags = np.arange(1, max_lag + 1, dtype=int)
    values = np.full(len(lags), np.nan)
    for index, lag in enumerate(lags):
        origins = np.arange(0, len(pos) - lag, origin_stride, dtype=int)
        if len(origins):
            displacement = pos[origins + lag] - pos[origins]
            values[index] = float(np.mean(np.sum(displacement[:, :, :2] ** 2, axis=2)))
    return lags, values


def fit_dxy(time_ps: np.ndarray, msd: np.ndarray, low: float, high: float) -> dict:
    mask = np.isfinite(time_ps) & np.isfinite(msd) & (time_ps >= low) & (time_ps <= high)
    if int(mask.sum()) < 5:
        return {"n": int(mask.sum()), "Dxy_cm2_s": None, "slope_A2_per_ps": None, "r2": None, "fit_window_ps": [low, high]}
    slope, intercept = np.polyfit(time_ps[mask], msd[mask], 1)
    pred = slope * time_ps[mask] + intercept
    total = float(np.sum((msd[mask] - np.mean(msd[mask])) ** 2))
    r2 = None if total == 0 else 1.0 - float(np.sum((msd[mask] - pred) ** 2)) / total
    return {"n": int(mask.sum()), "Dxy_cm2_s": float(slope / 4.0 * 1e-4), "slope_A2_per_ps": float(slope), "intercept_A2": float(intercept), "r2": r2, "fit_window_ps": [low, high]}


def load_cr_positions(trajectory: Path, write_interval_steps: int = 200, timestep_fs: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions, steps = [], []
    species = None
    cell = None
    for index, atoms in enumerate(iread(str(trajectory), format="extxyz")):
        current_species = tuple(atoms.get_chemical_symbols())
        if species is None:
            species = current_species
            cell = np.asarray(atoms.cell.array, dtype=float)
        if len(atoms) != 2050 or tuple(bool(x) for x in atoms.pbc) != (True, True, True):
            raise RuntimeError(f"identity/PBC failure at frame {index}")
        if current_species != species or not np.allclose(atoms.cell.array, cell, atol=1e-8, rtol=0):
            raise RuntimeError(f"species/cell failure at frame {index}")
        if not np.isfinite(atoms.positions).all():
            raise RuntimeError(f"non-finite positions at frame {index}")
        mask = np.asarray(current_species) == "Cr"
        positions.append(atoms.positions[mask].copy())
        steps.append(int(atoms.info.get("step", index * write_interval_steps)))
    steps = np.asarray(steps, dtype=int)
    keep = np.r_[True, np.diff(steps) > 0]
    positions = np.asarray(positions)[keep]
    steps = steps[keep]
    frac = np.einsum("nij,jk->nik", positions, np.linalg.inv(cell))
    return unwrap(frac) @ cell, steps, cell


def block_statistics(rows: list[dict[str, str]], burnin_ps: float, n_blocks: int) -> list[dict]:
    numeric = {}
    for key in ("potential_energy_eV", "kinetic_energy_eV", "total_energy_eV", "temperature_K", "max_force_eV_A", "min_distance_A", "pressure_GPa"):
        numeric[key] = np.asarray([as_float(r.get(key)) for r in rows], dtype=float)
    time = np.asarray([as_float(r.get("relative_time_ps")) for r in rows], dtype=float)
    keep = np.isfinite(time) & (time >= burnin_ps)
    indices = np.where(keep)[0]
    if len(indices) < n_blocks:
        return []
    bounds = np.linspace(0, len(indices), n_blocks + 1, dtype=int)
    output = []
    for block in range(n_blocks):
        selected = indices[bounds[block]:bounds[block + 1]]
        item = {"block": block + 1, "start_ps": float(time[selected[0]]), "end_ps": float(time[selected[-1]])}
        for key, values in numeric.items():
            subset = values[selected]
            finite = subset[np.isfinite(subset)]
            item[f"{key}_mean"] = float(np.mean(finite)) if len(finite) else None
            item[f"{key}_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else None
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="completed per-run postprocess directories")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--runs-root", type=Path, help="optional campaign runs directory for relocated postprocess outputs")
    ap.add_argument("--burnin-ps", type=float, default=100.0)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--origin-stride", type=int, default=10)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    run_rows, thermo_block_rows, dxy_block_rows = [], [], []
    for metrics_path in sorted(args.root.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        if not metrics.get("complete_step_gate") or not metrics.get("finite_gate"):
            continue
        run_dir = metrics_path.parent
        trajectory = resolve_run_artifact(run_dir, metrics.get("trajectory"), "trajectory.extxyz", args.runs_root)
        manifest_path = resolve_run_artifact(run_dir, metrics.get("manifest"), "run_manifest.json", args.runs_root)
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        family = manifest.get("family", "")
        seed = manifest.get("training_seed", "")
        if not family:
            parts = run_dir.name.split("__")
            family = parts[0]
            seed = seed or next((part.removeprefix("seed") for part in parts if part.startswith("seed")), "")
        timeseries = read_csv(run_dir / "timeseries.csv")
        blocks = block_statistics(timeseries, args.burnin_ps, args.n_blocks)
        positions, steps, _ = load_cr_positions(trajectory, int(manifest.get("write_interval_steps", 200)), float(manifest.get("timestep_fs", 1.0)))
        frame_ps = float(manifest.get("write_interval_steps", 200)) * float(manifest.get("timestep_fs", 1.0)) * 1e-3
        time_ps = (steps - steps[0]) * float(manifest.get("timestep_fs", 1.0)) * 1e-3
        production = time_ps >= args.burnin_ps
        pos = positions[production]
        if len(pos) < args.n_blocks + 2:
            raise RuntimeError(f"not enough production frames for {run_dir}")
        lag, msd = multi_origin_msd(pos, min(4000, len(pos) - 1), args.origin_stride)
        dxy_global = fit_dxy(lag * frame_ps, msd, 200.0, 500.0)
        dxy_late = fit_dxy(lag * frame_ps, msd, 500.0, 900.0)
        bounds = np.linspace(0, len(pos), args.n_blocks + 1, dtype=int)
        dxy_blocks = []
        for block in range(args.n_blocks):
            segment = pos[bounds[block]:bounds[block + 1]]
            if len(segment) < 12:
                continue
            local_lag, local_msd = multi_origin_msd(segment, min(1000, len(segment) - 1), args.origin_stride)
            local_time = local_lag * frame_ps
            duration = float(local_time[-1]) if len(local_time) else 0.0
            estimate = fit_dxy(local_time, local_msd, max(frame_ps, 0.2 * duration), 0.8 * duration)
            dxy_blocks.append({"run": run_dir.name, "family": family, "training_seed": seed, "block": block + 1, **estimate})
        thermo_block_rows.extend({"run": run_dir.name, "family": family, "training_seed": seed, **item} for item in blocks)
        run_rows.append({
            "run": run_dir.name,
            "family": family,
            "training_seed": seed,
            "duration_ps": metrics.get("duration_ps"),
            "burnin_ps": args.burnin_ps,
            "production_ps": float(time_ps[-1] - args.burnin_ps),
            "Dxy_200_500_cm2_s": dxy_global.get("Dxy_cm2_s"),
            "Dxy_500_900_cm2_s": dxy_late.get("Dxy_cm2_s"),
            "Dxy_block_mean_cm2_s": float(np.nanmean([x["Dxy_cm2_s"] for x in dxy_blocks])) if dxy_blocks and any(x["Dxy_cm2_s"] is not None for x in dxy_blocks) else None,
            "Dxy_block_std_cm2_s": float(np.nanstd([x["Dxy_cm2_s"] for x in dxy_blocks], ddof=1)) if len([x for x in dxy_blocks if x["Dxy_cm2_s"] is not None]) > 1 else None,
            "temperature_mean_K": metrics.get("temperature_K", {}).get("mean"),
            "pressure_mean_GPa": metrics.get("pressure_GPa", {}).get("mean"),
            "kappa_final_W_mK": metrics.get("heat_flux", {}).get("kappa_final_W_mK"),
            "HFACF_status": metrics.get("heat_flux", {}).get("status"),
            "convergence_status": "complete_trajectory_transport_still_requires_window_and_replica_audit",
        })
        dxy_block_rows.extend(dxy_blocks)

    replica_rows = []
    for family in sorted({r["family"] for r in run_rows}):
        family_rows = [r for r in run_rows if r["family"] == family]
        for metric in ("Dxy_200_500_cm2_s", "Dxy_500_900_cm2_s", "Dxy_block_mean_cm2_s", "kappa_final_W_mK"):
            values = np.asarray([as_float(r.get(metric)) for r in family_rows], dtype=float)
            values = values[np.isfinite(values)]
            replica_rows.append({"family": family, "metric": metric, "n_replicas": int(len(values)), "mean": float(np.mean(values)) if len(values) else None, "std": float(np.std(values, ddof=1)) if len(values) > 1 else None, "sem": float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None, "status": "replica_uncertainty_available" if len(values) > 1 else "n=1_no_replica_uncertainty"})

    write_csv(args.output / "production_run_summary.csv", run_rows)
    write_csv(args.output / "production_block_statistics.csv", thermo_block_rows)
    write_csv(args.output / "production_dxy_blocks.csv", dxy_block_rows)
    write_csv(args.output / "production_replica_summary.csv", replica_rows)
    report = {
        "burnin_ps": args.burnin_ps,
        "n_blocks": args.n_blocks,
        "origin_stride": args.origin_stride,
        "runs": len(run_rows),
        "scientific_status": "FORMAL" if run_rows else "BLOCKED",
        "blocked_reason": None if run_rows else "all candidate metrics failed complete_step_gate or finite_gate acceptance",
        "family_replica_policy": "training-seed replicas are summarized within family; baseline n=1 remains explicitly underpowered",
        "transport_gate": "1 ns completion is not transport convergence; use window, block and replica evidence",
        "output_files": ["production_run_summary.csv", "production_block_statistics.csv", "production_dxy_blocks.csv", "production_replica_summary.csv"],
    }
    (args.output / "uncertainty_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "uncertainty_report.md").write_text(
        "# Production uncertainty audit\n\n"
        f"- Balance/equilibration exclusion: first `{args.burnin_ps:g} ps` removed.\n"
        f"- Production blocks: `{args.n_blocks}`.\n"
        f"- Cr MSD/Dxy: multi-origin, fixed-cell unwrapped coordinates, origin stride `{args.origin_stride}`; Dxy uses the 2D Einstein factor 4.\n"
        "- Family replica standard errors are reported only when at least two training-seed replicas pass the production gates.\n"
        "- A completed 1 ns trajectory remains exploratory until fit-window, block and replica estimates agree.\n"
        + ("- Status: BLOCKED; zero candidate runs passed the complete-step and finite-state production gates, so no block/replica estimate is reported.\n" if not run_rows else "")
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
