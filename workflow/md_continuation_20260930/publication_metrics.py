#!/usr/bin/env python
"""Auditable four-model MD metrics for the isolated publication analysis release.

The script never edits the MD campaign.  It reads manifests/trajectories and
writes JSON/CSV artifacts to a caller-provided output directory.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from ase import units
from ase.io import iread, read


MODEL_DIRS = [
    "0-foundation-omat",
    "1-from-scratch",
    "2-naive-fine-tuning",
    "3-multi_T-fine-tuning",
]

DISPLAY_NAMES = {
    "0-foundation-omat": "Foundation-OMAT",
    "1-from-scratch": "From-scratch",
    "2-naive-fine-tuning": "FT-600K",
    "3-multi_T-fine-tuning": "FT-MultiT",
}


def display_name(raw_id):
    """Return the manuscript-facing name without changing raw provenance IDs."""
    return DISPLAY_NAMES.get(raw_id, raw_id)


def sha256(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def finite_energy(atoms):
    value = atoms.info.get("energy")
    if value is None and atoms.calc is not None:
        value = atoms.calc.results.get("energy")
    return (float(value) if value is not None else float("nan"))


def temperature(atoms):
    if "temperature" in atoms.info:
        return float(atoms.info["temperature"])
    p = atoms.get_momenta()
    m = atoms.get_masses()[:, None]
    kinetic = float(0.5 * np.sum(p * p / m))
    return 2.0 * kinetic / (3.0 * len(atoms) * units.kB)


def frame_metrics(atoms, cr_mask=None, min_distance=False):
    if cr_mask is None:
        cr_mask = np.array(atoms.get_chemical_symbols()) == "Cr"
    forces = atoms.get_forces()
    dmin = float("nan")
    if min_distance:
        d = atoms.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        dmin = float(np.nanmin(d))
    return {
        "natoms": int(len(atoms)),
        "pbc": [bool(x) for x in atoms.pbc],
        "finite_positions": bool(np.isfinite(atoms.positions).all()),
        "finite_momenta": bool(np.isfinite(atoms.get_momenta()).all()),
        "finite_energy": bool(np.isfinite(finite_energy(atoms))),
        "energy_eV": finite_energy(atoms),
        "finite_forces": bool(np.isfinite(forces).all()),
        "max_force_eV_A": float(np.nanmax(np.abs(forces))),
        "temperature_K": temperature(atoms),
        "min_distance_A": dmin,
    }


def manifest_for(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    status = json.loads((run_dir / "status.json").read_text())
    return manifest, status


def audit_model(run_dir, include_trajectory_hash=True):
    run_dir = Path(run_dir)
    manifest, status = manifest_for(run_dir)
    paths = {
        "input": Path(manifest["input"]),
        "model": Path(manifest["model"]),
        "segment": run_dir / Path(manifest["segment"]).name,
        "checkpoint": run_dir / "latest_restart.extxyz",
    }
    out = {
        "model_raw_id": run_dir.name,
        "model": display_name(run_dir.name),
        "run_dir": str(run_dir),
        "manifest": manifest,
        "status": status,
        "files": {},
    }
    for key, path in paths.items():
        item = {"path": str(path), "exists": path.exists()}
        if path.exists():
            st = path.stat()
            item.update({"size": st.st_size, "mtime": st.st_mtime})
            if key in {"input", "model"} or include_trajectory_hash:
                item["sha256"] = sha256(path)
        out["files"][key] = item
    try:
        atoms = read(paths["checkpoint"], index=-1)
        symbols = tuple(atoms.get_chemical_symbols())
        out["checkpoint_validation"] = {
            "natoms": len(atoms),
            "pbc": [bool(x) for x in atoms.pbc],
            "species_sha256": hashlib.sha256(" ".join(symbols).encode()).hexdigest(),
            "finite_positions": bool(np.isfinite(atoms.positions).all()),
            "finite_momenta": bool(np.isfinite(atoms.get_momenta()).all()),
            "finite_energy": bool(np.isfinite(finite_energy(atoms))),
            "finite_forces": bool(np.isfinite(atoms.get_forces()).all()),
            "max_force_eV_A": float(np.nanmax(np.abs(atoms.get_forces()))),
            "temperature_K": temperature(atoms),
        }
    except Exception as exc:
        out["checkpoint_validation"] = {"error": repr(exc)}
    return out


def unwrap(frac, cell):
    result = np.empty_like(frac)
    result[0] = frac[0]
    for i in range(1, len(frac)):
        df = frac[i] - frac[i - 1]
        df -= np.rint(df)
        result[i] = result[i - 1] + df
    return result @ cell


def fit(t, y, lo, hi):
    mask = np.isfinite(y) & (t >= lo) & (t <= hi)
    if int(mask.sum()) < 5:
        return {"n": int(mask.sum()), "slope_A2_per_ps": None, "D_cm2_s": None}
    slope, intercept = np.polyfit(t[mask], y[mask], 1)
    return {
        "n": int(mask.sum()),
        "slope_A2_per_ps": float(slope),
        "intercept_A2": float(intercept),
        "D_cm2_s": float(slope / 4.0 * 1e-4),
    }


def load_trajectory(original, segment, interval=200, min_distance_stride=100):
    records = []
    cr_xyz, cr_vel = [], []
    energies, temps, forces, distances, steps = [], [], [], [], []
    species = None
    cell = None
    frame_index = 0
    for path, offset in [(original, 0), (segment, None)]:
        for atoms in iread(str(path), format="extxyz"):
            if species is None:
                species = tuple(atoms.get_chemical_symbols())
                cell = np.asarray(atoms.cell.array)
            elif tuple(atoms.get_chemical_symbols()) != species:
                raise ValueError(f"species changed at frame {frame_index}")
            info = frame_metrics(
                atoms,
                min_distance=(frame_index % min_distance_stride == 0),
            )
            step = atoms.info.get("step")
            if step is None:
                step = offset if offset is not None else 0
                step += frame_index * interval
            mask = np.array(atoms.get_chemical_symbols()) == "Cr"
            mass = atoms.get_masses()[mask][:, None]
            cr_xyz.append(atoms.positions[mask].copy())
            cr_vel.append(atoms.get_momenta()[mask] / mass)
            energies.append(info["energy_eV"])
            temps.append(info["temperature_K"])
            forces.append(info["max_force_eV_A"])
            distances.append(info["min_distance_A"])
            steps.append(int(step))
            frame_index += 1
    steps = np.asarray(steps)
    keep = np.r_[True, np.diff(steps) > 0]
    return {
        "xyz": np.asarray(cr_xyz)[keep],
        "vel": np.asarray(cr_vel)[keep],
        "energy": np.asarray(energies)[keep],
        "temperature": np.asarray(temps)[keep],
        "max_force": np.asarray(forces)[keep],
        "min_distance": np.asarray(distances)[keep],
        "steps": steps[keep],
        "cell": cell,
        "species": species,
    }


def msd_series(pos, max_lag=4000, origin_stride=10):
    max_lag = min(max_lag, len(pos) - 1)
    lag = np.arange(1, max_lag + 1)
    msd = np.full(max_lag, np.nan)
    for j, k in enumerate(lag):
        origins = np.arange(0, len(pos) - k, origin_stride)
        if len(origins):
            dr = pos[origins + k] - pos[origins]
            msd[j] = float(np.mean(np.sum(dr * dr, axis=2)))
    return lag, msd


def block_slopes(pos, frame_ps=0.2, n_blocks=8, origin_stride=10):
    n = len(pos)
    bounds = np.linspace(0, n, n_blocks + 1, dtype=int)
    rows = []
    for b in range(n_blocks):
        lo, hi = bounds[b], bounds[b + 1]
        lag, msd = msd_series(pos, max_lag=min(4000, n - 1), origin_stride=origin_stride)
        t = lag * frame_ps
        for wname, wlo, whi in [("200_500_ps", 200, 500), ("500_900_ps", 500, 900)]:
            # Restrict origins to the block and recompute the curve for this block.
            values = np.full(len(lag), np.nan)
            for j, k in enumerate(lag):
                origins = np.arange(lo, min(hi, n - k), origin_stride)
                if len(origins):
                    dr = pos[origins + k] - pos[origins]
                    values[j] = float(np.mean(np.sum(dr * dr, axis=2)))
            result = fit(t, values, wlo, whi)
            rows.append({"block": b, "window": wname, **result})
    return rows


def hop_counts(pos, frame_ps=0.2, hop_interval_ps=1.0):
    stride = max(1, int(round(hop_interval_ps / frame_ps)))
    dr = pos[stride:] - pos[:-stride]
    disp = np.sqrt(np.sum(dr * dr, axis=2))
    out = {"interval_ps": hop_interval_ps, "thresholds_A": {}}
    for threshold in (1.5, 2.0, 2.5, 3.0):
        hit = disp > threshold
        out["thresholds_A"][str(threshold)] = {
            "total_events": int(hit.sum()),
            "frames_with_event": int(np.any(hit, axis=1).sum()),
            "unique_Cr_atoms": int(np.any(hit, axis=0).sum()),
        }
    return out


def analyze_stable(run_dir, output_dir):
    manifest, status = manifest_for(run_dir)
    data = load_trajectory(manifest["input"], manifest["segment"])
    pos = unwrap(
        np.einsum("nij,jk->nik", data["xyz"], np.linalg.inv(data["cell"])),
        data["cell"],
    )
    time_ps = (data["steps"] - data["steps"][0]) * 0.001
    lag, msd = msd_series(pos)
    lag_ps = lag * 0.2
    v = data["vel"] - data["vel"].mean(axis=1, keepdims=True)
    vacf = np.array([
        np.mean(v * v) if k == 0 else np.mean(v[:-k] * v[k:])
        for k in range(min(len(v), 2000))
    ])
    vacf /= vacf[0] if vacf[0] else 1.0
    result = {
        "model_raw_id": run_dir.name,
        "model": DISPLAY_NAMES.get(run_dir.name, run_dir.name),
        "frames": int(len(data["steps"])),
        "start_step": int(data["steps"][0]),
        "last_step": int(data["steps"][-1]),
        "time_ps": float(time_ps[-1]),
        "natoms": int(len(data["species"])),
        "pbc": [True, True, True],
        "energy_finite": bool(np.isfinite(data["energy"]).all()),
        "force_finite": bool(np.isfinite(data["max_force"]).all()),
        "temperature_finite": bool(np.isfinite(data["temperature"]).all()),
        "max_force_eV_A": float(np.nanmax(data["max_force"])),
        "min_distance_A": float(np.nanmin(data["min_distance"])),
        "temperature_K": {
            "min": float(np.nanmin(data["temperature"])),
            "mean": float(np.nanmean(data["temperature"])),
            "max": float(np.nanmax(data["temperature"])),
        },
        "Dxy": {
            "200_500_ps": fit(lag_ps, msd, 200, 500),
            "500_900_ps": fit(lag_ps, msd, 500, 900),
        },
        "vacf": {"frames": int(len(vacf)), "initial": float(vacf[0])},
        "block_slopes": block_slopes(pos),
        "hop_counts": hop_counts(pos),
        "status": status,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output_dir / "timeseries.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "time_ps", "energy_eV", "temperature_K", "max_force_eV_A", "min_distance_A"])
        for row in zip(data["steps"], time_ps, data["energy"], data["temperature"], data["max_force"], data["min_distance"]):
            w.writerow(row)
    with (output_dir / "msd.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["lag_ps", "msd_A2"])
        w.writerows(zip(lag_ps, msd))
    with (output_dir / "vacf.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["lag_ps", "vacf_normalized"])
        w.writerows((i * 0.2, x) for i, x in enumerate(vacf))
    return result


def scan_failure(run_dir, output_dir, sample_stride=10):
    manifest, status = manifest_for(run_dir)
    data = load_trajectory(manifest["input"], manifest["segment"], min_distance_stride=sample_stride)
    rows = []
    for i, step in enumerate(data["steps"]):
        if i % sample_stride == 0 or i == len(data["steps"]) - 1:
            rows.append({
                "frame": i, "step": int(step), "time_ps": float((step - data["steps"][0]) * 0.001),
                "energy_eV": float(data["energy"][i]), "temperature_K": float(data["temperature"][i]),
                "max_force_eV_A": float(data["max_force"][i]), "min_distance_A": float(data["min_distance"][i]),
            })
    thresholds = {"temperature_K": 5000.0, "max_force_eV_A": 100.0, "min_distance_A": 0.5}
    first = None
    for i in range(len(data["steps"])):
        d = float(data["min_distance"][i])
        bad = (data["temperature"][i] > thresholds["temperature_K"] or
               data["max_force"][i] > thresholds["max_force_eV_A"] or
               (np.isfinite(d) and d < thresholds["min_distance_A"]) or
               not np.isfinite(data["temperature"][i]) or
               not np.isfinite(data["max_force"][i]))
        if bad:
            first = {"frame": i, "step": int(data["steps"][i]),
                     "time_ps": float((data["steps"][i] - data["steps"][0]) * 0.001),
                     "temperature_K": float(data["temperature"][i]),
                     "max_force_eV_A": float(data["max_force"][i]),
                     "min_distance_A": d, "thresholds": thresholds}
            break
    result = {
        "model_raw_id": run_dir.name,
        "model": display_name(run_dir.name),
        "first_instability": first,
        "status": status,
        "thresholds": thresholds,
    }
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "failure_scan.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output_dir / "failure_scan.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["audit", "stable", "failure"], required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--no-trajectory-hash", action="store_true")
    args = ap.parse_args()
    root = Path(args.root); run = root / args.model_dir
    out = Path(args.output) / args.model_dir
    if args.mode == "audit":
        out.mkdir(parents=True, exist_ok=True)
        (out / "audit.json").write_text(json.dumps(audit_model(run, not args.no_trajectory_hash), indent=2) + "\n")
    elif args.mode == "stable":
        analyze_stable(run, out)
    else:
        scan_failure(run, out)


if __name__ == "__main__":
    main()
