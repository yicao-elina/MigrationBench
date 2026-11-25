#!/usr/bin/env python
import argparse, json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ase.io import iread
from ase import units

def frames(path):
    return iread(str(path), format="extxyz")

def load_cr(path, start_step=None, interval=200):
    xyz, vel, energy, temperature, force_max, min_dist, steps = [], [], [], [], [], [], []
    species = None; cell = None; pbc = None
    for i, a in enumerate(frames(path)):
        if species is None:
            species = tuple(a.get_chemical_symbols()); cell = np.asarray(a.cell.array); pbc = tuple(a.pbc)
            if len(a) != 2050 or pbc != (True, True, True):
                raise ValueError(f"{path}: unexpected natoms/pbc {len(a)}/{pbc}")
        if tuple(a.get_chemical_symbols()) != species:
            raise ValueError(f"{path}: species identity changed at frame {i}")
        if not np.isfinite(a.positions).all() or not np.isfinite(a.get_momenta()).all():
            raise ValueError(f"{path}: non-finite geometry/momenta at frame {i}")
        xyz.append(a.positions[np.array(species) == "Cr"])
        masses = a.get_masses()[np.array(species) == "Cr"][:, None]
        vel.append(a.get_momenta()[np.array(species) == "Cr"] / masses)
        e = a.info.get("energy")
        if e is None and a.calc is not None:
            e = a.calc.results.get("energy", np.nan)
        energy.append(float(e) if e is not None else np.nan)
        p = a.get_momenta()
        m = a.get_masses()[:, None]
        kinetic = float(0.5 * np.sum(p * p / m))
        temperature.append(2.0 * kinetic / (3.0 * len(a) * units.kB))
        f = a.get_forces()
        force_max.append(float(np.abs(f).max()) if np.isfinite(f).all() else np.inf)
        # exact all-pairs minimum is cheap enough for 2050 atoms only on sampled frames
        if i == 0 or i == (0 if start_step is None else 0) or i % 100 == 0:
            d = a.get_all_distances(mic=True)
            np.fill_diagonal(d, np.inf); min_dist.append(float(d.min()))
        else:
            min_dist.append(np.nan)
        if "step" in a.info: steps.append(int(a.info["step"]))
        else: steps.append((start_step or 0) + i * interval)
    if not xyz: raise ValueError(f"no frames in {path}")
    return dict(xyz=np.asarray(xyz), vel=np.asarray(vel), energy=np.asarray(energy),
                temperature=np.asarray(temperature),
                force_max=np.asarray(force_max), min_dist=np.asarray(min_dist),
                steps=np.asarray(steps), cell=cell, species=species)

def unwrap(frac, cell):
    out = np.empty_like(frac); out[0] = frac[0]
    for i in range(1, len(frac)):
        df = frac[i] - frac[i-1]
        df -= np.rint(df)
        out[i] = out[i-1] + df
    return out @ cell

def multi_origin_msd(pos, max_lag=None, origin_stride=10):
    n = len(pos); max_lag = max_lag or (n - 1)
    lags = np.arange(1, max_lag + 1); msd = np.full(len(lags), np.nan)
    for j, lag in enumerate(lags):
        origins = np.arange(0, n - lag, origin_stride)
        if len(origins):
            dr = pos[origins + lag] - pos[origins]
            msd[j] = np.mean(np.sum(dr * dr, axis=2), axis=0).mean()
    return lags, msd

def fit_window(t, y, lo, hi, dim=4.0):
    mask = np.isfinite(y) & (t >= lo) & (t <= hi)
    if mask.sum() < 5: return {"n": int(mask.sum()), "slope_A2_per_ps": None, "D_cm2_s": None}
    slope, intercept = np.polyfit(t[mask], y[mask], 1)
    # A^2/ps -> cm^2/s: 1 A^2/ps = 1e-4 cm^2/s
    return {"n": int(mask.sum()), "slope_A2_per_ps": float(slope),
            "intercept_A2": float(intercept), "D_cm2_s": float(slope / dim * 1e-4)}

def analyze(name, original, segment, out):
    base = load_cr(original, start_step=0)
    seg = load_cr(segment, start_step=int(base["steps"][-1])) if segment and Path(segment).exists() else None
    chunks = [base] + ([seg] if seg else [])
    xyz = np.concatenate([x["xyz"] for x in chunks], axis=0)
    vel = np.concatenate([x["vel"] for x in chunks], axis=0)
    energy = np.concatenate([x["energy"] for x in chunks])
    temperature = np.concatenate([x["temperature"] for x in chunks])
    force_max = np.concatenate([x["force_max"] for x in chunks])
    min_dist = np.concatenate([x["min_dist"] for x in chunks])
    steps = np.concatenate([x["steps"] for x in chunks])
    # Drop a duplicate boundary frame only if exact step repeats.
    keep = np.r_[True, np.diff(steps) > 0]
    xyz, vel, energy, temperature, force_max, min_dist, steps = xyz[keep], vel[keep], energy[keep], temperature[keep], force_max[keep], min_dist[keep], steps[keep]
    cell = base["cell"]
    frac = np.einsum("nij,jk->nik", xyz, np.linalg.inv(cell))
    unwrapped = unwrap(frac, cell)
    t = (steps - steps[0]) * 0.001
    lag, msd = multi_origin_msd(unwrapped, max_lag=min(len(unwrapped)-1, 4000))
    lag_t = lag * 0.2
    v0 = vel - vel.mean(axis=1, keepdims=True)
    vacf = np.array([np.mean(v0[:-k] * v0[k:]) if k else np.mean(v0 * v0) for k in range(min(len(v0), 2000))])
    vacf /= vacf[0] if vacf[0] else 1.0
    result = {"model": name, "frames": int(len(steps)), "start_step": int(steps[0]), "last_step": int(steps[-1]),
              "time_ps": float(t[-1]), "natoms": int(len(base["species"])),
              "finite": bool(np.isfinite(xyz).all() and np.isfinite(vel).all() and np.isfinite(energy).all() and np.isfinite(temperature).all() and np.isfinite(force_max).all()),
              "energy_finite": bool(np.isfinite(energy).all()),
              "temperature_K": {"min": float(np.nanmin(temperature)), "mean": float(np.nanmean(temperature)), "max": float(np.nanmax(temperature))},
              "max_force_eV_A": float(np.nanmax(force_max)), "min_distance_A_sampled": float(np.nanmin(min_dist)),
              "msd_Dxy": {"200_500_ps": fit_window(lag_t, msd, 200, 500, 4), "500_plus_ps": fit_window(lag_t, msd, 500, min(900, lag_t[-1]), 4)},
              "vacf_frames": int(len(vacf)), "vacf_initial": float(vacf[0])}
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.linewidth": 1.4})
    fig, ax = plt.subplots(figsize=(5.4, 3.8)); ax.plot(lag_t, msd, lw=1.8, color="#245a9a"); ax.set(xlabel="Lag time (ps)", ylabel="MSD (Å$^2$)"); ax.tick_params(width=1.2, length=4); fig.tight_layout(); fig.savefig(out / "msd.pdf"); fig.savefig(out / "msd.svg"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.4, 3.8)); ax.plot(np.arange(len(vacf))*0.2, vacf, lw=1.8, color="#b24d35"); ax.set(xlabel="Lag time (ps)", ylabel="Normalized VACF"); ax.tick_params(width=1.2, length=4); fig.tight_layout(); fig.savefig(out / "vacf.pdf"); fig.savefig(out / "vacf.svg"); plt.close(fig)
    return result

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--original", required=True); ap.add_argument("--segment", required=False); ap.add_argument("--out", required=True); a = ap.parse_args()
    print(json.dumps(analyze(a.name, a.original, a.segment, Path(a.out)), indent=2))
