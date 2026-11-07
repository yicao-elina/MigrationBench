#!/usr/bin/env python3
"""Run the foundation-model MD smoke campaign and summarize pressure stability.

The smoke mode intentionally runs only two models and a short trajectory.  It
checks that the same input, calculator interface, trajectory writer, plotting,
and uncertainty code work together; it is not a scientific production run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ase import units
from ase.io import Trajectory, read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary


ROOT = Path(__file__).parents[1]
RAW = ROOT / "results" / "raw"
PROCESSED = ROOT / "results" / "processed"
FIGURES = ROOT / "results" / "figures"


def attach_calculator(atoms, model: str, device: str, dtype: str):
    from mace.calculators import mace_mp

    atoms.calc = mace_mp(model=model, device=device, default_dtype=dtype)
    return atoms


def run_model(xyz: Path, model: str, steps: int, seed: int, log_every: int, device: str, dtype: str):
    atoms = read(xyz)
    MaxwellBoltzmannDistribution(atoms, temperature_K=600.0, rng=np.random.default_rng(seed))
    Stationary(atoms)
    attach_calculator(atoms, model, device, dtype)

    traj_path = RAW / f"md_{model}_600K_{steps}steps.traj"
    log_path = RAW / f"md_{model}_600K_{steps}steps_log.csv"
    dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=600.0, friction=0.02)
    trajectory = Trajectory(traj_path, "w", atoms)
    dyn.attach(trajectory.write, interval=log_every)
    rows: list[tuple[float, float, float, float]] = []

    def log_step():
        kinetic = atoms.get_kinetic_energy()
        temperature = kinetic / (1.5 * units.kB * len(atoms))
        try:
            pressure = -float(np.mean(atoms.get_stress(voigt=True)[:3])) / units.GPa
        except Exception:
            pressure = math.nan
        rows.append((dyn.get_time() / units.fs, atoms.get_potential_energy(), temperature, pressure))

    dyn.attach(log_step, interval=log_every)
    dyn.run(steps)
    trajectory.close()
    with log_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_fs", "potential_energy_eV", "temperature_K", "pressure_GPa"))
        writer.writerows(rows)
    return log_path


def integrated_autocorrelation_time(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 4 or np.std(values) == 0:
        return 0.5
    centered = values - np.mean(values)
    variance = np.dot(centered, centered) / len(centered)
    acf = np.correlate(centered, centered, mode="full")[len(centered) - 1 :] / len(centered)
    acf /= variance
    positive = acf[1:]
    cutoff = np.flatnonzero(positive <= 0)
    end = int(cutoff[0]) if len(cutoff) else len(positive)
    return max(0.5, 0.5 + float(np.sum(positive[:end])))


def summarize(log_paths: list[Path]) -> Path:
    import pandas as pd

    records = []
    for path in log_paths:
        frame = pd.read_csv(path)
        pressure = frame["pressure_GPa"].to_numpy(dtype=float)
        finite = np.isfinite(pressure)
        pressure = pressure[finite]
        if len(pressure) < 4:
            raise RuntimeError(f"not enough finite pressure samples in {path}")
        tau = integrated_autocorrelation_time(pressure)
        burn_in = min(len(pressure) - 2, max(1, int(math.ceil(5 * tau))))
        steady = pressure[burn_in:]
        block = max(1, int(math.ceil(2 * tau)))
        blocks = np.array([np.mean(steady[i : i + block]) for i in range(0, len(steady), block)])
        se = float(np.std(blocks, ddof=1) / math.sqrt(len(blocks))) if len(blocks) > 1 else 0.0
        model = path.name.removeprefix("md_").split("_600K_", 1)[0]
        records.append(
            {
                "model": model,
                "n_samples": len(pressure),
                "burn_in_samples": burn_in,
                "integrated_autocorrelation_time_samples": tau,
                "block_size_samples": block,
                "pressure_mean_GPa": float(np.mean(steady)),
                "pressure_se_GPa": se,
            }
        )
    output = PROCESSED / "steady_state_stats.csv"
    pd.DataFrame.from_records(records).to_csv(output, index=False)
    return output


def plot(log_paths: list[Path]) -> Path:
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    for path in log_paths:
        frame = pd.read_csv(path)
        label = path.name.removeprefix("md_").split("_600K_", 1)[0]
        axes[0].plot(frame["time_fs"], frame["temperature_K"], label=label)
        axes[1].plot(frame["time_fs"], frame["pressure_GPa"], label=label)
    axes[0].set_ylabel("Temperature (K)")
    axes[1].set_ylabel("Pressure (GPa)")
    axes[1].set_xlabel("Time (fs)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    output = FIGURES / "multi_file_comparison.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def write_manifest(model: str, paths: list[Path]) -> Path:
    manifest = {
        "run_id": f"foundation_model_md_smoke_{model}",
        "model": model,
        "temperature_K": 600,
        "files": [],
    }
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["files"].append(
            {"path": str(path.relative_to(ROOT)), "sha256": digest}
        )
    output = ROOT / "manifests" / f"smoke_{model}.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run two models for CI validation")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--models", nargs="+", default=["medium-mpa-0", "medium-0b3"])
    parser.add_argument("--device", default="")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    if args.steps < 1 or args.log_every < 1:
        raise SystemExit("--steps and --log-every must be positive")
    models = ["medium-mpa-0", "medium-0b3"] if args.smoke else args.models
    for directory in (RAW, PROCESSED, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    xyz = ROOT / "data" / "raw" / "sb2te3_cr_81atom.xyz"
    logs = [run_model(xyz, model, args.steps, 42 + i, args.log_every, args.device, args.dtype)
            for i, model in enumerate(models)]
    stats = summarize(logs)
    figure = plot(logs)
    for model, log in zip(models, logs):
        trajectory = RAW / f"md_{model}_600K_{args.steps}steps.traj"
        write_manifest(model, [trajectory, log, stats, figure])
    print(f"completed {len(logs)} model(s) for {args.steps} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
