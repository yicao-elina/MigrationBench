#!/usr/bin/env python
"""Run one provenance-locked, non-overwriting MACE MD segment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import socket
import sys
import tempfile
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.neighborlist import neighbor_list
from mace.calculators import MACECalculator

STOP_REQUESTED = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def temperature(atoms) -> float:
    momenta = atoms.get_momenta()
    masses = atoms.get_masses()[:, None]
    kinetic = 0.5 * np.sum(momenta * momenta / masses)
    return float(2.0 * kinetic / (3.0 * len(atoms) * units.kB))


def minimum_distance(atoms, cutoff: float) -> float:
    distances = neighbor_list("d", atoms, cutoff=cutoff)
    return float(np.min(distances)) if len(distances) else float("nan")


def request_stop(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"received signal {signum}; stopping at next checkpoint", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--family", required=True)
    parser.add_argument("--train-seed", required=True, type=int)
    parser.add_argument("--md-seed", required=True, type=int)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--write-interval", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=600.0)
    parser.add_argument("--friction", type=float, default=0.01)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)

    atoms = read(args.input, index=-1)
    symbols = tuple(atoms.get_chemical_symbols())
    if len(atoms) != 2050:
        raise RuntimeError(f"expected 2050 atoms, found {len(atoms)}")
    if tuple(bool(x) for x in atoms.pbc) != (True, True, True):
        raise RuntimeError(f"expected fully periodic input, found {atoms.pbc}")
    if not atoms.has("momenta"):
        raise RuntimeError("input has no momenta; fresh-start provenance is incomplete")
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.get_momenta()).all():
        raise RuntimeError("input positions or momenta are non-finite")

    manifest = {
        "schema_version": "from_scratch_ablation_md_20261005.v1",
        "campaign": "mace_from_scratch_ablation_MD_20261005",
        "family": args.family,
        "training_seed": args.train_seed,
        "md_seed": args.md_seed,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "natoms": len(atoms),
        "species_sha256": hashlib.sha256(" ".join(symbols).encode()).hexdigest(),
        "cell_A": np.asarray(atoms.cell.array).tolist(),
        "pbc": [bool(x) for x in atoms.pbc],
        "initial_temperature_K": temperature(atoms),
        "temperature_target_K": args.temperature,
        "friction_fs^-1": args.friction,
        "timestep_fs": args.timestep_fs,
        "steps_requested": args.steps,
        "write_interval_steps": args.write_interval,
        "device": args.device,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "python": sys.executable,
    }
    try:
        import ase
        import mace
        import torch

        manifest["software"] = {
            "ase": getattr(ase, "__version__", "unknown"),
            "mace": getattr(mace, "__version__", "unknown"),
            "torch": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except Exception as exc:  # pragma: no cover - diagnostics only
        manifest["software_error"] = repr(exc)

    atomic_json(args.output / "run_manifest.json", manifest)
    trajectory = args.output / "trajectory.extxyz"
    checkpoint = args.output / "checkpoint.extxyz"
    status_path = args.output / "status.json"
    metrics_path = args.output / "metrics.csv"
    rows = []
    stop_reason = "target_steps"
    stop_step = 0

    calculator = MACECalculator(
        model_paths=str(args.model), device=args.device, default_dtype="float64"
    )
    atoms.calc = calculator
    dynamics = Langevin(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=args.friction,
        rng=np.random.default_rng(args.md_seed),
    )

    def checkpoint_now(step: int) -> None:
        write(checkpoint.with_suffix(".tmp.extxyz"), atoms, format="extxyz")
        os.replace(checkpoint.with_suffix(".tmp.extxyz"), checkpoint)
        atomic_json(
            status_path,
            {
                "status": "running",
                "last_step": int(step),
                "target_step": args.steps,
                "stop_reason": stop_reason,
                "trajectory": str(trajectory),
                "checkpoint": str(checkpoint),
            },
        )

    def record() -> None:
        nonlocal stop_reason, stop_step
        step = int(dynamics.nsteps)
        forces = atoms.get_forces()
        energy = float(atoms.get_potential_energy())
        temp = temperature(atoms)
        min_dist = minimum_distance(atoms, cutoff=6.0)
        row = {
            "step": step,
            "time_ps": step * args.timestep_fs * 1.0e-3,
            "potential_energy_eV": energy,
            "temperature_K": temp,
            "max_force_eV_A": float(np.max(np.abs(forces))),
            "rms_force_eV_A": float(np.sqrt(np.mean(forces * forces))),
            "min_distance_A": min_dist,
        }
        rows.append(row)
        atoms.info.update(row)
        write(trajectory, atoms, format="extxyz", append=trajectory.exists())
        checkpoint_now(step)
        bad = (
            not np.isfinite(
                [energy, temp, row["max_force_eV_A"], row["rms_force_eV_A"]]
            ).all()
            or temp > 5000.0
            or row["max_force_eV_A"] > 100.0
            or (np.isfinite(min_dist) and min_dist < 0.5)
        )
        if bad:
            stop_reason = "instability_threshold"
            stop_step = step
            raise RuntimeError("scientific instability threshold reached")
        if STOP_REQUESTED:
            stop_reason = "signal"
            stop_step = step
            raise InterruptedError("received scheduler stop signal")

    try:
        dynamics.attach(record, interval=args.write_interval)
        dynamics.run(args.steps)
    except (RuntimeError, InterruptedError) as exc:
        print(str(exc), file=sys.stderr, flush=True)
        stop_step = stop_step or int(dynamics.nsteps)
    finally:
        with metrics_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["step"])
            writer.writeheader()
            writer.writerows(rows)
        final_step = stop_step or int(dynamics.nsteps)
        complete = final_step >= args.steps and stop_reason == "target_steps"
        status = {
            "status": "complete" if complete else "failed" if stop_reason == "instability_threshold" else "interrupted",
            "last_step": final_step,
            "target_step": args.steps,
            "stop_reason": stop_reason,
            "n_metric_frames": len(rows),
        }
        atomic_json(status_path, status)
        manifest["final_status"] = status
        atomic_json(args.output / "run_manifest.json", manifest)

    print(json.dumps(status, sort_keys=True))
    return 0 if status["status"] == "complete" else 99


if __name__ == "__main__":
    raise SystemExit(main())
