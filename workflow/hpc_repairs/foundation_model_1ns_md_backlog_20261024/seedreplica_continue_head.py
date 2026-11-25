#!/usr/bin/env python
"""Continue a validated seed-replica smoke checkpoint to absolute step 1,000,000.

The source smoke run is never modified.  Every wall-time invocation writes a
new segment, while the isolated campaign directory owns its latest checkpoint
and status ledger.  Initial momenta are read from the smoke checkpoint; no
velocity initializer is called.  The NumPy Generator state is persisted after
each checkpoint so wall-time continuation does not restart the thermostat
random stream.
"""
import argparse
import csv
import hashlib
import json
import os
import signal
import tempfile
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from mace.calculators import MACECalculator

STOP = False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def request_stop(signum, frame):
    global STOP
    STOP = True
    print(f"received signal {signum}; checkpointing at the next frame", flush=True)


def temperature(atoms):
    p = atoms.get_momenta()
    m = atoms.get_masses()[:, None]
    return float(np.sum(p * p / m) / (3.0 * len(atoms) * units.kB))


def min_distance(atoms):
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    return float(np.nanmin(d))


def finite_state(atoms):
    momenta = atoms.get_momenta()
    return bool(
        np.isfinite(atoms.get_positions()).all()
        and np.isfinite(atoms.get_cell().array).all()
        and momenta is not None
        and np.isfinite(momenta).all()
        and len(atoms) == 2050
        and all(atoms.pbc)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--training-seed", type=int, required=True)
    ap.add_argument("--md-seed", type=int, default=1100)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target-step", type=int, default=1_000_000)
    ap.add_argument("--temperature-K", type=float, default=600.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--head", default=None)
    ap.add_argument("--velocity-seed", type=int, default=None)
    ap.add_argument("--write-interval", type=int, default=200)
    ap.add_argument("--checkpoint-interval", type=int, default=5000)
    args = ap.parse_args()

    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    source = Path(args.source_dir)
    source_status_path = source / "status.json"
    source_restart = source / "latest_restart.extxyz"
    status_path = run / "status.json"
    restart_path = run / "latest_restart.extxyz"
    ledger_path = run / "continuation_ledger.jsonl"
    input_path = Path(args.input)
    model_path = Path(args.model)

    if not input_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"missing input/model: {input_path} {model_path}")
    if not source_status_path.exists() or not source_restart.exists():
        raise RuntimeError("validated smoke source is incomplete")
    source_status = json.loads(source_status_path.read_text())
    if source_status.get("smoke_pass") is not True or source_status.get("reason") != "target_steps":
        raise RuntimeError(f"refusing non-passing smoke source: {source_status}")

    source_step = int(source_status["last_step"])
    source_checkpoint_sha = sha256(source_restart)
    model_sha = sha256(model_path)
    input_sha = sha256(input_path)

    if status_path.exists():
        status = json.loads(status_path.read_text())
        step = int(status["last_step"])
        if status.get("model_sha256") != model_sha or status.get("input_sha256") != input_sha:
            raise RuntimeError("model/input hash changed; refusing to continue")
        if status.get("model_label") != args.model_label:
            raise RuntimeError("model label changed; refusing to continue")
        if not restart_path.exists():
            raise RuntimeError("campaign status exists but latest_restart.extxyz is missing")
        atoms = read(restart_path, index=-1, format="extxyz")
        rng = np.random.default_rng(args.md_seed)
        if status.get("rng_state") is not None:
            rng.bit_generator.state = status["rng_state"]
        source_for_ledger = status.get("source_checkpoint_sha256", source_checkpoint_sha)
        print(f"resuming {args.model_label} from campaign step {step}", flush=True)
    else:
        atoms = read(source_restart, index=-1, format="extxyz")
        step = source_step
        if not atoms.has("momenta") or atoms.get_momenta() is None:
            raise RuntimeError("source checkpoint has no momenta")
        write(restart_path, atoms, format="extxyz", append=False)
        rng = np.random.default_rng(args.md_seed)
        source_for_ledger = source_checkpoint_sha
        status = {}

    if not finite_state(atoms):
        raise RuntimeError("checkpoint failed 2050/PBC/finite-state validation")
    if step >= args.target_step:
        print(f"already complete at {step}", flush=True)
        return

    segment_path = run / f"segment_from_{step:09d}.extxyz"
    if segment_path.exists():
        raise RuntimeError(f"refusing to overwrite existing segment: {segment_path}")
    metrics_path = run / f"metrics_from_{step:09d}.csv"
    if metrics_path.exists():
        raise RuntimeError(f"refusing to overwrite existing metrics: {metrics_path}")

    manifest = {
        "model_label": args.model_label,
        "model": str(model_path),
        "model_sha256": model_sha,
        "input": str(input_path),
        "input_sha256": input_sha,
        "source_smoke_dir": str(source),
        "source_checkpoint_sha256": source_for_ledger,
        "source_smoke_last_step": source_step,
        "training_seed": args.training_seed,
        "md_velocity_seed": args.md_seed,
        "velocity_seed": args.velocity_seed,
        "velocity_initialization": "fresh Maxwell-Boltzmann + stationary COM" if args.velocity_seed is not None else "preserved smoke checkpoint momenta",
        "start_step": step,
        "target_step": args.target_step,
        "timestep_fs": 1.0,
        "temperature_target_K": args.temperature_K,
        "friction": 0.01,
        "head": args.head,
        "write_interval_steps": args.write_interval,
        "checkpoint_interval_steps": args.checkpoint_interval,
        "natoms": len(atoms),
        "pbc": [bool(x) for x in atoms.pbc],
        "segment": str(segment_path),
        "metrics": str(metrics_path),
    }
    atomic_json(run / "run_manifest.json", manifest)
    with ledger_path.open("a") as f:
        f.write(json.dumps({"event": "segment_start", "step": step,
                            "segment": str(segment_path), "source_checkpoint_sha256": source_for_ledger}) + "\n")

    calc = MACECalculator(model_paths=str(model_path), device=args.device,
                          default_dtype="float64", head=args.head)
    atoms.calc = calc
    dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=args.temperature_K,
                   friction=0.01, rng=rng)

    current_step = step
    rows = []
    bad_reason = None
    fields = ["step", "time_ps", "energy_eV", "temperature_K", "max_force_eV_A", "min_distance_A"]

    def checkpoint(reason="periodic"):
        payload = {
            "last_step": int(current_step),
            "target_step": args.target_step,
            "model_label": args.model_label,
            "training_seed": args.training_seed,
            "md_velocity_seed": args.md_seed,
            "velocity_seed": args.velocity_seed,
            "model_sha256": model_sha,
            "input_sha256": input_sha,
            "source_checkpoint_sha256": source_for_ledger,
            "segment": str(segment_path),
            "metrics": str(metrics_path),
            "reason": reason,
            "natoms": len(atoms),
            "pbc": [bool(x) for x in atoms.pbc],
            "rng_state": rng.bit_generator.state,
        }
        write(restart_path, atoms, format="extxyz", append=False)
        atomic_json(status_path, payload)

    def record():
        nonlocal current_step, bad_reason
        current_step = step + int(dyn.nsteps)
        forces = atoms.get_forces()
        energy = float(atoms.get_potential_energy())
        temp = temperature(atoms)
        dmin = min_distance(atoms) if current_step % 1000 == 0 else np.nan
        atoms.info.update({"step": current_step, "time_fs": float(current_step),
                           "absolute_step": current_step})
        write(segment_path, atoms, format="extxyz", append=segment_path.exists())
        row = {"step": current_step, "time_ps": current_step * 0.001,
               "energy_eV": energy, "temperature_K": temp,
               "max_force_eV_A": float(np.abs(forces).max()),
               "min_distance_A": float(dmin)}
        rows.append(row)
        bad = (not finite_state(atoms) or not np.isfinite(forces).all()
               or not np.isfinite(energy) or not np.isfinite(temp)
               or temp > 5000.0 or row["max_force_eV_A"] > 100.0
               or (np.isfinite(dmin) and dmin < 0.5))
        if bad:
            bad_reason = "scientific_threshold"
        if STOP or bad:
            checkpoint("walltime_signal" if STOP and not bad else bad_reason)
            raise SystemExit(99 if STOP and not bad else 42)

    record()
    dyn.attach(record, interval=args.write_interval)
    dyn.attach(lambda: checkpoint("periodic"), interval=args.checkpoint_interval)
    remaining = args.target_step - step
    exit_code = 0
    try:
        dyn.run(remaining)
    except SystemExit as exc:
        exit_code = int(exc.code)
    finally:
        if rows:
            with metrics_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        if current_step >= args.target_step and bad_reason is None:
            checkpoint("target_steps")
            with ledger_path.open("a") as f:
                f.write(json.dumps({"event": "target_reached", "step": current_step,
                                    "segment": str(segment_path)}) + "\n")
        elif exit_code == 42 or bad_reason is not None:
            checkpoint(bad_reason or "scientific_threshold")
            with ledger_path.open("a") as f:
                f.write(json.dumps({"event": "scientific_failure", "step": current_step,
                                    "reason": bad_reason or "scientific_threshold"}) + "\n")
        else:
            checkpoint("walltime_or_stop")
            with ledger_path.open("a") as f:
                f.write(json.dumps({"event": "segment_end", "step": current_step,
                                    "reason": "walltime_or_stop", "segment": str(segment_path)}) + "\n")
    if exit_code:
        raise SystemExit(exit_code)
    print(f"completed {args.model_label} at step {current_step}", flush=True)


if __name__ == "__main__":
    main()
