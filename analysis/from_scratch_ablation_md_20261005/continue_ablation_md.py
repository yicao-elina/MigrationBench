#!/usr/bin/env python
"""Continue one validated from-scratch MD run from a checkpoint.

The initial production invocation predates this continuation helper and does
not serialize NumPy's Langevin RNG state.  For its first continuation segment
we reconstruct that state by advancing the same Generator through the exact
two normal draws used by ASE's Langevin integrator at every preceding step.
Subsequent segments persist and restore the RNG state directly.  Each segment
has its own directory and never overwrites the source run.
"""

from __future__ import annotations

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
from ase.neighborlist import neighbor_list
from mace.calculators import MACECalculator


STOP_REQUESTED = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def atomic_json(path: Path, payload: dict) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def request_stop(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"received signal {signum}; stopping at next saved frame", flush=True)


def temperature(atoms) -> float:
    momenta = atoms.get_momenta()
    masses = atoms.get_masses()[:, None]
    kinetic = 0.5 * np.sum(momenta * momenta / masses)
    return float(2.0 * kinetic / (3.0 * len(atoms) * units.kB))


def minimum_distance(atoms, cutoff: float = 6.0) -> float:
    distances = neighbor_list("d", atoms, cutoff=cutoff)
    return float(np.min(distances)) if len(distances) else float("nan")


def validate_atoms(atoms) -> None:
    if len(atoms) != 2050:
        raise RuntimeError(f"expected 2050 atoms, found {len(atoms)}")
    if tuple(bool(x) for x in atoms.pbc) != (True, True, True):
        raise RuntimeError(f"expected fully periodic checkpoint, found {atoms.pbc}")
    if not atoms.has("momenta"):
        raise RuntimeError("checkpoint has no momenta")
    if not np.isfinite(atoms.positions).all():
        raise RuntimeError("checkpoint positions are non-finite")
    if not np.isfinite(atoms.get_momenta()).all():
        raise RuntimeError("checkpoint momenta are non-finite")
    if not np.isfinite(atoms.cell.array).all():
        raise RuntimeError("checkpoint cell is non-finite")


def reconstruct_rng(md_seed: int, preceding_steps: int, natoms: int) -> np.random.Generator:
    """Recreate the RNG state used by the original runner through step N.

    ASE Langevin draws one ``(natoms, 3)`` standard-normal array for ``xi``
    and one for ``eta`` per step.  The loop intentionally preserves that
    per-step draw order rather than batching xi and eta separately.
    """
    rng = np.random.default_rng(md_seed)
    for _ in range(preceding_steps):
        rng.standard_normal(size=(natoms, 3))
        rng.standard_normal(size=(natoms, 3))
    return rng


class WalltimeStop(Exception):
    pass


class ScientificFailure(Exception):
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--family", required=True)
    parser.add_argument("--train-seed", required=True, type=int)
    parser.add_argument("--md-seed", required=True, type=int)
    parser.add_argument("--start-step", required=True, type=int)
    parser.add_argument("--target-step", default=1_000_000, type=int)
    parser.add_argument("--write-interval", default=200, type=int)
    parser.add_argument("--temperature", default=600.0, type=float)
    parser.add_argument("--friction", default=0.01, type=float)
    parser.add_argument("--timestep-fs", default=1.0, type=float)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)

    source_checkpoint = args.checkpoint.resolve()
    model_path = args.model.resolve()
    output = args.output.resolve()
    if not source_checkpoint.exists() or not model_path.exists():
        raise FileNotFoundError(f"missing checkpoint/model: {source_checkpoint} {model_path}")
    if args.start_step < 0 or args.target_step <= args.start_step:
        raise ValueError("target step must be greater than non-negative start step")
    output.mkdir(parents=True, exist_ok=True)

    status_path = output / "status.json"
    restart_path = output / "latest_restart.extxyz"
    ledger_path = output / "continuation_ledger.jsonl"
    segment_path = output / f"segment_from_{args.start_step:09d}.extxyz"
    metrics_path = output / f"metrics_from_{args.start_step:09d}.csv"
    manifest_path = output / "run_manifest.json"

    checkpoint_sha = sha256_file(source_checkpoint)
    model_sha = sha256_file(model_path)
    rng_reconstruction = False

    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get("model_sha256") != model_sha:
            raise RuntimeError("model hash changed; refusing to resume segment")
        if int(status.get("target_step", -1)) != args.target_step:
            raise RuntimeError("target step changed; refusing to resume segment")
        if not restart_path.exists():
            raise RuntimeError("segment status exists but latest_restart.extxyz is missing")
        current_step = int(status["last_step"])
        atoms = read(restart_path, index=-1, format="extxyz")
        validate_atoms(atoms)
        rng = np.random.default_rng(args.md_seed)
        if status.get("rng_state") is None:
            raise RuntimeError("resumable segment lacks serialized rng_state")
        rng.bit_generator.state = status["rng_state"]
        if current_step >= args.target_step:
            return 0
        if segment_path.exists() and not metrics_path.exists():
            raise RuntimeError("segment exists without its metrics ledger")
    else:
        atoms = read(source_checkpoint, index=-1, format="extxyz")
        validate_atoms(atoms)
        checkpoint_step = atoms.info.get("step")
        if checkpoint_step is not None and int(checkpoint_step) != args.start_step:
            raise RuntimeError(
                f"checkpoint step {checkpoint_step} disagrees with --start-step {args.start_step}"
            )
        current_step = args.start_step
        if segment_path.exists() or metrics_path.exists() or restart_path.exists():
            raise RuntimeError("refusing to overwrite pre-existing continuation artifacts")
        rng = reconstruct_rng(args.md_seed, current_step, len(atoms))
        rng_reconstruction = True
        write(restart_path, atoms, format="extxyz")

    # ``dynamics.nsteps`` starts at zero for every scheduler invocation.  The
    # absolute offset therefore comes from the latest saved checkpoint, not
    # necessarily from the segment's original --start-step.
    run_start_step = current_step

    if not segment_path.exists():
        write(segment_path, atoms, format="extxyz")
    if not metrics_path.exists():
        with metrics_path.open("w", newline="") as handle:
            csv.DictWriter(
                handle,
                fieldnames=[
                    "step",
                    "time_ps",
                    "potential_energy_eV",
                    "temperature_K",
                    "max_force_eV_A",
                    "rms_force_eV_A",
                    "min_distance_A",
                ],
            ).writeheader()

    manifest = {
        "schema_version": "from_scratch_ablation_md_continuation_20261005.v1",
        "campaign": "mace_from_scratch_ablation_MD_20261005",
        "family": args.family,
        "training_seed": args.train_seed,
        "md_seed": args.md_seed,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": checkpoint_sha,
        "model": str(model_path),
        "model_sha256": model_sha,
        "start_step": args.start_step,
        "target_step": args.target_step,
        "timestep_fs": args.timestep_fs,
        "temperature_target_K": args.temperature,
        "friction_fs^-1": args.friction,
        "write_interval_steps": args.write_interval,
        "natoms": len(atoms),
        "pbc": [bool(x) for x in atoms.pbc],
        "rng_mode": "reconstructed_from_initial_md_seed" if rng_reconstruction else "restored_from_status",
        "segment": str(segment_path),
        "metrics": str(metrics_path),
    }
    if not manifest_path.exists():
        atomic_json(manifest_path, manifest)

    calc = MACECalculator(model_paths=str(model_path), device=args.device, default_dtype="float64")
    atoms.calc = calc
    dynamics = Langevin(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=args.friction,
        rng=rng,
    )

    def checkpoint(reason: str) -> None:
        status_value = {
            "periodic": "running",
            "target_steps": "complete",
            "scientific_failure": "failed",
            "runtime_failure": "interrupted",
            "walltime_or_signal": "interrupted",
        }.get(reason, reason)
        write(restart_path, atoms, format="extxyz")
        atomic_json(
            status_path,
            {
                "status": status_value,
                "last_step": int(current_step),
                "target_step": args.target_step,
                "reason": reason,
                "source_checkpoint_sha256": checkpoint_sha,
                "model_sha256": model_sha,
                "rng_state": rng.bit_generator.state,
                "segment": str(segment_path),
                "metrics": str(metrics_path),
            },
        )

    def record() -> None:
        nonlocal current_step
        current_step = run_start_step + int(dynamics.nsteps)
        forces = atoms.get_forces()
        energy = float(atoms.get_potential_energy())
        temp = temperature(atoms)
        dmin = minimum_distance(atoms)
        row = {
            "step": current_step,
            "time_ps": current_step * args.timestep_fs * 1.0e-3,
            "potential_energy_eV": energy,
            "temperature_K": temp,
            "max_force_eV_A": float(np.max(np.abs(forces))),
            "rms_force_eV_A": float(np.sqrt(np.mean(forces * forces))),
            "min_distance_A": dmin,
        }
        atoms.info.update(row)
        write(segment_path, atoms, format="extxyz", append=True)
        with metrics_path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=list(row)).writerow(row)
        finite = np.isfinite(
            [energy, temp, row["max_force_eV_A"], row["rms_force_eV_A"], dmin]
        ).all()
        bad = (
            not finite
            or temp > 5000.0
            or row["max_force_eV_A"] > 100.0
            or (np.isfinite(dmin) and dmin < 0.5)
        )
        if bad:
            checkpoint("scientific_failure")
            raise ScientificFailure(f"scientific threshold reached at step {current_step}")
        checkpoint("periodic")
        if STOP_REQUESTED:
            checkpoint("walltime_or_signal")
            raise WalltimeStop(f"checkpointed at step {current_step}")

    # The checkpoint itself is already a valid source frame.  The segment file
    # starts with that frame and then appends absolute-step frames every 200 fs.
    try:
        dynamics.attach(record, interval=args.write_interval)
        dynamics.run(args.target_step - run_start_step)
    except WalltimeStop as exc:
        print(str(exc), flush=True)
        return 99
    except ScientificFailure as exc:
        print(str(exc), flush=True)
        return 42
    except (RuntimeError, ValueError) as exc:
        checkpoint("runtime_failure")
        print(str(exc), flush=True)
        return 42

    if current_step < args.target_step:
        checkpoint("runtime_failure")
        raise RuntimeError("dynamics returned before target step")
    checkpoint("target_steps")
    atomic_json(manifest_path, {**manifest, "final_status": "complete", "last_step": current_step})
    print(json.dumps({"status": "complete", "last_step": current_step}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
