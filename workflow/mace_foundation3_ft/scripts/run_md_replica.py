#!/usr/bin/env python3
"""Run one auditable MACE-MD replica from a completed training artifact."""
import argparse
import csv
import hashlib
import json
import os
import signal
import tempfile
from pathlib import Path

import numpy as np
import yaml
from ase import units
from ase.io import read, write
from ase.md.langevin import Langevin
from mace.calculators import MACECalculator

STOP = False


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def stop_handler(signum, frame):
    global STOP
    STOP = True


def finite_state(atoms, required_natoms, required_pbc):
    momenta = atoms.get_momenta()
    return bool(
        len(atoms) == required_natoms
        and tuple(bool(x) for x in atoms.pbc) == tuple(required_pbc)
        and np.isfinite(atoms.get_positions()).all()
        and np.isfinite(atoms.get_cell().array).all()
        and momenta is not None
        and np.isfinite(momenta).all()
    )


def temperature(atoms):
    momenta = atoms.get_momenta()
    masses = atoms.get_masses()[:, None]
    return float(np.sum(momenta * momenta / masses) / (3.0 * len(atoms) * units.kB))


def min_distance(atoms):
    distances = atoms.get_all_distances(mic=True)
    np.fill_diagonal(distances, np.inf)
    return float(np.nanmin(distances))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop_handler)

    cfg = yaml.safe_load(args.config.read_text())
    md = cfg["md"]
    requested = os.environ.get("CAMPAIGN_FAMILIES")
    family_ids = [item.strip() for item in requested.split(",")] if requested else list(cfg["families"])
    family_ids = [item for item in family_ids if item]
    jobs = [(family, int(seed)) for family in family_ids for seed in cfg["training"]["seeds"]]
    if not 0 <= args.index < len(jobs):
        raise SystemExit("invalid MD array index %d" % args.index)
    family_id, training_seed = jobs[args.index]
    family = cfg["families"][family_id]
    campaign_root = args.output_root or Path(cfg["remote_root"])
    run_dir = campaign_root / "md" / family_id / ("seed%d" % training_seed)
    if run_dir.exists():
        if not (run_dir / "status.json").is_file():
            raise SystemExit("existing MD directory has no status.json; refusing ambiguous restart: %s" % run_dir)
    else:
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(exist_ok=False)
    training_manifest = Path(cfg["remote_root"]) / family["output_dir_template"].format(seed=training_seed) / "training_manifest.json"
    if not training_manifest.is_file():
        raise SystemExit("missing completed training manifest: %s" % training_manifest)
    training = json.loads(training_manifest.read_text())
    model_path = Path(training["artifact_path"])
    if not model_path.is_file() or sha256(model_path) != training["artifact_sha256"]:
        raise SystemExit("training artifact missing or hash changed")
    input_path = Path(md["input"])
    if not input_path.is_file() or sha256(input_path) != md["input_sha256"]:
        raise SystemExit("MD input missing or hash changed")

    status_path = run_dir / "status.json"
    restart_path = run_dir / "latest_restart.extxyz"
    ledger_path = run_dir / "continuation_ledger.jsonl"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status["model_sha256"] != training["artifact_sha256"] or status["input_sha256"] != md["input_sha256"]:
            raise SystemExit("model/input hash changed; refusing continuation")
        atoms = read(str(restart_path), index=-1, format="extxyz")
        step = int(status["last_step"])
    else:
        atoms = read(str(input_path), index=int(md.get("input_frame", -1)), format="extxyz")
        step = int(md.get("initial_absolute_step", 0))
        if atoms.get_momenta() is None:
            raise SystemExit("input frame has no momenta; refusing to invent velocities")
        write(str(restart_path), atoms, format="extxyz", append=False)
    if not finite_state(atoms, int(md["required_natoms"]), md["required_pbc"]):
        raise SystemExit("initial checkpoint failed atom/PBC/finite-state gate")
    target = int(md["target_step"])
    segment = run_dir / ("segment_from_%09d.extxyz" % step)
    metrics = run_dir / ("metrics.csv" if not (run_dir / "metrics.csv").exists() else "metrics_from_%09d.csv" % step)
    if segment.exists() or metrics.exists():
        raise SystemExit("refusing to overwrite an existing segment or metrics file")

    md_seed = int(md["md_velocity_seed_by_training_seed"][str(training_seed)])
    manifest = {
        "schema_version": "1.0", "campaign_id": cfg["campaign_id"], "family": family_id,
        "training_seed": training_seed, "md_velocity_seed": md_seed,
        "model": str(model_path), "model_sha256": training["artifact_sha256"],
        "model_type": training["model_type"],
        "input": str(input_path), "input_sha256": md["input_sha256"],
        "start_step": step, "target_step": target, "timestep_fs": md["timestep_fs"],
        "temperature_target_K": md["temperature_K"], "friction": md["friction"],
        "segment": str(segment), "metrics": str(metrics), "natoms": len(atoms),
        "pbc": [bool(x) for x in atoms.pbc],
    }
    run_manifest_path = run_dir / "run_manifest.json"
    if run_manifest_path.exists():
        run_manifest_path = run_dir / ("run_manifest_from_%09d.json" % step)
    manifest["run_manifest"] = str(run_manifest_path)
    atomic_json(run_manifest_path, manifest)
    with ledger_path.open("a") as fh:
        fh.write(json.dumps({"event": "segment_start", "step": step, "segment": str(segment)}) + "\n")

    rng = np.random.default_rng(md_seed)
    calc_kwargs = {
        "model_paths": str(model_path),
        "device": md.get("device", "cuda"),
        "default_dtype": "float64",
        "model_type": training["model_type"],
    }
    if training["model_type"] == "PolarMACE":
        calc_kwargs["pbc_handling"] = "auto"
    calc = MACECalculator(**calc_kwargs)
    atoms.calc = calc
    dyn = Langevin(atoms, timestep=float(md["timestep_fs"]) * units.fs, temperature_K=float(md["temperature_K"]), friction=float(md["friction"]), rng=rng)
    current = {"step": step}
    rows = []
    bad_reason = None
    fields = ["step", "time_ps", "energy_eV", "temperature_K", "max_force_eV_A", "min_distance_A", "stress_finite"]

    def checkpoint(reason):
        payload = dict(manifest)
        write(str(restart_path), atoms, format="extxyz", append=False)
        payload.update({"last_step": current["step"], "time_fs": float(current["step"]), "reason": reason, "rng_state": rng.bit_generator.state, "checkpoint_sha256": sha256(restart_path)})
        atomic_json(status_path, payload)

    def record():
        nonlocal bad_reason
        current["step"] = step + int(dyn.nsteps)
        forces = np.asarray(atoms.get_forces(), dtype=float)
        energy = float(atoms.get_potential_energy())
        temp = temperature(atoms)
        try:
            stress_finite = bool(np.isfinite(np.asarray(atoms.get_stress(), dtype=float)).all())
        except Exception:
            stress_finite = None
        dmin = min_distance(atoms) if current["step"] % 1000 == 0 else float("nan")
        atoms.info.update({"step": current["step"], "absolute_step": current["step"], "time_fs": float(current["step"])})
        write(str(segment), atoms, append=segment.exists(), format="extxyz")
        row = {"step": current["step"], "time_ps": current["step"] * 0.001, "energy_eV": energy, "temperature_K": temp, "max_force_eV_A": float(np.abs(forces).max()), "min_distance_A": dmin, "stress_finite": stress_finite}
        rows.append(row)
        bad = (not finite_state(atoms, int(md["required_natoms"]), md["required_pbc"]) or not np.isfinite(forces).all() or not np.isfinite(energy) or not np.isfinite(temp) or stress_finite is False or temp > float(md.get("max_temperature_K", 5000.0)) or row["max_force_eV_A"] > float(md.get("max_force_eV_A", 100.0)) or (np.isfinite(dmin) and dmin < float(md.get("min_distance_A", 0.5))))
        if bad:
            bad_reason = "scientific_threshold"
        if STOP or bad:
            checkpoint("walltime_signal" if STOP and not bad else bad_reason)
            raise SystemExit(99 if STOP and not bad else 42)

    record()
    dyn.attach(record, interval=int(md["write_interval_steps"]))
    dyn.attach(lambda: checkpoint("periodic"), interval=int(md["checkpoint_interval_steps"]))
    exit_code = 0
    try:
        dyn.run(max(target - step, 0))
    except SystemExit as exc:
        exit_code = int(exc.code)
    finally:
        with metrics.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        if current["step"] >= target and bad_reason is None:
            checkpoint("target_steps")
            event = {"event": "target_reached", "step": current["step"]}
        else:
            checkpoint(bad_reason or "walltime_or_stop")
            event = {"event": "segment_end", "step": current["step"], "reason": bad_reason or "walltime_or_stop"}
        with ledger_path.open("a") as fh:
            fh.write(json.dumps(event) + "\n")
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
