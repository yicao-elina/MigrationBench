#!/usr/bin/env python3
"""Run four-model single-point evaluations on identical snapshot inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from ase.io import read


STRESS_TO_GPA = 160.21766208


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite(x) -> bool:
    return bool(np.isfinite(np.asarray(x, dtype=float)).all())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", required=True, type=Path)
    ap.add_argument("--models", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float64")
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "npz").mkdir(exist_ok=True)
    records = []
    errors = []
    models = json.loads(args.models.read_text())
    # Import only after the manifest is parsed so a missing MACE runtime is a
    # recorded blocker rather than an unstructured process failure.
    try:
        from mace.calculators import MACECalculator
    except Exception as exc:
        msg = {"stage": "import_mace", "error": repr(exc), "traceback": traceback.format_exc()}
        (out / "runtime_import_error.json").write_text(json.dumps(msg, indent=2) + "\n")
        raise

    snapshot_paths = sorted(args.snapshots.glob("snapshot_step_*.extxyz"))
    if len(snapshot_paths) != 4:
        raise RuntimeError(f"expected four snapshot inputs, found {len(snapshot_paths)}")
    for model in models:
        model_path = Path(model["model_path"])
        model_hash = sha256(model_path) if model_path.exists() else None
        model["model_size_bytes"] = model_path.stat().st_size if model_path.exists() else None
        model["model_sha256_observed"] = model_hash
        model["model_exists"] = model_path.exists()
    (out / "models_observed.json").write_text(json.dumps(models, indent=2) + "\n")

    for model in models:
        model_id = model["model_id"]
        model_path = Path(model["model_path"])
        for snapshot_path in snapshot_paths:
            step = int(snapshot_path.stem.split("_")[-1])
            base = {"model_id": model_id, "model_label": model.get("model_label", model_id), "model_path": str(model_path), "model_sha256": model.get("model_sha256_observed"), "step": step, "snapshot_path": str(snapshot_path), "snapshot_sha256": sha256(snapshot_path), "device": args.device, "dtype": args.dtype}
            started = time.time()
            try:
                atoms = read(snapshot_path, index=-1, format="extxyz")
                symbols = tuple(atoms.get_chemical_symbols())
                if len(atoms) != 2050 or not all(atoms.pbc):
                    raise ValueError(f"input identity gate failed: natoms={len(atoms)}, pbc={tuple(atoms.pbc)}")
                if not model_path.exists():
                    raise FileNotFoundError(model_path)
                calc = MACECalculator(model_paths=str(model_path), device=args.device, default_dtype=args.dtype)
                atoms.calc = calc
                energy = float(atoms.get_potential_energy())
                forces = np.asarray(atoms.get_forces(), dtype=float)
                stress = None
                stress_error = None
                try:
                    stress = np.asarray(atoms.get_stress(voigt=False), dtype=float)
                except Exception as exc:
                    stress_error = repr(exc)
                if forces.shape != (len(atoms), 3):
                    raise ValueError(f"force shape {forces.shape}")
                npz_path = out / "npz" / f"{model_id}_step_{step:06d}.npz"
                np.savez_compressed(npz_path, positions=np.asarray(atoms.positions), forces=forces, stress=np.asarray(stress) if stress is not None else np.full((3, 3), np.nan), species=np.asarray(symbols), cell=np.asarray(atoms.cell.array), pbc=np.asarray(atoms.pbc), energy_eV=energy)
                rec = {**base, "status": "complete", "elapsed_s": time.time() - started, "energy_eV": energy, "force_shape": list(forces.shape), "force_finite": finite(forces), "force_norm_max_eV_A": float(np.linalg.norm(forces, axis=1).max()), "force_component_max_abs_eV_A": float(np.abs(forces).max()), "stress_available": stress is not None, "stress_finite": finite(stress) if stress is not None else False, "stress_eV_A3": stress.tolist() if stress is not None else None, "stress_GPa_voigt_like": (np.asarray(stress)[np.triu_indices(3)] * STRESS_TO_GPA).tolist() if stress is not None and np.asarray(stress).shape == (3, 3) else None, "stress_error": stress_error, "raw_output_npz": str(npz_path)}
                records.append(rec)
            except Exception as exc:
                rec = {**base, "status": "failed", "elapsed_s": time.time() - started, "error": repr(exc), "traceback": traceback.format_exc()}
                records.append(rec)
                errors.append(rec)
            with (out / "evaluation_records.jsonl").open("a") as log:
                log.write(json.dumps(records[-1], sort_keys=True) + "\n")

    (out / "evaluation_summary.json").write_text(json.dumps({"schema_version": "phase1-static-force-eval-1.0", "models_requested": len(models), "snapshots_requested": len(snapshot_paths), "evaluations_requested": len(models) * len(snapshot_paths), "records": len(records), "complete": sum(r["status"] == "complete" for r in records), "failed": len(errors), "errors": errors, "runtime": {"python": sys.version, "platform": platform.platform(), "hostname": platform.node(), "numpy": np.__version__}}, indent=2) + "\n")
    print(json.dumps({"requested": len(models) * len(snapshot_paths), "complete": sum(r["status"] == "complete" for r in records), "failed": len(errors), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
