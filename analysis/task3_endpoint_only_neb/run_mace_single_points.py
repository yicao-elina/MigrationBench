#!/usr/bin/env python3
"""MACE single-point energies/forces for non-equilibrium DFT-path samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import socket
import sys
import traceback
from pathlib import Path

import numpy as np
from ase import __version__ as ase_version
from ase.io import iread


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--model-label", required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--head", default=None, help="Explicit multi-head name for multi-head MACE models")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_sha = sha256(args.model_path) if args.model_path.is_file() else "missing"
    runtime = {"python": sys.version.split()[0], "ase": ase_version, "hostname": socket.gethostname(), "platform": platform.platform(), "device": args.device, "head": args.head or "default", "model_id": args.model_id, "model_sha256": model_sha}
    (args.output_dir / "runtime_manifest.json").write_text(json.dumps(runtime, indent=2) + "\n")
    rows = []
    try:
        from mace.calculators import MACECalculator
        calc_kwargs = {"model_paths": str(args.model_path), "device": args.device, "default_dtype": "float64"}
        if args.head:
            calc_kwargs["head"] = args.head
        calc = MACECalculator(**calc_kwargs)
    except Exception:
        err = traceback.format_exc()
        (args.output_dir / "model_load_failure.txt").write_text(err)
        with (args.output_dir / "mace_single_point_results.csv").open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=["status", "model_id", "model_sha256", "failure_reason"]).writeheader()
            csv.DictWriter(f, fieldnames=["status", "model_id", "model_sha256", "failure_reason"]).writerow({"status": "model_load_failure", "model_id": args.model_id, "model_sha256": model_sha, "failure_reason": err[-2000:]})
        raise
    for input_path in sorted(args.input_dir.glob("*.extxyz")):
        input_sha = sha256(input_path)
        for frame_index, atoms in enumerate(iread(input_path, index=":")):
            base = {"sample_file": input_path.name, "input_sha256": input_sha, "frame_index": frame_index, "model_id": args.model_id, "model_label": args.model_label, "model_sha256": model_sha, "scientific_success": False, "status": "failed", "energy_eV": np.nan, "max_force_eV_A": np.nan, "finite_energy": False, "finite_forces": False, "failure_reason": ""}
            try:
                atoms.calc = calc
                energy = float(atoms.get_potential_energy())
                forces = np.asarray(atoms.get_forces())
                base.update({"status": "success", "energy_eV": energy, "max_force_eV_A": float(np.max(np.linalg.norm(forces, axis=1))), "finite_energy": bool(np.isfinite(energy)), "finite_forces": bool(np.isfinite(forces).all()), "failure_reason": "" if np.isfinite(energy) and np.isfinite(forces).all() else "nonfinite_energy_or_force"})
                base["scientific_success"] = bool(base["finite_energy"] and base["finite_forces"])
            except Exception:
                base["failure_reason"] = traceback.format_exc()[-2000:]
            rows.append(base)
    fields = list(rows[0]) if rows else ["status", "model_id", "model_sha256", "failure_reason"]
    with (args.output_dir / "mace_single_point_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(json.dumps({"model_id": args.model_id, "rows": len(rows), "success": sum(r["scientific_success"] for r in rows), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__": main()
