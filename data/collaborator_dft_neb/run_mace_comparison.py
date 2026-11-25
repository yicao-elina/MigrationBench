#!/usr/bin/env python3
"""Run MACE relaxation on the 9 accepted collaborator (Victor Wu) DFT endpoints
and compare relative energies / geometry against the QE reference stored in
each frame's `info` dict (energy_ry).

Usage (per model):
  python run_mace_comparison.py \
    --input processed_endpoints.extxyz \
    --model-path /data/pclancy3/yi/.../finetuned_MACE_multihead0804_compiled.model \
    --model-tag FT-600K \
    --device cuda \
    --out-root mace_comparison_runs

Run once per model (FT-600K, Foundation, Scratch, FT-MultiT) to reproduce the
four-model comparison already used for the 47-endpoint campaign
(unique_041/042 methodology).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import FIRE

RY_TO_EV = 13.605693122994

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_calculator(model_path: str, device: str):
    from mace.calculators import MACECalculator
    try:
        return MACECalculator(model_paths=model_path, device=device)
    except TypeError:
        return MACECalculator(model_path=model_path, device=device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--fmax", type=float, default=0.03)
    ap.add_argument("--out-root", type=Path, required=True)
    args = ap.parse_args()

    frames = read(args.input, index=":")
    out_dir = args.out_root / args.model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    calc = build_calculator(args.model_path, args.device)
    model_sha256 = sha256_file(Path(args.model_path)) if Path(args.model_path).exists() else None

    rows = []
    for atoms in frames:
        endpoint_id = atoms.info["endpoint_id"]
        atoms.calc = calc
        t0 = time.time()
        opt = FIRE(atoms, logfile=str(out_dir / f"{endpoint_id}.log"))
        converged = bool(opt.run(fmax=args.fmax, steps=args.steps))
        energy_ev = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        max_force = float(np.linalg.norm(forces, axis=1).max())
        relaxed = atoms.copy()
        relaxed.calc = None
        relaxed.info.update({
            "model_tag": args.model_tag,
            "mace_energy_eV": energy_ev,
            "mace_max_force_eV_A": max_force,
            "mace_converged": converged,
            "mace_steps": int(opt.nsteps),
        })
        write(out_dir / f"{endpoint_id}_mace_relaxed.extxyz", relaxed)

        dft_energy_ev = None
        if "energy_ry" in atoms.info and atoms.info["energy_ry"] is not None:
            dft_energy_ev = float(atoms.info["energy_ry"]) * RY_TO_EV

        rows.append({
            "endpoint_id": endpoint_id,
            "system": atoms.info.get("system"),
            "site": atoms.info.get("site"),
            "model_tag": args.model_tag,
            "model_path": args.model_path,
            "model_sha256": model_sha256,
            "mace_energy_eV": energy_ev,
            "mace_max_force_eV_A": max_force,
            "mace_converged": converged,
            "mace_steps": int(opt.nsteps),
            "wall_time_s": time.time() - t0,
            "dft_energy_ry": atoms.info.get("energy_ry"),
            "dft_energy_eV": dft_energy_ev,
            "input_sha256": atoms.info.get("input_sha256"),
            "output_sha256": atoms.info.get("output_sha256"),
        })

    # Relative energies within each system group (Cr/Mn/Mo), MACE vs DFT
    by_system = defaultdict(list)
    for r in rows:
        by_system[r["system"]].append(r)
    for system, group in by_system.items():
        ref = min(group, key=lambda r: r["mace_energy_eV"])
        for r in group:
            r["mace_rel_energy_eV"] = r["mace_energy_eV"] - ref["mace_energy_eV"]
        if all(r["dft_energy_eV"] is not None for r in group):
            dft_ref = min(group, key=lambda r: r["dft_energy_eV"])
            for r in group:
                r["dft_rel_energy_eV"] = r["dft_energy_eV"] - dft_ref["dft_energy_eV"]
                r["rel_energy_abs_error_eV"] = abs(r["mace_rel_energy_eV"] - r["dft_rel_energy_eV"])
        else:
            for r in group:
                r["dft_rel_energy_eV"] = None
                r["rel_energy_abs_error_eV"] = None

    summary_path = out_dir / f"{args.model_tag}_endpoint_comparison.json"
    summary_path.write_text(json.dumps({
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_tag": args.model_tag,
        "model_path": args.model_path,
        "model_sha256": model_sha256,
        "device": args.device,
        "fmax_target_eV_A": args.fmax,
        "source_extxyz": str(args.input),
        "source_extxyz_note": "processed, quality-gated Victor Wu (vwu16) endpoints; raw QE trees remain on Rockfish",
        "rows": rows,
    }, indent=2) + "\n")
    print(f"Wrote {summary_path} ({len(rows)} endpoints, {len(by_system)} systems)")


if __name__ == "__main__":
    main()
