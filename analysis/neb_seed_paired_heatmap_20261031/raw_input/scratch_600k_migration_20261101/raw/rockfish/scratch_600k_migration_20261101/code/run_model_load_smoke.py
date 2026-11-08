#!/usr/bin/env python3
"""Finite-energy/force checkpoint smoke; never counts as a campaign row."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.io import read


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-table", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    with args.model_table.open() as handle:
        row = list(csv.DictReader(handle, delimiter="\t"))[args.index]
    out = args.out_root / row["model_id"]
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "migrationbench_model_load_smoke_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "model_id": row["model_id"],
        "model_path": row["model_path"],
        "model_sha256_expected": row["model_sha256"],
        "model_sha256_observed": sha256(Path(row["model_path"])) if Path(row["model_path"]).is_file() else "",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "status": "FAILURE_MODE",
        "failure_class": "",
    }
    try:
        from mace.calculators import MACECalculator

        calculator = MACECalculator(model_paths=row["model_path"], device=args.device, default_dtype="float64", head=row["model_head"])
        atoms = read(args.input)
        atoms.calc = calculator
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        record.update({
            "energy_eV": energy,
            "max_force_eV_A": float(np.linalg.norm(forces, axis=1).max()),
            "finite_energy": bool(np.isfinite(energy)),
            "finite_forces": bool(forces.shape == (len(atoms), 3) and np.isfinite(forces).all()),
        })
        record["status"] = "PASS" if record["finite_energy"] and record["finite_forces"] else "FAILURE_MODE"
        if record["status"] != "PASS":
            record["failure_class"] = "NONFINITE"
    except Exception:
        record["failure_class"] = "DESERIALIZATION" if any(token in traceback.format_exc().lower() for token in ("deserial", "attributeerror", "torch.jit", "unknown class")) else "LOAD_EXCEPTION"
        record["failure_reason"] = traceback.format_exc()[-6000:]
    (out / "model_load_smoke.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"model_id": row["model_id"], "status": record["status"], "failure_class": record.get("failure_class", "")}, indent=2))


if __name__ == "__main__":
    main()
