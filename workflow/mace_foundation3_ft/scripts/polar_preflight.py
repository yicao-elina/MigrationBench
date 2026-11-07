#!/usr/bin/env python3
"""Run the PolarMACE runtime/model/data-contract preflight without training."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from ase.io import read


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version(name: str):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    root = Path(cfg["remote_root"])
    output = root / "ledger" / "polar_preflight.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite preflight evidence: {output}")
    foundation = cfg["foundation_models"]["polar1m"]
    model_path = Path(foundation["source"])
    md_input = Path(cfg["md"]["input"])
    result = {
        "schema_version": "1.0",
        "campaign_id": cfg["campaign_id"],
        "foundation_name": foundation["canonical_name"],
        "foundation_path": str(model_path),
        "foundation_sha256_expected": foundation["sha256"],
        "foundation_sha256_actual": sha256(model_path) if model_path.is_file() else None,
        "foundation_public_url": foundation["public_url"],
        "model_type": foundation["model_type"],
        "input": str(md_input),
        "input_sha256_expected": cfg["md"]["input_sha256"],
        "input_sha256_actual": sha256(md_input) if md_input.is_file() else None,
        "device": args.device,
        "runtime": {
            "python": sys.version.replace("\n", " "),
            "python_executable": sys.executable,
            "mace_torch": version("mace-torch"),
            "torch": version("torch"),
            "ase": version("ase"),
            "graph_longrange": version("graph_longrange"),
        },
        "loader_import": False,
        "model_load": False,
        "static_energy_forces": None,
        "contract": {},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import torch
        from mace.calculators import mace_polar

        result["loader_import"] = True
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["foundation_sha256_actual"] != foundation["sha256"]:
            raise RuntimeError("foundation model SHA256 mismatch")
        if result["input_sha256_actual"] != cfg["md"]["input_sha256"]:
            raise RuntimeError("MD input SHA256 mismatch")
        calculator = mace_polar(model=str(model_path), device=args.device, default_dtype="float32", pbc_handling="auto")
        result["model_load"] = True
    except Exception as exc:
        result["status"] = "blocked"
        result["failure_reason"] = f"model_runtime_error: {type(exc).__name__}: {exc}"
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        raise SystemExit(42)

    atoms = read(str(md_input), index=int(cfg["md"].get("input_frame", -1)), format="extxyz")
    required = ("charge", "spin", "external_field")
    present = {key: key in atoms.info for key in required}
    result["contract"] = {
        "required_structure_info": list(required),
        "input_info_keys": sorted(atoms.info),
        "present": present,
        "natoms": len(atoms),
        "pbc": [bool(value) for value in atoms.pbc],
        "positions_finite": bool(np.isfinite(atoms.get_positions()).all()),
        "cell_finite": bool(np.isfinite(atoms.get_cell().array).all()),
    }
    missing = [key for key, value in present.items() if not value]
    if missing:
        result["status"] = "blocked"
        result["publication_status"] = "EXCLUDE_FROM_PUBLICATION"
        result["failure_reason"] = "missing_required_charge_spin_external_field_contract: " + ",".join(missing)
        result["scientific_action"] = "do not fabricate metadata; require a documented compatibility adapter or stop Polar branch"
    else:
        try:
            atoms.calc = calculator
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            stress = np.asarray(atoms.get_stress(), dtype=float)
            finite = bool(math.isfinite(energy) and np.isfinite(forces).all() and np.isfinite(stress).all())
            result["static_energy_forces"] = {
                "energy_eV": energy,
                "max_force_eV_A": float(np.abs(forces).max()),
                "stress_finite": bool(np.isfinite(stress).all()),
                "finite": finite,
            }
            result["status"] = "passed" if finite else "scientifically_failed"
        except Exception as exc:
            result["status"] = "scientifically_failed"
            result["failure_reason"] = f"static_evaluation_error: {type(exc).__name__}: {exc}"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(42)


if __name__ == "__main__":
    main()
