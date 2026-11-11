#!/usr/bin/env python3
"""Audit canonical model files before they enter the static candidate screen.

The audit is intentionally narrow: verify the immutable file hash, load the
compiled model with the production MACE calculator, check finite parameters,
and evaluate one finite Cr/Sb/Te periodic probe.  It does not promote a model
into the publication registry; that promotion remains an explicit release
step after this audit and the dataset/test gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def software() -> dict[str, object]:
    try:
        import ase

        ase_version = ase.__version__
    except Exception:
        ase_version = "unavailable"
    try:
        import mace

        mace_version = getattr(mace, "__version__", "unknown")
    except Exception:
        mace_version = "unavailable"
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
    except Exception as exc:
        torch_info = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "python": sys.version.split()[0],
        "ase": ase_version,
        "mace": mace_version,
        "torch": torch_info,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }


def finite_parameter_summary(calculator) -> dict[str, object]:
    """Return finite parameter counts for common MACE calculator layouts."""

    objects = []
    for attr in ("models", "model"):
        value = getattr(calculator, attr, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            objects.extend(value)
        else:
            objects.append(value)
    seen: set[int] = set()
    parameter_count = 0
    nonfinite = 0
    modules_with_parameters = 0
    for obj in objects:
        if id(obj) in seen or not hasattr(obj, "parameters"):
            continue
        seen.add(id(obj))
        modules_with_parameters += 1
        for parameter in obj.parameters():
            array = parameter.detach().cpu().numpy()
            parameter_count += int(array.size)
            nonfinite += int(np.size(array) - np.isfinite(array).sum())
    return {
        "parameter_count": parameter_count,
        "nonfinite_parameter_values": nonfinite,
        "modules_with_parameters": modules_with_parameters,
        "parameters_finite": bool(modules_with_parameters and nonfinite == 0),
    }


def make_probe():
    from ase import Atoms

    return Atoms(
        "CrSbTe",
        positions=[[0.0, 0.0, 0.0], [2.65, 0.0, 0.0], [0.0, 2.65, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )


def audit_model(record: dict[str, object], device: str) -> dict[str, object]:
    started = time.time()
    model_id = str(record["canonical_model_id"])
    model_path = Path(str(record["model_path"]))
    result: dict[str, object] = {
        "canonical_model_id": model_id,
        "model_family": str(record.get("model_family", "")),
        "training_seed": str(record.get("training_seed", "")),
        "model_path": str(model_path),
        "expected_sha256": str(record.get("model_sha256", "")),
        "device": device,
        "status": "FAIL",
        "failure_reasons": [],
    }
    failures: list[str] = []
    if not model_path.is_file():
        failures.append("model_file_missing")
        result["failure_reasons"] = failures
        result["elapsed_s"] = time.time() - started
        return result
    actual_hash = sha256(model_path)
    result["actual_sha256"] = actual_hash
    if actual_hash != result["expected_sha256"]:
        failures.append("model_hash_mismatch")
    try:
        from mace.calculators import MACECalculator

        calculator = MACECalculator(
            model_paths=str(model_path), device=device, default_dtype="float64"
        )
        result.update(finite_parameter_summary(calculator))
        if not bool(result.get("parameters_finite", False)):
            failures.append("nonfinite_or_uninspectable_parameters")
        probe = make_probe()
        probe.calc = calculator
        energy = float(probe.get_potential_energy())
        forces = np.asarray(probe.get_forces(), dtype=float)
        result.update(
            {
                "probe_species": probe.get_chemical_symbols(),
                "probe_natoms": len(probe),
                "probe_energy_eV": energy,
                "probe_forces_shape": list(forces.shape),
                "probe_energy_finite": bool(np.isfinite(energy)),
                "probe_forces_finite": bool(np.isfinite(forces).all()),
            }
        )
        if not np.isfinite(energy):
            failures.append("nonfinite_probe_energy")
        if forces.shape != (3, 3) or not np.isfinite(forces).all():
            failures.append("invalid_probe_forces")
    except Exception as exc:
        failures.append(f"calculator_failure:{type(exc).__name__}:{exc}")
    result["failure_reasons"] = failures
    result["status"] = "PASS" if not failures else "FAIL"
    result["elapsed_s"] = time.time() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(args.registry, sep="\t", dtype=str).fillna("")
    records = []
    skipped = []
    for record in registry.to_dict("records"):
        expected = str(record.get("model_sha256", ""))
        status = str(record.get("computational_status", ""))
        if not expected or status not in {"REGISTERED", "COMPLETED_UNACCEPTED"}:
            skipped.append(
                {
                    "canonical_model_id": record.get("canonical_model_id", ""),
                    "reason": "not_ready_for_gate",
                }
            )
            continue
        records.append(audit_model(record, args.device))
    payload = {
        "campaign_id": "canonical_model_matrix_20261016",
        "audit_name": "compiled_model_gate",
        "registry": str(Path(args.registry)),
        "device": args.device,
        "software": software(),
        "models_audited": len(records),
        "models_skipped": skipped,
        "results": records,
        "status": "PASS" if records and all(r["status"] == "PASS" for r in records) else "FAIL",
    }
    (output / "MODEL_GATE_AUDIT.json").write_text(json.dumps(payload, indent=2) + "\n")
    pd.DataFrame(records).to_csv(output / "MODEL_GATE_AUDIT.tsv", sep="\t", index=False)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
