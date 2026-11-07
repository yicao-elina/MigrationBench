#!/usr/bin/env python3
"""Evaluate one retained DFT path with one MLIP, without optimizing images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import socket
import sys
import traceback
from pathlib import Path

import numpy as np
from ase.io import read


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-head", default="default")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--path-id", required=True)
    parser.add_argument("--path-class", required=True)
    parser.add_argument("--endpoint-hash-initial", required=True)
    parser.add_argument("--endpoint-hash-final", required=True)
    parser.add_argument("--cell-hash", required=True)
    parser.add_argument("--dft-reference-id", required=True)
    parser.add_argument("--dft-reference-status", required=True)
    parser.add_argument("--expected-images", type=int, default=10)
    parser.add_argument("--expected-atoms", type=int, default=81)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    input_sha = sha256(args.input)
    model_sha = sha256(args.model_path) if args.model_path.is_file() else ""
    runtime = {
        "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "ase": version("ase"),
        "mace-torch": version("mace-torch"),
        "torch": version("torch"),
        "device": args.device,
    }
    (args.output / "runtime_manifest.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    rows: list[dict] = []
    failure: str | None = None
    try:
        from mace.calculators import MACECalculator

        calculator = MACECalculator(
            model_paths=str(args.model_path),
            device=args.device,
            default_dtype="float64",
            head=args.model_head,
        )
        images = read(args.input, index=":")
        for index, atoms in enumerate(images):
            row = {
                "record_id": args.record_id,
                "path_id": args.path_id,
                "path_class": args.path_class,
                "image_index": index,
                "atoms": len(atoms),
                "input_sha256": input_sha,
                "model_id": args.model_id,
                "model_sha256": model_sha,
                "energy_eV": "",
                "relative_energy_eV": "",
                "max_force_eV_A": "",
                "finite_energy": False,
                "finite_forces": False,
                "status": "FAILURE_MODE",
                "failure_class": "",
                "failure_reason": "",
            }
            try:
                atoms.calc = calculator
                energy = float(atoms.get_potential_energy())
                forces = np.asarray(atoms.get_forces(), dtype=float)
                row.update(
                    {
                        "energy_eV": energy,
                        "max_force_eV_A": float(np.linalg.norm(forces, axis=1).max()),
                        "finite_energy": bool(np.isfinite(energy)),
                        "finite_forces": bool(forces.shape == (len(atoms), 3) and np.isfinite(forces).all()),
                        "status": "FINITE_DIAGNOSTIC",
                    }
                )
                if not row["finite_energy"] or not row["finite_forces"]:
                    row["failure_class"] = "NONFINITE"
                    row["failure_reason"] = "nonfinite energy or force"
            except Exception:
                row["failure_class"] = "EVALUATION_EXCEPTION"
                row["failure_reason"] = traceback.format_exc()[-4000:]
            rows.append(row)
        finite = len(images) == args.expected_images and all(
            len(image) == args.expected_atoms for image in images
        ) and all(row["status"] == "FINITE_DIAGNOSTIC" for row in rows)
        energies = np.asarray([float(row["energy_eV"]) for row in rows], dtype=float)
        if finite:
            relative = energies - energies[0]
            for row, value in zip(rows, relative):
                row["relative_energy_eV"] = float(value)
            forward = float(np.max(relative))
            reverse = float(np.max(energies - energies[-1]))
            scientific_status = "FINITE_DIAGNOSTIC"
            publication_status = "SUPPORTING_EVIDENCE"
        else:
            forward = reverse = None
            scientific_status = "FAILURE_MODE"
            publication_status = "DIAGNOSTIC_ONLY"
    except Exception:
        failure = traceback.format_exc()[-4000:]
        rows = []
        forward = reverse = None
        scientific_status = "FAILURE_MODE"
        publication_status = "DIAGNOSTIC_ONLY"
    profile = args.output / "image_profile.csv"
    fields = list(rows[0]) if rows else ["record_id", "path_id", "image_index", "status", "failure_class", "failure_reason"]
    with profile.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": "migrationbench_fixed_single_point_v1",
        "record_id": args.record_id,
        "campaign_id": "neb_fixed_vs_self_matrix_20261019",
        "observable": "FIXED_PATH_SINGLE_POINT",
        "initialization_method": "DFT_RETAINED_PATH",
        "initialization_source": "DFT_RETAINED_PATH",
        "endpoint_policy": "RETAINED_DFT_ENDPOINTS",
        "path_id": args.path_id,
        "path_class": args.path_class,
        "dft_reference_id": args.dft_reference_id,
        "dft_reference_status": args.dft_reference_status,
        "endpoint_hash_initial": args.endpoint_hash_initial,
        "endpoint_hash_final": args.endpoint_hash_final,
        "cell_hash": args.cell_hash,
        "model_id": args.model_id,
        "model_label": args.model_label,
        "model_path": str(args.model_path),
        "model_sha256": model_sha,
        "model_head": args.model_head,
        "input_path": str(args.input),
        "input_sha256": input_sha,
        "expected_images": args.expected_images,
        "expected_atoms": args.expected_atoms,
        "observed_images": len(rows),
        "forward_barrier_eV": forward,
        "reverse_barrier_eV": reverse,
        "scientific_status": scientific_status,
        "publication_status": publication_status,
        "failure_class": "MODEL_LOAD" if failure else next((row["failure_class"] for row in rows if row["failure_class"]), "NA"),
        "failure_reason": failure or next((row["failure_reason"] for row in rows if row["failure_reason"]), ""),
        "runtime_manifest": str(args.output / "runtime_manifest.json"),
        "profile_artifact": str(profile),
        "profile_sha256": sha256(profile),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "created_at_utc": runtime["created_at_utc"],
    }
    (args.output / "fixed_single_point_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"record_id": args.record_id, "scientific_status": scientific_status, "forward_barrier_eV": forward}, indent=2))


if __name__ == "__main__":
    main()
