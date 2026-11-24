#!/usr/bin/env python3
"""Evaluate the MIC-aware dense fixed-path diagnostic for one model/system."""

from __future__ import annotations

import argparse
import csv
import os
import traceback
from pathlib import Path
import sys

import numpy as np
from ase import Atoms
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, model, now_utc, read_tsv, root_path, runtime_record, sha256_file, system, write_json


def reference_coordinates(path: Path) -> list[float]:
    with path.open(newline="") as handle:
        return [float(row["Reaction_Coord"]) for row in csv.DictReader(handle)]


def dense_images(images: list[Atoms], reaction: list[float], cell: np.ndarray, pbc: list[bool], segments: int) -> tuple[list[Atoms], list[float]]:
    inverse = np.linalg.inv(cell)
    result = []
    coordinates = []
    for index, (left, right) in enumerate(zip(images, images[1:])):
        left_frac = left.positions @ inverse
        right_frac = right.positions @ inverse
        delta = right_frac - left_frac
        delta -= np.floor(delta + 0.5)
        for substep in range(segments):
            fraction = substep / segments
            if substep == 0:
                positions = left.positions.copy()
            else:
                positions = (left_frac + fraction * delta) @ cell
            atoms = Atoms(symbols=left.get_chemical_symbols(), positions=positions, cell=cell, pbc=pbc)
            result.append(atoms)
            coordinates.append(reaction[index] + fraction * (reaction[index + 1] - reaction[index]))
    last = images[-1].copy()
    last.set_cell(cell)
    last.set_pbc(pbc)
    result.append(last)
    coordinates.append(reaction[-1])
    return result, coordinates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-table", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config, root = load_config(args.config)
    row = read_tsv(args.job_table)[args.index]
    model_id = row["model_id"]
    system_id = row["system_id"]
    spec = model(config, model_id)
    system_spec = system(config, system_id)
    input_path = root_path(root, system_spec["xyz"])
    output = root / "runs" / "dense" / model_id / system_id / f"attempt_{os.environ.get('SLURM_JOB_ID', f'LOCAL_{args.index}') }"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "sliding_dense_fixed_path_run_v1",
        "campaign_id": config["campaign_id"],
        "observable": "FIXED_PATH_DENSE_DIAGNOSTIC",
        "initialization_method": "MIC_AWARE_SEGMENT_INTERPOLATION",
        "initialization_source": "retained DFT/QE path images",
        "model_id": model_id,
        "model_label": spec["label"],
        "model_family": spec["family"],
        "scratch_subfamily": spec.get("scratch_subfamily", "none"),
        "training_seed": spec.get("training_seed", "NA"),
        "model_path": spec["model_path"],
        "expected_model_sha256": spec["model_sha256"],
        "system_id": system_id,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path) if input_path.is_file() else "",
        "cell_angstrom": system_spec["cell_angstrom"],
        "pbc": system_spec["pbc"],
        "segments_per_dft_interval": config["protocol"]["dense_segments"],
        "dft_reference_status": config["protocol"]["dft_reference_status"],
        "runtime": runtime_record(),
        "created_at_utc": now_utc(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "LOCAL"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "LOCAL"),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    model_path = Path(spec["model_path"])
    actual_model_sha = sha256_file(model_path) if model_path.is_file() else ""
    manifest["actual_model_sha256"] = actual_model_sha
    manifest["model_hash_match"] = bool(actual_model_sha and actual_model_sha == spec["model_sha256"])
    dense_rows = []
    try:
        images = read(input_path, index=":")
        reaction = reference_coordinates(root_path(root, system_spec["dft_csv"]))
        cell = np.asarray(system_spec["cell_angstrom"], dtype=float)
        dense, dense_coord = dense_images(images, reaction, cell, system_spec["pbc"], config["protocol"]["dense_segments"])
        manifest["observed_dense_images"] = len(dense)
        if not manifest["model_hash_match"]:
            raise RuntimeError(f"model hash mismatch expected={spec['model_sha256']} actual={actual_model_sha}")
        from mace.calculators import MACECalculator
        calculator = MACECalculator(
            model_paths=str(model_path),
            device=args.device or config["environment"]["device"],
            default_dtype=config["environment"]["default_dtype"],
            head="default",
        )
        for image_index, (atoms, coord) in enumerate(zip(dense, dense_coord)):
            record = {
                "model_id": model_id,
                "system_id": system_id,
                "dense_index": image_index,
                "reaction_coordinate": coord,
                "atoms": len(atoms),
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
                finite_energy = bool(np.isfinite(energy))
                finite_forces = bool(forces.shape == (len(atoms), 3) and np.isfinite(forces).all())
                record.update({
                    "energy_eV": energy,
                    "max_force_eV_A": float(np.linalg.norm(forces, axis=1).max()) if finite_forces else "",
                    "finite_energy": finite_energy,
                    "finite_forces": finite_forces,
                    "status": "FINITE_DIAGNOSTIC" if finite_energy and finite_forces else "FAILURE_MODE",
                    "failure_class": "" if finite_energy and finite_forces else "NONFINITE",
                    "failure_reason": "" if finite_energy and finite_forces else "non-finite energy or forces",
                })
            except Exception:
                record.update({"failure_class": "EVALUATION_EXCEPTION", "failure_reason": traceback.format_exc()[-6000:]})
            dense_rows.append(record)
        finite = len(dense_rows) == len(dense) and all(item["status"] == "FINITE_DIAGNOSTIC" for item in dense_rows)
        if finite:
            energies = np.asarray([float(item["energy_eV"]) for item in dense_rows], dtype=float)
            for item, value in zip(dense_rows, energies - energies[0]):
                item["relative_energy_eV"] = float(value)
            manifest.update({
                "status": "PASS",
                "scientific_status": "DIAGNOSTIC_ONLY",
                "publication_status": "SUPPORTING_DIAGNOSTIC",
                "failure_class": "",
                "failure_reason": "",
            })
        else:
            manifest.update({
                "status": "FAILURE_MODE",
                "scientific_status": "FAILURE_MODE",
                "publication_status": "DIAGNOSTIC_ONLY",
                "failure_class": next((item["failure_class"] for item in dense_rows if item["failure_class"]), "FINITE_GATE"),
                "failure_reason": next((item["failure_reason"] for item in dense_rows if item["failure_reason"]), "non-finite dense evaluation"),
            })
    except Exception:
        manifest.update({
            "status": "FAILURE_MODE",
            "scientific_status": "FAILURE_MODE",
            "publication_status": "DIAGNOSTIC_ONLY",
            "failure_class": "DENSE_EVALUATION_EXCEPTION",
            "failure_reason": traceback.format_exc()[-6000:],
        })
    fields = [
        "model_id", "system_id", "dense_index", "reaction_coordinate", "atoms",
        "energy_eV", "relative_energy_eV", "max_force_eV_A", "finite_energy",
        "finite_forces", "status", "failure_class", "failure_reason",
    ]
    with (output / "dense_profile.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(dense_rows)
    manifest["profile_artifact"] = str(output / "dense_profile.csv")
    manifest["profile_sha256"] = sha256_file(output / "dense_profile.csv")
    write_json(output / "dense_manifest.json", manifest)
    print(f"{manifest['status']} {model_id} {system_id} {output}")
    if manifest["status"] != "PASS":
        raise SystemExit(5)


if __name__ == "__main__":
    main()

