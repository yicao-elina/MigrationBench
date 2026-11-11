#!/usr/bin/env python3
"""Evaluate one fixed-path row with MACE energies, forces, and finite checks."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import os
import traceback
from pathlib import Path
import sys

import numpy as np
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import file_record, load_config, model, now_utc, read_tsv, root_path, runtime_record, sha256_file, system, write_json


def write_profile(path: Path, rows: list[dict]) -> None:
    fields = [
        "model_id", "system_id", "image_index", "atoms", "energy_eV",
        "relative_energy_eV", "max_force_eV_A", "finite_energy", "finite_forces",
        "status", "failure_class", "failure_reason",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def base_manifest(config: dict, root: Path, model_id: str, system_id: str, input_path: Path, output: Path) -> dict:
    spec = model(config, model_id)
    system_spec = system(config, system_id)
    return {
        "schema_version": "sliding_fixed_path_run_v1",
        "campaign_id": config["campaign_id"],
        "phase": config["phase"],
        "observable": config["observable"],
        "initialization_method": "DFT_RETAINED_PATH",
        "initialization_source": "exact retained DFT/QE image file",
        "endpoint_policy": "RETAINED_DFT_ENDPOINTS",
        "optimization": "none",
        "model_id": model_id,
        "model_label": spec["label"],
        "model_family": spec["family"],
        "scratch_subfamily": spec.get("scratch_subfamily", "none"),
        "training_seed": spec.get("training_seed", "NA"),
        "seed_semantics": spec.get("seed_semantics", "not_applicable"),
        "model_path": spec["model_path"],
        "expected_model_sha256": spec["model_sha256"],
        "model_provenance_status": spec.get("provenance_status", "PROVENANCE_PARTIAL"),
        "publication_status": spec["publication_status"],
        "system_id": system_id,
        "system_label": system_spec["label"],
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path) if input_path.is_file() else "",
        "expected_images": config["protocol"]["fixed_images"],
        "expected_atoms": system_spec["atoms"],
        "dft_reference_id": system_spec["dft_column"],
        "dft_reference_status": config["protocol"]["dft_reference_status"],
        "dft_reference_method": config["protocol"]["dft_reference_method"],
        "runtime": runtime_record(),
        "created_at_utc": now_utc(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "LOCAL"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "LOCAL"),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-table", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config, root = load_config(args.config)
    rows = read_tsv(args.job_table)
    row = rows[args.index]
    model_id = row["model_id"]
    system_id = row["system_id"]
    spec = model(config, model_id)
    system_spec = system(config, system_id)
    input_path = root_path(root, system_spec["xyz"])
    output = root / "runs" / "fixed" / model_id / system_id / f"attempt_{os.environ.get('SLURM_JOB_ID', f'LOCAL_{args.index}') }"
    output.mkdir(parents=True, exist_ok=True)
    manifest = base_manifest(config, root, model_id, system_id, input_path, output)
    manifest["job_row"] = row
    model_path = Path(spec["model_path"])
    actual_model_sha = sha256_file(model_path) if model_path.is_file() else ""
    manifest["actual_model_sha256"] = actual_model_sha
    manifest["model_hash_match"] = bool(actual_model_sha and actual_model_sha == spec["model_sha256"])
    profile_rows: list[dict] = []
    failure = None
    images = []
    try:
        images = read(input_path, index=":")
    except Exception:
        failure = traceback.format_exc()[-6000:]
    if not manifest["model_hash_match"]:
        manifest.update({
            "status": "MODEL_HASH_MISMATCH" if model_path.is_file() else "MODEL_MISSING",
            "scientific_status": "FAILURE_MODE",
            "publication_status": "DIAGNOSTIC_ONLY",
            "failure_class": "MODEL_HASH_MISMATCH" if model_path.is_file() else "MODEL_MISSING",
            "failure_reason": f"expected={spec['model_sha256']} actual={actual_model_sha}",
        })
        write_profile(output / "image_profile.csv", profile_rows)
        manifest["profile_artifact"] = str(output / "image_profile.csv")
        manifest["profile_sha256"] = sha256_file(output / "image_profile.csv")
        write_json(output / "fixed_manifest.json", manifest)
        print(manifest["status"])
        raise SystemExit(3)
    if failure:
        manifest.update({
            "status": "INPUT_READ_FAILURE",
            "scientific_status": "FAILURE_MODE",
            "publication_status": "DIAGNOSTIC_ONLY",
            "failure_class": "INPUT_READ_FAILURE",
            "failure_reason": failure,
        })
        write_profile(output / "image_profile.csv", profile_rows)
        manifest["profile_artifact"] = str(output / "image_profile.csv")
        manifest["profile_sha256"] = sha256_file(output / "image_profile.csv")
        write_json(output / "fixed_manifest.json", manifest)
        print(manifest["status"])
        raise SystemExit(4)
    try:
        from mace.calculators import MACECalculator

        device = args.device or config["environment"]["device"]
        calculator = MACECalculator(
            model_paths=str(model_path),
            device=device,
            default_dtype=config["environment"]["default_dtype"],
            head="default",
        )
        for image_index, atoms in enumerate(images):
            record = {
                "model_id": model_id,
                "system_id": system_id,
                "image_index": image_index,
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
            profile_rows.append(record)
        finite = (
            len(images) == config["protocol"]["fixed_images"]
            and all(len(image) == system_spec["atoms"] for image in images)
            and len(profile_rows) == len(images)
            and all(row["status"] == "FINITE_DIAGNOSTIC" for row in profile_rows)
        )
        if finite:
            energies = np.asarray([float(item["energy_eV"]) for item in profile_rows], dtype=float)
            relative = energies - energies[0]
            for item, value in zip(profile_rows, relative):
                item["relative_energy_eV"] = float(value)
            manifest.update({
                "status": "PASS",
                "scientific_status": "FINITE_DIAGNOSTIC",
                "publication_status": "SUPPORTING_EVIDENCE",
                "failure_class": "",
                "failure_reason": "",
                "observed_images": len(images),
                "observed_atoms": sorted({len(image) for image in images}),
                "max_relative_energy_eV": float(np.max(relative)),
                "peak_image_index": int(np.argmax(relative)),
                "endpoint_drift_eV": float(relative[-1]),
            })
        else:
            manifest.update({
                "status": "FAILURE_MODE",
                "scientific_status": "FAILURE_MODE",
                "publication_status": "DIAGNOSTIC_ONLY",
                "failure_class": next((item["failure_class"] for item in profile_rows if item["failure_class"]), "SHAPE_OR_FINITE_GATE"),
                "failure_reason": next((item["failure_reason"] for item in profile_rows if item["failure_reason"]), "wrong image/atom count or non-finite result"),
                "observed_images": len(images),
                "observed_atoms": sorted({len(image) for image in images}),
            })
    except Exception:
        manifest.update({
            "status": "MODEL_LOAD_FAILURE",
            "scientific_status": "FAILURE_MODE",
            "publication_status": "DIAGNOSTIC_ONLY",
            "failure_class": "MODEL_LOAD_FAILURE",
            "failure_reason": traceback.format_exc()[-6000:],
            "observed_images": len(images),
        })
    write_profile(output / "image_profile.csv", profile_rows)
    manifest["profile_artifact"] = str(output / "image_profile.csv")
    manifest["profile_sha256"] = sha256_file(output / "image_profile.csv")
    write_json(output / "fixed_manifest.json", manifest)
    print(f"{manifest['status']} {model_id} {system_id} {output}")
    if manifest["status"] != "PASS":
        raise SystemExit(5)


if __name__ == "__main__":
    main()

