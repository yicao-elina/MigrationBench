#!/usr/bin/env python3
"""Run the isolated Task 2 diagnostic profiles on an A100 node.

The archived NEB paths are retained as non-equilibrium snapshots.  The script
records finite energy/force probes for Foundation, the completed broad-roster
Cr fine-tune, and Cr-specific Scratch.  Because the source NEB SCF gate is not
passed, every profile is explicitly diagnostic/provisional and no transfer
gain or publication-grade DFT error is emitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from ase import Atoms


PATHS = [
    ("Cu", "neb"), ("Cu", "neb-QLs"), ("Cu", "neb-gap-diffuse"), ("Cu", "neb-gap2QL"),
    ("Li", "neb-QLs"), ("Li", "neb-gap-diffuse"), ("Li", "neb-gap2Ql"),
    ("Na", "neb-QLs"), ("Na", "neb-gap-diffuse"), ("Na", "neb-gap2QL"),
]
BOHR_TO_ANGSTROM = 0.529177210903
PERTURBATIONS = {
    "base": np.array([0.0, 0.0, 0.0]),
    "perturb_a_dx020": np.array([0.2, 0.0, 0.0]),
    "perturb_b_tilt020": np.array([-0.1, 0.14, 0.1]),
}
ATOMIC_NUMBERS = {"Cr": 24, "Cu": 29, "Li": 3, "Na": 11, "Sb": 51, "Te": 52}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_path(path: Path, elements: list[str], cell: list[list[float]]) -> list[Atoms]:
    lines = path.read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("Image:")]
    images = []
    for index, start in enumerate(starts):
        energy = float(lines[start + 1].strip())
        coords = [[float(x) * BOHR_TO_ANGSTROM for x in line.split()[:3]] for line in lines[start + 2:start + 2 + len(elements)]]
        atoms = Atoms(elements, positions=coords, cell=np.asarray(cell), pbc=True)
        atoms.info["image_index"] = index + 1
        atoms.info["dft_path_energy_eV"] = energy
        images.append(atoms)
    if len(images) != 10:
        raise ValueError(f"{path} contains {len(images)} images, expected 10")
    return images


def calculate(calculator, atoms: Atoms) -> dict:
    obj = atoms.copy()
    obj.calc = calculator
    energy = float(obj.get_potential_energy())
    forces = np.asarray(obj.get_forces(), dtype=float)
    norms = np.linalg.norm(forces, axis=1)
    return {
        "energy_eV": energy,
        "max_force_eV_A": float(np.max(norms)),
        "force_rms_eV_A": float(np.sqrt(np.mean(forces * forces))),
        "finite": bool(np.isfinite(energy) and np.isfinite(forces).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models-json", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    manifest = json.loads(args.manifest.read_text())
    model_specs = json.loads(args.models_json.read_text())
    audit = {r["path_id"]: r for r in manifest["records"]}
    started = datetime.now(timezone.utc).isoformat()
    model_records, profile_rows, failures = [], [], []

    for model_id, spec in model_specs.items():
        model_path = Path(spec["path"])
        record = {
            "model_id": model_id,
            "model_label": spec["label"],
            "model_family": spec["family"],
            "training_elements": spec["training_elements"],
            "model_path": str(model_path),
            "model_sha256": sha256(model_path),
            "software": {"python": platform.python_version(), "torch": torch.__version__, "mace": "0.3.12"},
            "device": "cuda",
            "load_status": "pending",
            "model_atomic_numbers": [],
            "model_supported_elements": [],
            "element_probe": {},
            "profile_status": "not_run",
            "scientific_success": False,
        }
        try:
            from mace.calculators import MACECalculator
            calculator = MACECalculator(model_paths=str(model_path), device="cuda", default_dtype="float32", enable_cueq=False, fullgraph=False, warmup=False)
            record["load_status"] = "loaded"
            numbers = [int(z) for z in calculator.models[0].atomic_numbers.detach().cpu().tolist()]
            record["model_atomic_numbers"] = numbers
            record["model_supported_elements"] = [e for e, z in ATOMIC_NUMBERS.items() if z in numbers]
            for element, z in ATOMIC_NUMBERS.items():
                if z not in numbers:
                    record["element_probe"][element] = {"status": "element_unsupported", "atomic_number": z}
                    continue
                probe = Atoms([element, "Sb", "Te"], positions=[[0, 0, 0], [2.5, 2.5, 2.5], [5, 5, 5]], cell=np.diag([12., 12., 12.]), pbc=True)
                try:
                    record["element_probe"][element] = {"status": "supported", **calculate(calculator, probe)}
                except Exception as exc:
                    record["element_probe"][element] = {"status": "probe_error", "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
            all_rows = 0
            for element, label in PATHS:
                path_id = f"{element}_{label}"
                source = root / "inputs" / f"{path_id}.sb2te3.path"
                dft_record = audit[path_id]
                if not source.exists():
                    failures.append({"model_id": model_id, "path_id": path_id, "status": "missing_input", "path": str(source)})
                    continue
                images = parse_path(source, dft_record["element_order"]["first_image"], dft_record["cell_A"])
                source_hash = sha256(source)
                for variant, displacement in PERTURBATIONS.items():
                    staged = []
                    for image in images:
                        obj = image.copy()
                        obj.positions[-1] += displacement
                        try:
                            result = calculate(calculator, obj)
                            staged.append({
                                "model_id": model_id, "model_label": spec["label"], "model_sha256": record["model_sha256"],
                                "path_id": path_id, "element": element, "image": image.info["image_index"],
                                "variant": variant, "perturbation_A": displacement.tolist(), **result,
                                "dft_path_energy_eV": image.info["dft_path_energy_eV"], "input_sha256": source_hash,
                                "dft_gate_status": "failed_scf_convergence", "scientific_success": False,
                                "failure_reason": "archived NEB source has SCF convergence warning; diagnostic snapshot only",
                            })
                        except Exception as exc:
                            failures.append({"model_id": model_id, "path_id": path_id, "image": image.info["image_index"], "variant": variant, "status": "calculation_error", "reason": f"{type(exc).__name__}: {str(exc)[:500]}"})
                    if staged:
                        minimum = min(row["energy_eV"] for row in staged)
                        for row in staged:
                            row["relative_energy_eV"] = row["energy_eV"] - minimum
                            profile_rows.append(row)
                            all_rows += 1
            record["profile_status"] = "complete" if all_rows == 300 else "partial_or_failed"
        except Exception as exc:
            record["load_status"] = "load_error"
            record["failure_reason"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
        model_records.append(record)

    result_dir = root / "results/broad_roster"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "model_element_probe.json").write_text(json.dumps({"created_utc": started, "records": model_records}, indent=2) + "\n", encoding="utf-8")
    fields = ["model_id", "model_label", "model_sha256", "path_id", "element", "image", "variant", "perturbation_A", "energy_eV", "relative_energy_eV", "max_force_eV_A", "force_rms_eV_A", "finite", "dft_path_energy_eV", "input_sha256", "dft_gate_status", "scientific_success", "failure_reason"]
    with (result_dir / "diagnostic_profiles.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(profile_rows)
    groups = {}
    for row in profile_rows:
        groups.setdefault((row["model_id"], row["path_id"], row["variant"]), []).append(row)
    summaries = []
    for (model_id, path_id, variant), rows in sorted(groups.items()):
        summaries.append({"model_id": model_id, "path_id": path_id, "element": rows[0]["element"], "variant": variant, "n_images": len(rows), "max_relative_energy_eV": max(float(r["relative_energy_eV"]) for r in rows), "barrier_image": max(rows, key=lambda r: float(r["relative_energy_eV"]))["image"], "max_force_eV_A": max(float(r["max_force_eV_A"]) for r in rows), "force_rms_mean_eV_A": float(np.mean([float(r["force_rms_eV_A"]) for r in rows])), "all_finite": all(r["finite"] for r in rows), "scientific_success": False, "interpretation": "provisional_non_equilibrium_snapshot_only"})
    with (result_dir / "diagnostic_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    (result_dir / "failure_retry_ledger.json").write_text(json.dumps({"created_utc": started, "records": failures}, indent=2) + "\n", encoding="utf-8")
    (result_dir / "run_manifest.json").write_text(json.dumps({"created_utc": started, "namespace": "task2_cross_element_broad_roster_20261003", "models": model_records, "profile_rows": len(profile_rows), "summary_rows": len(summaries), "failure_count": len(failures), "dft_gate_policy": "all scientific_success=false until explicit SCF and NEB convergence"}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"models": len(model_records), "loaded": sum(r["load_status"] == "loaded" for r in model_records), "profile_rows": len(profile_rows), "summary_rows": len(summaries), "failures": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
