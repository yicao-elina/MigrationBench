#!/usr/bin/env python3
"""Run Foundation-MACE profiles on all archived NEB images plus two perturbations.

The input is intentionally the archived, non-equilibrium DFT path.  Every row
is labelled provisional and scientific_success=False until the DFT SCF/NEB
gate is repaired.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase import Atoms

from model_smoke import calculate, load_calculator, resolve_model_path, resolve_model_path

ROOT = Path(__file__).resolve().parents[1]
PERTURBATIONS = {"base": np.array([0.0, 0.0, 0.0]), "perturb_a_dx020": np.array([0.2, 0.0, 0.0]), "perturb_b_tilt020": np.array([-0.1, 0.14, 0.1])}
BOHR_TO_ANGSTROM = 0.529177210903


def parse_path(path: Path, elements, cell):
    lines = path.read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("Image:")]
    images = []
    for index, start in enumerate(starts):
        rows = lines[start + 2 : start + 2 + len(elements)]
        coords = [[float(x) * BOHR_TO_ANGSTROM for x in line.split()[:3]] for line in rows]
        image = Atoms(elements, positions=coords, cell=np.asarray(cell), pbc=True)
        image.info["image_index"] = index + 1
        image.info["dft_path_energy_eV"] = float(lines[start + 1].strip())
        images.append(image)
    return images


def main():
    manifest = json.loads((ROOT / "manifests/dft_neb_provenance_manifest.json").read_text())
    audit = {r["path_id"]: r for r in json.loads((ROOT / "results/provisional_neb_iteration_audit.json").read_text())["records"]}
    model_path = resolve_model_path("Foundation-MACE-matpes-pbe-omat-ft.model")
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    calculator = load_calculator(model_path)
    rows = []
    failures = []
    for record in manifest["records"]:
        path_copy = ROOT / "inputs" / f"{record['path_id']}.sb2te3.path"
        if not path_copy.exists():
            failures.append({"path_id": record["path_id"], "status": "missing_input_copy", "reason": str(path_copy)})
            continue
        images = parse_path(path_copy, record["element_order"]["first_image"], record["cell_A"])
        path_sha = hashlib.sha256(path_copy.read_bytes()).hexdigest()
        for variant, displacement in PERTURBATIONS.items():
            energies = []
            staged = []
            for image in images:
                atoms = image.copy()
                atoms.positions[-1] += displacement
                try:
                    result = calculate(calculator, atoms)
                    result.update({"path_id": record["path_id"], "element": record["element"], "image": image.info["image_index"], "variant": variant, "perturbation_A": displacement.tolist(), "dft_path_energy_eV": image.info["dft_path_energy_eV"], "model_id": "foundation", "model_sha256": model_sha, "input_sha256": hashlib.sha256((path_sha + variant + str(image.info["image_index"])).encode()).hexdigest(), "dft_gate_status": "failed_scf_convergence", "provisional_snapshot_usable": audit[record["path_id"]]["provisional_snapshot_usable"], "scientific_success": False, "failure_reason": "archived DFT path has SCF convergence warning; provisional single-point only"})
                    energies.append(result["energy_eV"])
                    staged.append(result)
                except Exception as exc:
                    failures.append({"path_id": record["path_id"], "image": image.info["image_index"], "variant": variant, "status": "mace_calculation_error", "reason": f"{type(exc).__name__}: {str(exc)[:400]}"})
            if staged:
                minimum = min(energies)
                for row in staged:
                    row["relative_energy_eV"] = row["energy_eV"] - minimum
                    rows.append(row)
    out = ROOT / "results/provisional_mace_profiles.csv"
    fields = ["model_id", "model_sha256", "path_id", "element", "image", "variant", "perturbation_A", "energy_eV", "relative_energy_eV", "max_force_eV_A", "finite", "dft_path_energy_eV", "input_sha256", "dft_gate_status", "provisional_snapshot_usable", "scientific_success", "failure_reason"]
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (ROOT / "results/provisional_mace_failure_retry_ledger.json").write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "model_id": "foundation", "records": failures}, indent=2) + "\n")
    (ROOT / "results/provisional_mace_profile_manifest.json").write_text(json.dumps({"schema_version": "task2-provisional-mace-profile-1.0", "created_utc": datetime.now(timezone.utc).isoformat(), "model_id": "foundation", "model_sha256": model_sha, "variants": {k: v.tolist() for k, v in PERTURBATIONS.items()}, "input_namespace": "task2_cross_element/inputs", "dft_gate_policy": "scientific_success false for all rows until SCF and NEB convergence are explicitly repaired", "rows": len(rows), "failures": len(failures)}, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "failures": len(failures), "paths": len(set(r["path_id"] for r in rows)), "variants": len(set(r["variant"] for r in rows))}, indent=2))


if __name__ == "__main__":
    main()
