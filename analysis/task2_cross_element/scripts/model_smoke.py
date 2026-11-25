#!/usr/bin/env python3
"""Model loading, element-support, and one diagnostic path profile smoke.

This is a health gate only.  The representative Cu DFT path failed the DFT
convergence gate, so its MACE results are diagnostic and are not publication
fidelity or transfer metrics.
"""

from __future__ import annotations

import hashlib
import json
import platform
import os
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from ase import Atoms


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("MIGRATIONBENCH_MODEL_DIR", os.environ.get("MIGRATIONBENCH_MODEL_ARCHIVE", "/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_git_large_assets_20261124/models")))
def resolve_model_path(filename):
    return MODEL_DIR / filename

MODELS = {
    "foundation": {"label": "Foundation", "family": "Foundation", "training_elements": ["broad/foundation; exact training roster not present in local manifest"], "path": resolve_model_path("Foundation-MACE-matpes-pbe-omat-ft.model")},
    "cr_ft": {"label": "Cr-specific FT", "family": "Cr-specific FT-600K", "training_elements": ["Cr", "Sb", "Te"], "path": resolve_model_path("FT-600K.model")},
    "cr_scratch": {"label": "Cr-specific Scratch", "family": "Cr-specific Scratch", "training_elements": ["Cr", "Sb", "Te"], "path": resolve_model_path("Scratch_Multi_T.model")},
    "multi_element_ft": {"label": "Multi-element FT", "family": "Multi-element FT-MultiT", "training_elements": ["multi-element; exact roster not present in local manifest"], "path": resolve_model_path("FT_Multi_T.model")},
}
ELEMENTS = ["Cr", "Cu", "Li", "Na", "Sb", "Te"]
ATOMIC_NUMBERS = {"Cr": 24, "Cu": 29, "Li": 3, "Na": 11, "Sb": 51, "Te": 52}
BOHR_TO_ANGSTROM = 0.529177210903


def canonical_hash(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def probe_atoms(symbols: list[str]) -> Atoms:
    n = len(symbols)
    positions = np.array([[1.2 + 2.0 * (i % 3), 1.3 + 2.0 * ((i // 3) % 2), 2.0 + 1.5 * (i // 6)] for i in range(n)], float)
    return Atoms(symbols, positions=positions, cell=np.diag([12.0, 12.0, 16.0]), pbc=True)


def load_calculator(path: Path):
    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=str(path), device="cpu", default_dtype="float32", enable_cueq=False, fullgraph=False, warmup=False)


def calculate(calculator, atoms: Atoms) -> dict:
    atoms = atoms.copy()
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    return {"energy_eV": energy, "max_force_eV_A": float(np.max(np.linalg.norm(forces, axis=1))), "finite": bool(np.isfinite(energy) and np.isfinite(forces).all())}


def parse_path(path: Path, elements: list[str], cell: list[list[float]]) -> list[Atoms]:
    lines = path.read_text().splitlines()
    image_indexes = [i for i, line in enumerate(lines) if line.startswith("Image:")]
    if len(image_indexes) < 2:
        raise ValueError(f"could not find path images in {path}")
    images = []
    for image_index, start in enumerate(image_indexes):
        energy_line = lines[start + 1].strip()
        coords = []
        for line in lines[start + 2 : start + 2 + len(elements)]:
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"malformed coordinate line at image {image_index + 1}: {line}")
            coords.append([float(x) * BOHR_TO_ANGSTROM for x in fields[:3]])
        if len(coords) != len(elements):
            raise ValueError(f"image {image_index + 1} has {len(coords)} coordinates, expected {len(elements)}")
        atoms = Atoms(elements, positions=coords, cell=np.asarray(cell), pbc=True)
        atoms.info["dft_path_energy_eV"] = float(energy_line)
        atoms.info["image_index"] = image_index + 1
        images.append(atoms)
    return images


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    manifest = json.loads((ROOT / "manifests/dft_neb_provenance_manifest.json").read_text())
    representative = next(r for r in manifest["records"] if r["path_id"] == "Cu_neb")
    path_copy = ROOT / "inputs/Cu_neb.sb2te3.path"
    path_copy.parent.mkdir(parents=True, exist_ok=True)
    if not path_copy.exists():
        raise FileNotFoundError(f"expected read-only source copy: {path_copy}")
    elements = representative["element_order"]["first_image"]
    images = parse_path(path_copy, elements, representative["cell_A"])
    source_hash = hashlib.sha256(path_copy.read_bytes()).hexdigest()
    results = []
    for model_id, spec in MODELS.items():
        row = {"model_id": model_id, "model_label": spec["label"], "model_family": spec["family"], "training_elements": spec["training_elements"], "model_path": str(spec["path"]), "model_sha256": hashlib.sha256(spec["path"].read_bytes()).hexdigest(), "software": {"python": platform.python_version(), "torch": torch.__version__}, "device": "cpu", "load_status": None, "element_support": {}, "representative_path": {"path_id": "Cu_neb", "input_sha256": source_hash, "dft_gate_status": "failed: no explicit NEB convergence and SCF warning on image 2", "image_count": len(images), "records": []}, "scientific_success": False, "failure_reason": None}
        try:
            t0 = time.time()
            calculator = load_calculator(spec["path"])
            row["load_status"] = "loaded"
            row["load_seconds"] = time.time() - t0
            model_atomic_numbers = [int(z) for z in calculator.models[0].atomic_numbers.detach().cpu().tolist()]
            row["model_atomic_numbers"] = model_atomic_numbers
            row["model_supported_elements"] = [element for element, z in ATOMIC_NUMBERS.items() if z in model_atomic_numbers]
            for element in ELEMENTS:
                if ATOMIC_NUMBERS[element] not in model_atomic_numbers:
                    row["element_support"][element] = {"status": "element_unsupported", "atomic_number": ATOMIC_NUMBERS[element], "reason": "element absent from model atomic_numbers metadata"}
                    continue
                try:
                    result = calculate(calculator, probe_atoms([element, "Sb", "Te"] if element not in {"Sb", "Te"} else [element, "Cr", "Te"]))
                    row["element_support"][element] = {"status": "supported", **result}
                except Exception as exc:
                    message = str(exc).replace("\n", " ")[:400]
                    row["element_support"][element] = {"status": "element_unsupported" if re.search(r"element|atomic|species|z=|type", message, re.I) else "probe_error", "error": f"{type(exc).__name__}: {message}"}
            if row["element_support"].get("Cu", {}).get("status") == "supported":
                for image in images:
                    try:
                        result = calculate(calculator, image)
                        result["image_index"] = image.info["image_index"]
                        result["dft_path_energy_eV"] = image.info["dft_path_energy_eV"]
                        row["representative_path"]["records"].append(result)
                    except Exception as exc:
                        row["representative_path"]["records"].append({"image_index": image.info["image_index"], "status": "error", "error": f"{type(exc).__name__}: {str(exc)[:400]}"})
            row["representative_path"]["profile_status"] = "complete" if len(row["representative_path"]["records"]) == len(images) and all(r.get("finite", False) for r in row["representative_path"]["records"]) else "failed_or_unsupported"
        except Exception as exc:
            row["load_status"] = "load_error"
            row["failure_reason"] = f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:500]}"
        if row["failure_reason"] is None and row["load_status"] != "loaded":
            row["failure_reason"] = "model_load_failed"
        results.append(row)
    output = {"schema_version": "task2-model-smoke-1.0", "created_utc": started, "namespace": "task2_cross_element", "scope": "health_gate_only; no DFT path passed", "representative_input_sha256": source_hash, "records": results}
    out = ROOT / "smoke/model_element_support_and_profile.json"
    out.write_text(json.dumps(output, indent=2) + "\n")
    rows = []
    for r in results:
        for element, support in r["element_support"].items():
            rows.append({"model_id": r["model_id"], "model_label": r["model_label"], "model_sha256": r["model_sha256"], "element": element, "status": support["status"], "energy_eV": support.get("energy_eV"), "max_force_eV_A": support.get("max_force_eV_A"), "error": support.get("error", "")})
    import csv

    with (ROOT / "smoke/model_element_support.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"models": len(results), "loaded": sum(r["load_status"] == "loaded" for r in results), "representative_profiles": {r["model_id"]: r["representative_path"].get("profile_status") for r in results}}, indent=2))


if __name__ == "__main__":
    main()
