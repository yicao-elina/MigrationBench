#!/usr/bin/env python3
"""Build the isolated Victor four-arm benchmark manifest.

This deliberately records model paths and hashes from the existing model
inventory; it does not invent or silently replace model assets.
"""
from __future__ import annotations

import csv, hashlib, json, shutil
from pathlib import Path
from ase.io import read, write

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / "data" / "collaborator_dft_neb"
EXTXYZ = DATA / "processed_endpoints.extxyz"
PROV = DATA / "endpoint_provenance.csv"
RAW_QE = ROOT / "raw_qe"
MODELS = Path(__file__).resolve().parents[1] / "mace_candidate_eval_20261001" / "models_rockfish.tsv"

VARIANTS = {
    "base": (0.0, 0.0, 0.0),
    "perturb_a_dx020": (0.20, 0.0, 0.0),
    "perturb_b_tilt020": (-0.10, 0.14, 0.10),
}

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def main() -> None:
    (ROOT / "inputs").mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    frames = read(EXTXYZ, index=":")
    provenance = {r["endpoint_id"]: r for r in csv.DictReader(PROV.open())}
    tasks = []
    for frame in frames:
        eid = frame.info["endpoint_id"]
        base = frame.copy()
        qe_path = RAW_QE / f"{eid}.in"
        dft_input = read(qe_path, format="espresso-in")
        # Existing processed endpoint is the DFT-relaxed output/reference.
        out_name = f"{eid}.dft_relaxed.extxyz"
        out_path = ROOT / "inputs" / out_name
        write(out_path, base)
        for source, source_atoms in (("dft_relaxed", base), ("dft_input", dft_input)):
            for variant, delta in VARIANTS.items():
                if source == "dft_relaxed" and variant != "base":
                    continue
                atoms = source_atoms.copy()
                if variant != "base":
                    dopants = [s for s in ("Cr", "Mn", "Mo") if s in frame.symbols]
                    if len(dopants) != 1:
                        raise RuntimeError(f"Expected one dopant species for {eid}: {dopants}")
                    dopant = dopants[0]
                    cr = [i for i, s in enumerate(atoms.symbols) if s == dopant]
                    if len(cr) != 1:
                        raise RuntimeError(f"Cr/Mn/Mo identity not unique for {eid}: {cr}")
                    atoms.positions[cr[0]] += delta
                name = f"{eid}.{source}.{variant}.extxyz"
                path = ROOT / "inputs" / name
                write(path, atoms)
                task_id = f"{eid}__{source}__{variant}"
                task = {"task_id": task_id, "endpoint_id": eid,
                    "source_type": source, "variant": variant, "input_path": str(path),
                    "input_extxyz": name,
                    "input_sha256": sha256(path), "dft_reference_path": out_name,
                    "perturbation_A": list(delta), "perturbation_norm_A": sum(x*x for x in delta) ** 0.5,
                    "n_atoms": len(atoms), "formula": atoms.get_chemical_formula(),
                    "cell_A": atoms.cell.array.tolist(), "pbc": atoms.pbc.tolist(),
                    "raw_input_path": provenance[eid]["raw_input_path"],
                    "raw_output_path": provenance[eid]["raw_output_path"]}
                tasks.append(task)
                (ROOT / "task_json").mkdir(exist_ok=True)
                (ROOT / "task_json" / f"{task_id}.json").write_text(json.dumps({
                    "task_id": task_id, "endpoint_id": eid, "source_type": source,
                    "variant": variant, "input_extxyz": name,
                    "input_extxyz_sha256": task["input_sha256"]}, sort_keys=True) + "\n")
    models = [dict(r) for r in csv.DictReader(MODELS.open(), delimiter="\t")]
    manifest = {"schema_version": "1.0", "campaign": "victor_dft_reference_multi_arm_mace",
        "created": "2026-10-01", "source_of_truth": {
            "perturbation_script": "/Users/alina/Documents/Codex/2026-09-25/cr-sb2te3-mace-model-comparison/work/campaign_code/prepare_campaign.py",
            "runner": "/Users/alina/Documents/Codex/2026-09-25/cr-sb2te3-mace-model-comparison/work/campaign_code/run_endpoint_relax.py",
            "processed_endpoints": str(EXTXYZ), "processed_endpoints_sha256": sha256(EXTXYZ)},
        "protocol": {"optimizer": "FIRE", "fmax_eV_A": 0.03, "max_steps": 500, "cell_relaxation": False},
        "expected": {"endpoints": len(frames), "source_variant_tasks": len(tasks), "models": len(models), "runs": len(tasks)*len(models)},
        "models": models, "tasks": tasks}
    (ROOT / "provenance_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with (ROOT / "tasks.csv").open("w", newline="") as f:
        fields = list(tasks[0])
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(tasks)
    with (ROOT / "run_matrix.csv").open("w", newline="") as f:
        fields = ["array_index", "model_id", "model_label", "model_family", "seed", "model_path", "task_json", "task_id"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        i = 0
        for model in models:
            for task in tasks:
                i += 1
                w.writerow({"array_index": i, "model_id": model["model_id"], "model_label": model["model_label"],
                    "model_family": model["model_family"], "seed": model["seed"], "model_path": model["model_path"],
                    "task_json": f"task_json/{task['task_id']}.json", "task_id": task["task_id"]})
    with (ROOT / "run_matrix.tsv").open("w") as f:
        for row in csv.DictReader((ROOT / "run_matrix.csv").open()):
            f.write("\t".join(row[k] for k in ["array_index", "model_id", "model_label", "model_family", "seed", "model_path", "task_json", "task_id"]) + "\n")
    print(json.dumps(manifest["expected"], sort_keys=True))

if __name__ == "__main__": main()
