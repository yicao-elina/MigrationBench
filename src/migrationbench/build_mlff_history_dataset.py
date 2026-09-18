#!/usr/bin/env python3
"""Build normalized audit tables from corrected lossless MLFF NEB histories."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import read


def stable_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_id(prefix, value):
    return prefix + hashlib.sha256(stable_json(value).encode()).hexdigest()[:32]


def sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_identity(history_path, history):
    record = history.get("runtime_provenance")
    if not record:
        return {"uri": None, "sha256": None, "environment_id": None}
    path = history_path.parent / Path(record["uri"]).name
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise ValueError(f"Runtime provenance missing or hash mismatch: {path}")
    payload = json.loads(path.read_text())
    consequential = {
        "platform": payload.get("platform"),
        "python": payload.get("python"),
        "packages": payload.get("packages"),
        "loaded_modules": payload.get("environment", {}).get("LOADEDMODULES"),
        "binaries": payload.get("binaries"),
    }
    return {
        "uri": str(path.resolve()),
        "sha256": record["sha256"],
        "environment_id": stable_id("ENV_", consequential),
    }


def composition(symbols):
    counts = Counter(symbols)
    return "".join(f"{symbol}{counts[symbol]}" for symbol in sorted(counts))


def reaction_coordinates(images):
    cumulative = [0.0]
    for left, right in zip(images, images[1:]):
        delta = right.get_positions() - left.get_positions()
        mic, _ = find_mic(delta, left.get_cell(), left.get_pbc())
        cumulative.append(cumulative[-1] + float(np.sqrt(np.mean(np.sum(mic * mic, axis=1)))))
    total = cumulative[-1]
    normalized = [value / total if total > 0 else 0.0 for value in cumulative]
    return cumulative, normalized


def write_table(name, rows, out_dir):
    jsonl = out_dir / f"{name}.jsonl"
    with jsonl.open("w") as handle:
        for row in rows:
            handle.write(stable_json(row) + "\n")
    csv_path = out_dir / f"{name}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: stable_json(value) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })
    return {"rows": len(rows), "jsonl": str(jsonl), "csv": str(csv_path)}


def load_dual_history(main_path, main, expected_frames):
    tether = main.get("reference_tether", {})
    if tether.get("enabled") is not True:
        return None, None
    outputs = main.get("outputs", {})
    extxyz_path = main_path.parent / Path(outputs["dual_potential_history_extxyz"]).name
    csv_path = main_path.parent / Path(outputs["dual_potential_history_csv"]).name
    if sha256(extxyz_path) != outputs.get("dual_potential_history_extxyz_sha256"):
        raise ValueError(f"Dual-potential extxyz hash mismatch: {extxyz_path}")
    if sha256(csv_path) != outputs.get("dual_potential_history_csv_sha256"):
        raise ValueError(f"Dual-potential CSV hash mismatch: {csv_path}")
    frames = read(extxyz_path, index=":")
    if len(frames) != expected_frames:
        raise ValueError("Dual-potential frame count does not match optimization history")
    indexed = {}
    for atoms in frames:
        iteration = int(atoms.info["optimizer_iteration"])
        image_index = int(atoms.info.get("image_index", atoms.info.get("neb_image")))
        key = (iteration, image_index)
        if key in indexed:
            raise ValueError(f"Duplicate dual-potential frame: {key}")
        if "base_forces" not in atoms.arrays or "restrained_forces" not in atoms.arrays:
            raise ValueError(f"Dual-potential frame lacks force arrays: {key}")
        indexed[key] = atoms
    manifest_path = main_path.parent / Path(
        outputs.get(
            "dual_potential_history_manifest",
            "mlff_neb_dual_potential_history_manifest.json",
        )
    ).name
    manifest = None
    if manifest_path.is_file():
        expected_hash = outputs.get("dual_potential_history_manifest_sha256")
        if expected_hash and sha256(manifest_path) != expected_hash:
            raise ValueError(f"Dual-potential manifest hash mismatch: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("complete") is not True:
            raise ValueError("Dual-potential manifest is incomplete")
        if manifest.get("reported_barrier_source") != "base_MACE_without_reference_tether":
            raise ValueError("Dual-potential manifest has unsafe barrier semantics")
    return indexed, {
        "extxyz": str(extxyz_path.resolve()),
        "extxyz_sha256": sha256(extxyz_path),
        "csv": str(csv_path.resolve()),
        "csv_sha256": sha256(csv_path),
        "manifest": str(manifest_path.resolve()) if manifest_path.is_file() else None,
        "manifest_sha256": sha256(manifest_path) if manifest_path.is_file() else None,
        "semantics": (
            manifest.get("field_semantics") if manifest else {
                "base_energy_eV": "base MACE physical energy",
                "base_forces": "base MACE physical atomic forces",
                "restrained_energy_eV": "optimization energy including reference tether",
                "restrained_forces": "optimization atomic forces including reference tether",
            }
        ),
    }


def build_run(main_path, history_path, system_id, base_pathway_id, created_at, model_hash_override=None):
    main = json.loads(main_path.read_text())
    history = json.loads(history_path.read_text())
    runtime = runtime_identity(history_path, history)
    extxyz_path = history_path.parent / Path(history["outputs"]["extxyz"]).name
    if sha256(extxyz_path) != history["outputs"]["extxyz_sha256"]:
        raise ValueError(f"History extxyz hash mismatch: {extxyz_path}")
    frames = read(extxyz_path, index=":")
    n_images = int(history["n_images"])
    if len(frames) != n_images * int(history["n_optimizer_iterations_including_initial"]):
        raise ValueError("History frame count does not match manifest")
    dual_frames, dual_provenance = load_dual_history(main_path, main, len(frames))
    tethered = dual_frames is not None
    model_hash = main.get("model_path_sha256") or model_hash_override
    if not model_hash:
        raise ValueError("MLFF main manifest requires model_path_sha256")
    run_id = main_path.parent.name
    pathway_id = f"{base_pathway_id}_{model_hash[:8]}_s{main['seed']}"
    split_group_id = "GR_" + hashlib.sha256(
        f"{system_id}:{main.get('source_candidate_manifest_sha256')}".encode()
    ).hexdigest()[:24]
    configs, calculations, iterations = [], [], []
    by_iteration = {}
    for atoms in frames:
        iteration = int(atoms.info["optimizer_iteration"])
        by_iteration.setdefault(iteration, []).append(atoms)
    for iteration, images in sorted(by_iteration.items()):
        images.sort(key=lambda atoms: int(atoms.info["image_index"]))
        cumulative, normalized = reaction_coordinates(images)
        for atoms, coordinate_A, coordinate in zip(images, cumulative, normalized):
            image_index = int(atoms.info["image_index"])
            dual = dual_frames[(iteration, image_index)] if tethered else None
            if dual is not None:
                if dual.get_chemical_symbols() != atoms.get_chemical_symbols():
                    raise ValueError("Dual/optimization atom order mismatch")
                if not np.allclose(dual.get_positions(), atoms.get_positions(), atol=1.0e-8):
                    raise ValueError("Dual/optimization coordinates are not aligned")
                if not np.allclose(dual.get_cell().array, atoms.get_cell().array, atol=1.0e-8):
                    raise ValueError("Dual/optimization cells are not aligned")
            symbols = atoms.get_chemical_symbols()
            geometry = {
                "symbols": symbols,
                "cell_A": atoms.get_cell().array.round(10).tolist(),
                "pbc": [bool(value) for value in atoms.get_pbc()],
                "positions_A": atoms.get_positions().round(10).tolist(),
            }
            structure_hash = stable_id("ST_", geometry)
            configuration_id = stable_id("CO_", {
                "run_id": run_id,
                "optimizer_iteration": iteration,
                "image_index": image_index,
                "structure_hash": structure_hash,
            })
            calculation_id = stable_id("CA_", {
                "configuration_id": configuration_id,
                "model_path_sha256": model_hash,
                "run_id": run_id,
            })
            iteration_record_id = stable_id("IT_", {
                "run_id": run_id,
                "optimizer_iteration": iteration,
                "image_index": image_index,
            })
            try:
                true_forces = np.asarray(
                    dual.arrays["base_forces"] if dual is not None else atoms.get_forces(),
                    dtype=float,
                )
                optimization_forces = np.asarray(
                    dual.arrays["restrained_forces"] if dual is not None else true_forces,
                    dtype=float,
                )
            except Exception as error:
                raise ValueError(
                    f"Missing calculator forces at iteration {iteration}, image {image_index}"
                ) from error
            if true_forces.shape != (len(atoms), 3) or not np.isfinite(true_forces).all():
                raise ValueError(f"Invalid true forces at iteration {iteration}, image {image_index}")
            if optimization_forces.shape != (len(atoms), 3) or not np.isfinite(optimization_forces).all():
                raise ValueError(f"Invalid optimization forces at iteration {iteration}, image {image_index}")
            is_endpoint = image_index in {0, n_images - 1}
            neb_forces = atoms.arrays.get("neb_forces")
            if is_endpoint and neb_forces is not None:
                raise ValueError("Fixed endpoint must not contain projected NEB forces")
            if not is_endpoint and neb_forces is None:
                raise ValueError("Internal image is missing projected NEB forces")
            energy = float(dual.info["base_energy_eV"] if dual is not None else atoms.info["energy_eV"])
            optimization_energy = float(
                dual.info["restrained_energy_eV"] if dual is not None else energy
            )
            regularization_energy = float(
                dual.info["tether_energy_eV"] if dual is not None else 0.0
            )
            source_path = dual_provenance["extxyz"] if tethered else str(extxyz_path)
            source_hash = dual_provenance["extxyz_sha256"] if tethered else history["outputs"]["extxyz_sha256"]
            configs.append({
                "configuration_id": configuration_id,
                "structure_hash": structure_hash,
                "system_id": system_id,
                "composition": composition(symbols),
                "elements": sorted(set(symbols)),
                "nsites": len(atoms),
                "cell_A": geometry["cell_A"],
                "pbc": geometry["pbc"],
                "symbols": symbols,
                "positions_A": geometry["positions_A"],
                "source_uri": source_path,
                "source_sha256": source_hash,
                "created_at_utc": created_at,
                "metadata_json": stable_json({
                    "pathway_id": pathway_id,
                    "optimizer_iteration": iteration,
                    "image_index": image_index,
                }),
            })
            calculations.append({
                "calculation_id": calculation_id,
                "configuration_id": configuration_id,
                "pathway_id": pathway_id,
                "image_index": image_index,
                "reaction_coordinate": coordinate,
                "calculator_family": "MLFF",
                "calculator_label": main["calculator"],
                "model_checkpoint_uri": main.get("model_path"),
                "method": "trust_region_mlff_neb_base_rescore" if tethered else "self_consistent_mlff_neb",
                "energy_eV": energy,
                "forces_eV_A": true_forces.tolist(),
                "stress": None,
                "force_label_status": "real",
                "convergence_status": (
                    "restrained_converged_base_unverified"
                    if tethered and main.get("optimizer_converged_under_optimization_potential")
                    else "converged" if main.get("optimizer_converged") else "unconverged"
                ),
                "run_id": run_id,
                "raw_log_uri": main.get("outputs", {}).get("log"),
                "raw_log_sha256": None,
                "runtime_provenance_uri": runtime["uri"],
                "runtime_provenance_sha256": runtime["sha256"],
                "runtime_environment_id": runtime["environment_id"],
                "software": stable_json(main.get("software_versions", {})),
                "split_group_id": split_group_id,
                "split": "audit",
                "dataset_role": "proposal_dependent_preconditioner_diagnostic",
                "metadata_json": stable_json({
                    "proposal_model_path_sha256": model_hash,
                    "source_candidate_manifest_sha256": main.get("source_candidate_manifest_sha256"),
                    "run_role": main.get("run_role"),
                    "reference_tether": main.get("reference_tether"),
                    "energy_force_semantics": "base_MACE_physical_labels" if tethered else "base_calculator",
                }),
            })
            iterations.append({
                "iteration_record_id": iteration_record_id,
                "run_id": run_id,
                "pathway_id": pathway_id,
                "endpoint_basin_id": None,
                "optimizer_iteration": iteration,
                "image_index": image_index,
                "configuration_id": configuration_id,
                "calculation_id": calculation_id,
                "energy_eV": energy,
                "forces_eV_A": true_forces.tolist(),
                "path_energy_gradient_eV_A": (-true_forces).tolist(),
                "neb_force_eV_A": np.asarray(neb_forces).tolist() if neb_forces is not None else None,
                "optimization_energy_eV": optimization_energy,
                "optimization_atomic_forces_eV_A": optimization_forces.tolist(),
                "regularization_energy_eV": regularization_energy,
                "neb_residual_eV_A": atoms.info.get("neb_residual_eV_A"),
                "reaction_coordinate": coordinate,
                "reaction_coordinate_A": coordinate_A,
                "scf_iterations": None,
                "convergence_state": stable_json({
                    "optimizer_converged": bool(main.get("optimizer_converged")),
                    "optimizer_converged_under_optimization_potential": bool(
                        main.get("optimizer_converged_under_optimization_potential", main.get("optimizer_converged"))
                    ),
                    "is_endpoint": is_endpoint,
                    "endpoint_neb_force_applicable": False if is_endpoint else True,
                }),
                "raw_log_uri": main.get("outputs", {}).get("log"),
                "raw_log_sha256": None,
                "runtime_provenance_uri": runtime["uri"],
                "runtime_provenance_sha256": runtime["sha256"],
                "split_group_id": split_group_id,
                "split": "audit",
                "metadata_json": stable_json({
                    "model_path_sha256": model_hash,
                    "spring_constant_eV_A2": history["spring_constant_eV_A2"],
                    "climb": history["climb"],
                    "neb_method": history["neb_method"],
                    "neb_force_potential": "base_plus_reference_tether" if tethered else "base_calculator",
                    "physical_energy_force_source": "base_MACE_without_reference_tether" if tethered else "base_calculator",
                }),
            })
    final_iteration = max(by_iteration)
    final_calcs = [row for row in calculations if any(
        item["calculation_id"] == row["calculation_id"] and item["optimizer_iteration"] == final_iteration
        for item in iterations
    )]
    final_calcs.sort(key=lambda row: row["image_index"])
    energies = [row["energy_eV"] for row in final_calcs]
    barrier = max(energies) - energies[0]
    path_status = (
        "diagnostic_mlff_trust_region_preconditioner"
        if tethered else
        "diagnostic_mlff_converged_unverified_endpoints"
        if main.get("optimizer_converged") else "diagnostic_mlff_unconverged"
    )
    neb_path = {
        "pathway_id": pathway_id,
        "system_id": system_id,
        "path_label": base_pathway_id,
        "family": "unverified_candidate",
        "n_images": n_images,
        "endpoint_configuration_ids": [final_calcs[0]["configuration_id"], final_calcs[-1]["configuration_id"]],
        "reference_calculation_ids": [],
        "barrier_eV": barrier,
        "reverse_barrier_eV": max(energies) - energies[-1],
        "barrier_convention": (
            "base-MACE proxy peak relative to endpoint; reference-tether energy excluded"
            if tethered else "MLFF proxy peak relative to endpoint"
        ),
        "status": path_status,
        "acceptance_notes": "Proposal-dependent MLFF diagnostic; endpoints unverified; not a DFT reference.",
        "split_group_id": split_group_id,
        "metadata_json": stable_json({
            "model_path_sha256": model_hash,
            "source_candidate_manifest_sha256": main.get("source_candidate_manifest_sha256"),
            "final_max_neb_residual_eV_A": main.get("final_max_neb_residual_eV_A"),
            "reference_tether": main.get("reference_tether"),
            "dual_potential_provenance": dual_provenance,
        }),
    }
    derived = {
        "barrier_result_id": stable_id("BR_", {"pathway_id": pathway_id, "final_iteration": final_iteration}),
        "pathway_id": pathway_id,
        "protocol": "trust_region_mlff_neb_base_rescore" if tethered else "self_consistent_mlff_neb",
        "calculator_label": main["calculator"],
        "barrier_eV": barrier,
        "reverse_barrier_eV": max(energies) - energies[-1],
        "reference_barrier_eV": None,
        "error_eV": None,
        "source_calculation_ids": [row["calculation_id"] for row in final_calcs],
        "convergence_status": "diagnostic_restrained" if tethered else "converged" if main.get("optimizer_converged") else "unconverged",
        "manuscript_allowed": False,
        "exclusion_reason": "mlff_proxy_unverified_endpoints_not_dft_reference",
        "metadata_json": stable_json({
            "optimizer_iteration": final_iteration,
            "model_path_sha256": model_hash,
            "reference_tether": main.get("reference_tether"),
            "barrier_energy_source": "base_MACE_without_reference_tether" if tethered else "base_calculator",
        }),
    }
    return configs, calculations, iterations, neb_path, derived, {
        "main_manifest": str(main_path), "main_manifest_sha256": sha256(main_path),
        "history_manifest": str(history_path), "history_manifest_sha256": sha256(history_path),
        "runtime_provenance": runtime,
        "dual_potential_provenance": dual_provenance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="BASE_PATHWAY_ID=MAIN_MANIFEST,HISTORY_MANIFEST[,MODEL_SHA256]")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--system-id", default="Cr-doped-Sb2Te3")
    args = parser.parse_args()
    created_at = datetime.now(timezone.utc).isoformat()
    tables = {name: [] for name in ("configurations", "calculations", "optimization_iterations", "neb_paths", "derived_barriers")}
    sources = []
    for spec in args.run:
        base, paths = spec.split("=", 1)
        fields = paths.split(",")
        if len(fields) not in {2, 3}:
            raise ValueError("Each --run requires main, history, and optional model SHA-256")
        main_raw, history_raw = fields[:2]
        model_hash_override = fields[2] if len(fields) == 3 else None
        result = build_run(
            Path(main_raw).resolve(), Path(history_raw).resolve(), args.system_id,
            base, created_at, model_hash_override,
        )
        for name, rows in zip(tables, result[:5]):
            tables[name].extend(rows if isinstance(rows, list) else [rows])
        sources.append(result[5])
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {name: write_table(name, rows, out_dir) for name, rows in tables.items()}
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": created_at,
        "dataset_repo_id": "alinacao2000/MigrationBench",
        "publication_status": "local_staging_mlff_audit_not_uploaded",
        "leakage_policy": "models share a candidate-derived split_group_id; all rows are audit-only and proposal-dependent",
        "tables": outputs,
        "sources": sources,
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"tables": {name: len(rows) for name, rows in tables.items()}, "output": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
