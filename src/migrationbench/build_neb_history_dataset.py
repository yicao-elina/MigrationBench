#!/usr/bin/env python3
"""Build normalized MigrationBench tables from provenance-complete QE histories."""

import argparse
import csv
import hashlib
import json
import math
import shlex
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def stable_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(value, prefix):
    return prefix + hashlib.sha256(stable_json(value).encode()).hexdigest()[:32]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_identity(history_dir):
    path = Path(history_dir) / "runtime_provenance.json"
    if not path.is_file():
        return {"uri": None, "sha256": None, "environment_id": None}
    record = json.loads(path.read_text())
    consequential = {
        "platform": record.get("platform"),
        "python": record.get("python"),
        "packages": record.get("packages"),
        "loaded_modules": record.get("environment", {}).get("LOADEDMODULES"),
        "binaries": record.get("binaries"),
    }
    return {
        "uri": str(path.resolve()),
        "sha256": sha256_file(path),
        "environment_id": stable_hash(consequential, "ENV_"),
    }


def parse_scalar(value):
    if value in {"nan", "NaN", "None", ""}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            parsed = float(value)
            return None if math.isnan(parsed) else parsed
        except ValueError:
            return value


def read_extxyz(path):
    with path.open() as handle:
        while True:
            count_line = handle.readline()
            if not count_line:
                return
            atom_count = int(count_line.strip())
            metadata = {}
            for token in shlex.split(handle.readline().strip()):
                if "=" in token:
                    key, value = token.split("=", 1)
                    metadata[key] = parse_scalar(value)
            symbols, positions, forces, gradients, masks = [], [], [], [], []
            for _ in range(atom_count):
                fields = handle.readline().split()
                if len(fields) != 13:
                    raise ValueError(f"Expected 13 extxyz columns in {path}, found {len(fields)}")
                symbols.append(fields[0])
                values = [float(value) for value in fields[1:10]]
                positions.append(values[0:3])
                forces.append(values[3:6])
                gradients.append(values[6:9])
                masks.append([int(value) for value in fields[10:13]])
            lattice = [float(value) for value in str(metadata["Lattice"]).split()]
            cell = [lattice[0:3], lattice[3:6], lattice[6:9]]
            yield {
                "metadata": metadata,
                "symbols": symbols,
                "positions_A": positions,
                "forces_eV_A": forces,
                "path_energy_gradient_eV_A": gradients,
                "move_mask": masks,
                "cell_A": cell,
            }


def composition_formula(symbols):
    counts = Counter(symbols)
    return "".join(f"{symbol}{counts[symbol]}" for symbol in sorted(counts))


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
            writer.writerow(
                {
                    key: stable_json(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )
    return {"rows": len(rows), "jsonl": str(jsonl), "csv": str(csv_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--system-id", default="Cr-doped-Sb2Te3")
    parser.add_argument("--method", default="PBE-D3")
    parser.add_argument("--software", default="Quantum ESPRESSO")
    args = parser.parse_args()

    configurations = []
    calculations = []
    optimization_iterations = []
    pathway_frames = {}
    source_manifests = []
    created_at = datetime.now(timezone.utc).isoformat()
    for history_dir in args.history_dir:
        history_dir = history_dir.resolve()
        manifest = json.loads((history_dir / "neb_full_history_manifest.json").read_text())
        runtime = runtime_identity(history_dir)
        manifest["runtime_provenance"] = runtime
        source_manifests.append(manifest)
        pathway_id = manifest["pathway_id"]
        frames = list(read_extxyz(history_dir / "neb_full_history.extxyz"))
        pathway_frames[pathway_id] = frames
        for frame in frames:
            info = frame["metadata"]
            configuration_id = info["configuration_id"]
            iteration_record_id = info["iteration_record_id"]
            config = {
                "configuration_id": configuration_id,
                "structure_hash": stable_hash(
                    {
                        "symbols": frame["symbols"],
                        "cell_A": frame["cell_A"],
                        "positions_A": frame["positions_A"],
                        "pbc": [True, True, True],
                    },
                    "ST_",
                ),
                "system_id": args.system_id,
                "composition": composition_formula(frame["symbols"]),
                "elements": sorted(set(frame["symbols"])),
                "nsites": len(frame["symbols"]),
                "cell_A": frame["cell_A"],
                "pbc": [True, True, True],
                "symbols": frame["symbols"],
                "positions_A": frame["positions_A"],
                "source_uri": manifest["run_dir"],
                "source_sha256": manifest["source_files"]["path_restart_sha256"].get(
                    f"sb2te3.path{info['path_iteration']}"
                ),
                "created_at_utc": created_at,
                "metadata_json": stable_json(
                    {
                        "pathway_id": pathway_id,
                        "path_iteration": info["path_iteration"],
                        "image_index": info["image_index"],
                        "move_mask": frame["move_mask"],
                    }
                ),
            }
            configurations.append(config)
            calculation_id = stable_hash(
                {
                    "configuration_id": configuration_id,
                    "run_id": manifest["run_id"],
                    "method": args.method,
                },
                "CA_",
            )
            calc = {
                "calculation_id": calculation_id,
                "configuration_id": configuration_id,
                "pathway_id": pathway_id,
                "image_index": info["image_index"],
                "reaction_coordinate": info["reaction_coordinate_normalized"],
                "calculator_family": "DFT",
                "calculator_label": args.software,
                "model_checkpoint_uri": None,
                "method": args.method,
                "energy_eV": info["energy"],
                "forces_eV_A": frame["forces_eV_A"],
                "stress": None,
                "force_label_status": "real",
                "convergence_status": "scf_converged_path_unconverged",
                "run_id": manifest["run_id"],
                "raw_log_uri": manifest["run_dir"],
                "raw_log_sha256": None,
                "runtime_provenance_uri": runtime["uri"],
                "runtime_provenance_sha256": runtime["sha256"],
                "runtime_environment_id": runtime["environment_id"],
                "software": args.software,
                "split_group_id": pathway_id,
                "split": "audit",
                "dataset_role": "unconverged_audit",
                "metadata_json": stable_json({"iteration_record_id": iteration_record_id}),
            }
            calculations.append(calc)
            optimization_iterations.append(
                {
                    "iteration_record_id": iteration_record_id,
                    "run_id": manifest["run_id"],
                    "pathway_id": pathway_id,
                    "endpoint_basin_id": None,
                    "optimizer_iteration": info["path_iteration"],
                    "image_index": info["image_index"],
                    "configuration_id": configuration_id,
                    "calculation_id": calculation_id,
                    "energy_eV": info["energy"],
                    "forces_eV_A": frame["forces_eV_A"],
                    "path_energy_gradient_eV_A": frame["path_energy_gradient_eV_A"],
                    "neb_force_eV_A": None,
                    "neb_residual_eV_A": info["neb_residual_eV_A"],
                    "reaction_coordinate": info["reaction_coordinate_normalized"],
                    "reaction_coordinate_A": info["reaction_coordinate_A"],
                    "scf_iterations": info["scf_iterations"],
                    "convergence_state": stable_json(
                        {
                            "scf": "converged",
                            "neb_iteration_complete": info["neb_residual_eV_A"] is not None,
                            "path": "unconverged",
                        }
                    ),
                    "raw_log_uri": manifest["run_dir"],
                    "raw_log_sha256": None,
                    "runtime_provenance_uri": runtime["uri"],
                    "runtime_provenance_sha256": runtime["sha256"],
                    "split_group_id": pathway_id,
                    "split": "audit",
                    "metadata_json": stable_json(
                        {
                            "frozen_endpoint_force_reused": info["frozen_endpoint_force_reused"],
                            "gradient_semantics": info.get("gradient_semantics"),
                        }
                    ),
                }
            )

    neb_paths = []
    barriers = []
    for pathway_id, frames in pathway_frames.items():
        complete_iterations = sorted(
            {
                int(frame["metadata"]["path_iteration"])
                for frame in frames
                if frame["metadata"]["neb_residual_eV_A"] is not None
            }
        )
        latest = complete_iterations[-1]
        final_frames = sorted(
            [frame for frame in frames if int(frame["metadata"]["path_iteration"]) == latest],
            key=lambda frame: int(frame["metadata"]["image_index"]),
        )
        energies = [float(frame["metadata"]["energy"]) for frame in final_frames]
        max_residual = max(float(frame["metadata"]["neb_residual_eV_A"]) for frame in final_frames)
        forward = max(energies) - energies[0]
        reverse = max(energies) - energies[-1]
        calculation_ids = [
            next(
                row["calculation_id"]
                for row in calculations
                if row["configuration_id"] == frame["metadata"]["configuration_id"]
            )
            for frame in final_frames
        ]
        neb_paths.append(
            {
                "pathway_id": pathway_id,
                "system_id": args.system_id,
                "path_label": pathway_id,
                "family": "unresolved_hidden_basin",
                "n_images": len(final_frames),
                "endpoint_configuration_ids": [
                    final_frames[0]["metadata"]["configuration_id"],
                    final_frames[-1]["metadata"]["configuration_id"],
                ],
                "reference_calculation_ids": calculation_ids,
                "barrier_eV": forward,
                "reverse_barrier_eV": reverse,
                "barrier_convention": "peak relative to endpoint",
                "status": "unconverged",
                "acceptance_notes": f"latest complete iteration {latest}; max NEB residual {max_residual:.6f} eV/A; endpoints unverified",
                "split_group_id": pathway_id,
                "metadata_json": stable_json({"latest_complete_iteration": latest, "max_neb_residual_eV_A": max_residual}),
            }
        )
        barriers.append(
            {
                "barrier_result_id": stable_hash(
                    {"pathway_id": pathway_id, "iteration": latest, "calculation_ids": calculation_ids}, "BR_"
                ),
                "pathway_id": pathway_id,
                "protocol": "dft_neb",
                "calculator_label": args.software,
                "barrier_eV": forward,
                "reverse_barrier_eV": reverse,
                "reference_barrier_eV": None,
                "error_eV": None,
                "source_calculation_ids": calculation_ids,
                "convergence_status": "unconverged",
                "manuscript_allowed": False,
                "exclusion_reason": "path_force_not_converged_and_endpoints_unverified",
                "metadata_json": stable_json({"optimizer_iteration": latest}),
            }
        )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "configurations": configurations,
        "calculations": calculations,
        "optimization_iterations": optimization_iterations,
        "neb_paths": neb_paths,
        "derived_barriers": barriers,
    }
    outputs = {name: write_table(name, rows, out_dir) for name, rows in tables.items()}
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": created_at,
        "dataset_repo_id": "alinacao2000/MigrationBench",
        "publication_status": "local_staging_unconverged_not_uploaded",
        "leakage_policy": "all images and iterations from one pathway share split_group_id; current records are audit-only",
        "tables": outputs,
        "source_manifests": [
            {"run_id": source["run_id"], "pathway_id": source["pathway_id"], "run_dir": source["run_dir"], "runtime_provenance": source["runtime_provenance"]}
            for source in source_manifests
        ],
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"tables": {name: len(rows) for name, rows in tables.items()}, "output": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
