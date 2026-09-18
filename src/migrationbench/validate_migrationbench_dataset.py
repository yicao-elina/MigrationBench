#!/usr/bin/env python3
"""Fail closed on MigrationBench relational, physical, and leakage invariants."""

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PRIMARY_KEYS = {
    "configurations": "configuration_id",
    "calculations": "calculation_id",
    "optimization_iterations": "iteration_record_id",
    "neb_paths": "pathway_id",
    "derived_barriers": "barrier_result_id",
}


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def collect_sha256_fields(value):
    hashes = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.endswith("sha256") and isinstance(nested, str):
                hashes.append(nested)
            else:
                hashes.extend(collect_sha256_fields(nested))
    elif isinstance(value, list):
        for nested in value:
            hashes.extend(collect_sha256_fields(nested))
    return hashes


def trust_region_semantics_valid(row, calc, config, tolerance=2.0e-6):
    if calc.get("method") != "trust_region_mlff_neb_base_rescore":
        return True
    optimization_forces = row.get("optimization_atomic_forces_eV_A")
    optimization_energy = row.get("optimization_energy_eV")
    regularization_energy = row.get("regularization_energy_eV")
    metadata = row.get("metadata_json", {})
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return (
        isinstance(optimization_forces, list)
        and len(optimization_forces) == config["nsites"]
        and all(isinstance(vector, list) and len(vector) == 3 for vector in optimization_forces)
        and isinstance(optimization_energy, (int, float))
        and isinstance(regularization_energy, (int, float))
        and math.isfinite(float(optimization_energy))
        and math.isfinite(float(regularization_energy))
        and float(regularization_energy) >= -1.0e-10
        and abs(
            float(optimization_energy)
            - float(row["energy_eV"])
            - float(regularization_energy)
        ) <= tolerance
        and metadata.get("physical_energy_force_source")
        == "base_MACE_without_reference_tether"
        and metadata.get("neb_force_potential")
        == "base_plus_reference_tether"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gradient-force-tolerance-eV-A", type=float, default=2.0e-5)
    parser.add_argument("--quarantine-manifest", type=Path)
    args = parser.parse_args()

    root = args.dataset_dir.resolve()
    tables = {name: load_jsonl(root / f"{name}.jsonl") for name in PRIMARY_KEYS}
    failures = []
    metrics = {}
    ids = {}
    for name, key in PRIMARY_KEYS.items():
        values = [row[key] for row in tables[name]]
        ids[name] = set(values)
        duplicate_count = len(values) - len(ids[name])
        metrics[f"{name}_rows"] = len(values)
        metrics[f"{name}_duplicate_primary_keys"] = duplicate_count
        if duplicate_count:
            failures.append(f"{name}: duplicate primary keys={duplicate_count}")

    missing_calc_config = [row["calculation_id"] for row in tables["calculations"] if row["configuration_id"] not in ids["configurations"]]
    missing_iter_config = [row["iteration_record_id"] for row in tables["optimization_iterations"] if row["configuration_id"] not in ids["configurations"]]
    missing_iter_calc = [row["iteration_record_id"] for row in tables["optimization_iterations"] if row["calculation_id"] not in ids["calculations"]]
    missing_barrier_path = [row["barrier_result_id"] for row in tables["derived_barriers"] if row["pathway_id"] not in ids["neb_paths"]]
    metrics.update(
        {
            "missing_calculation_configuration_fk": len(missing_calc_config),
            "missing_iteration_configuration_fk": len(missing_iter_config),
            "missing_iteration_calculation_fk": len(missing_iter_calc),
            "missing_barrier_path_fk": len(missing_barrier_path),
        }
    )
    if any([missing_calc_config, missing_iter_config, missing_iter_calc, missing_barrier_path]):
        failures.append("foreign-key integrity failure")

    configs = {row["configuration_id"]: row for row in tables["configurations"]}
    calcs = {row["calculation_id"]: row for row in tables["calculations"]}
    max_gradient_force_residual = 0.0
    max_iteration_calculation_force_residual = 0.0
    shape_failures = 0
    endpoint_neb_force_violations = 0
    internal_mlff_neb_force_missing = 0
    trust_region_semantic_failures = 0
    for row in tables["optimization_iterations"]:
        config = configs[row["configuration_id"]]
        calc = calcs[row["calculation_id"]]
        forces = calc["forces_eV_A"]
        gradients = row["path_energy_gradient_eV_A"]
        iteration_forces = row["forces_eV_A"]
        if (
            len(forces) != config["nsites"]
            or len(iteration_forces) != config["nsites"]
            or len(gradients) != config["nsites"]
        ):
            shape_failures += 1
            continue
        for force, iteration_force, gradient in zip(forces, iteration_forces, gradients):
            if len(force) != 3 or len(gradient) != 3:
                shape_failures += 1
                continue
            max_gradient_force_residual = max(
                max_gradient_force_residual,
                max(abs(force_component + gradient_component) for force_component, gradient_component in zip(force, gradient)),
            )
            max_iteration_calculation_force_residual = max(
                max_iteration_calculation_force_residual,
                max(abs(a - b) for a, b in zip(force, iteration_force)),
            )
        state = row.get("convergence_state", {})
        if isinstance(state, str):
            state = json.loads(state)
        is_endpoint = state.get("is_endpoint")
        if is_endpoint and row.get("neb_force_eV_A") is not None:
            endpoint_neb_force_violations += 1
        if (
            is_endpoint is False
            and calcs[row["calculation_id"]].get("method") in {
                "self_consistent_mlff_neb",
                "trust_region_mlff_neb_base_rescore",
            }
            and row.get("neb_force_eV_A") is None
        ):
            internal_mlff_neb_force_missing += 1
        if calc.get("method") == "trust_region_mlff_neb_base_rescore":
            if not trust_region_semantics_valid(row, calc, config):
                trust_region_semantic_failures += 1
    metrics["array_shape_failures"] = shape_failures
    metrics["max_gradient_plus_force_residual_eV_A"] = max_gradient_force_residual
    metrics["max_iteration_calculation_force_residual_eV_A"] = max_iteration_calculation_force_residual
    metrics["endpoint_neb_force_violations"] = endpoint_neb_force_violations
    metrics["internal_mlff_neb_force_missing"] = internal_mlff_neb_force_missing
    metrics["trust_region_semantic_failures"] = trust_region_semantic_failures
    if shape_failures:
        failures.append(f"atomic array shape failures={shape_failures}")
    if max_gradient_force_residual > args.gradient_force_tolerance_eV_A:
        failures.append(f"gradient/force residual={max_gradient_force_residual} eV/A")
    if max_iteration_calculation_force_residual > args.gradient_force_tolerance_eV_A:
        failures.append(f"iteration/calculation force residual={max_iteration_calculation_force_residual} eV/A")
    if endpoint_neb_force_violations:
        failures.append("fixed endpoints contain inapplicable projected NEB forces")
    if internal_mlff_neb_force_missing:
        failures.append("internal MLFF NEB images are missing projected forces")
    if trust_region_semantic_failures:
        failures.append("trust-region rows mix or omit base and optimization-potential semantics")

    group_splits = defaultdict(set)
    structure_splits = defaultdict(set)
    for row in tables["calculations"]:
        group_splits[row["split_group_id"]].add(row["split"])
        structure_splits[configs[row["configuration_id"]]["structure_hash"]].add(row["split"])
    split_group_violations = {key: sorted(value) for key, value in group_splits.items() if len(value) > 1}
    cross_split_structure_violations = {key: sorted(value) for key, value in structure_splits.items() if len(value) > 1}
    metrics["split_groups"] = len(group_splits)
    metrics["split_group_violations"] = len(split_group_violations)
    metrics["cross_split_structure_violations"] = len(cross_split_structure_violations)
    metrics["splits"] = sorted({value for values in group_splits.values() for value in values})
    if split_group_violations:
        failures.append("one pathway split_group_id appears in multiple splits")
    if cross_split_structure_violations:
        failures.append("identical structure_hash appears in multiple splits")

    manuscript_gate_failures = []
    paths = {row["pathway_id"]: row for row in tables["neb_paths"]}
    for barrier in tables["derived_barriers"]:
        if barrier["manuscript_allowed"]:
            metadata = barrier.get("metadata_json", {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            hashes = metadata.get("acceptance_artifact_hashes", {})
            valid_hashes = all(
                isinstance(hashes.get(key), str) and len(hashes[key]) == 64
                for key in (
                    "path_acceptance_sha256",
                    "endpoint_acceptance_sha256",
                    "calculator_acceptance_sha256",
                    "runtime_provenance_sha256",
                )
            )
            if (
                paths[barrier["pathway_id"]]["status"] != "converged"
                or barrier.get("convergence_status") != "converged"
                or barrier.get("protocol") != "dft_neb"
                or metadata.get("manuscript_gate_reasons")
                or not valid_hashes
            ):
                manuscript_gate_failures.append(barrier["barrier_result_id"])
    metrics["manuscript_gate_failures"] = len(manuscript_gate_failures)
    if manuscript_gate_failures:
        failures.append("barrier marked manuscript_allowed without all convergence and acceptance evidence")

    quarantined_hashes = set()
    quarantined_hash_hits = []
    if args.quarantine_manifest:
        quarantine = json.loads(args.quarantine_manifest.read_text())
        quarantined_hashes = {row["sha256"] for row in quarantine.get("affected_files", [])}
        quarantined_hashes.add(quarantine.get("source_script", {}).get("sha256", ""))
        quarantined_hashes.discard("")
        for table_name, rows in tables.items():
            for row_index, row in enumerate(rows):
                for digest in collect_sha256_fields(row):
                    if digest in quarantined_hashes:
                        quarantined_hash_hits.append(
                            {"table": table_name, "row_index": row_index, "sha256": digest}
                        )
    metrics["quarantined_source_hashes"] = len(quarantined_hashes)
    metrics["quarantined_source_hash_hits"] = len(quarantined_hash_hits)
    if quarantined_hash_hits:
        failures.append("quarantined source hash entered dataset")

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(root),
        "status": "pass" if not failures else "fail",
        "metrics": metrics,
        "failures": failures,
    }
    output = args.output or root / "validation_report.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
