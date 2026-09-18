#!/usr/bin/env python3
"""Compare direct and repaired MLFF NEB runs without confusing basin change for speedup."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from ase.io import read


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "migrationbench"))

from analyze_neb_path_topology import choose_migrant, sha256_file  # noqa: E402
from compare_mlff_preconditioner_runs import first_step_below, trapezoid_auc  # noqa: E402
from select_representative_trajectories import (  # noqa: E402
    COMPONENTS,
    arc_coordinates,
    displacement_signature,
    finalize_distances,
    local_signature,
    normalized_energy_curve,
    pair_signature,
    raw_pair_distances,
    resample_curve,
    topology_signature,
)


def atoms_dict(atoms) -> dict:
    return {
        "symbols": atoms.get_chemical_symbols(),
        "positions": [tuple(row) for row in atoms.get_positions()],
        "cell": [tuple(row) for row in atoms.cell.array] if atoms.cell.rank == 3 else None,
    }


def final_path_descriptor(main: dict, main_path: Path, system_id: str, selection_config: dict) -> dict:
    images_path = main_path.parent / "mlff_neb_images.extxyz"
    images = [atoms_dict(atoms) for atoms in read(images_path, index=":")]
    migrant = choose_migrant(images, "Cr")
    coordinates, _ = arc_coordinates(images)
    count = int(selection_config["resample_points"])
    curves = {
        "geometry_frechet": [pair_signature(image) for image in images],
        "local_environment": [
            local_signature(image, migrant, int(selection_config["local_neighbors_per_species"]))
            for image in images
        ],
        "displacement_field": [
            displacement_signature(image, images[0], migrant) for image in images
        ],
        "energy_profile": normalized_energy_curve(main["profile"]),
        "bond_topology": [
            topology_signature(image, migrant, list(selection_config["coordination_radii_A"]))
            for image in images
        ],
    }
    return {
        "path_id": main_path.parent.name,
        "system_id": system_id,
        "curves": {name: resample_curve(values, coordinates, count) for name, values in curves.items()},
        "images_path": str(images_path),
        "images_sha256": sha256_file(images_path),
    }


def load_run(main_path: Path) -> dict:
    main_path = main_path.resolve()
    main = json.loads(main_path.read_text())
    history_path = main_path.parent / "mlff_neb_iteration_history_manifest.json"
    history = json.loads(history_path.read_text())
    iterations = history["iteration_summaries"]
    if not iterations or iterations[0]["optimizer_iteration"] != 0:
        raise ValueError(f"Incomplete optimizer history: {history_path}")
    return {
        "main_path": main_path,
        "main": main,
        "history_path": history_path,
        "history": history,
        "iterations": iterations,
    }


def protocol_identity(run: dict) -> dict:
    main, history = run["main"], run["history"]
    return {
        "calculator": main["calculator"],
        "model_path_sha256": main["model_path_sha256"],
        "seed": main["seed"],
        "n_images": main["n_images"],
        "fmax_target_eV_A": main["fmax_target_eV_A"],
        "steps_requested": main["steps_requested"],
        "spring_constant_eV_A2": history["spring_constant_eV_A2"],
        "climb": history["climb"],
        "neb_method": history["neb_method"],
    }


def run_metrics(run: dict, threshold: float) -> dict:
    main, points = run["main"], run["iterations"]
    first, last = points[0], points[-1]
    return {
        "input_images_sha256": main["input_images_sha256"],
        "optimizer_converged": bool(main["optimizer_converged"]),
        "optimizer_steps_completed": int(main["optimizer_steps_completed"]),
        "steps_to_target": first_step_below(points, "max_neb_residual_eV_A", threshold),
        "initial_neb_residual_eV_A": first["max_neb_residual_eV_A"],
        "final_neb_residual_eV_A": last["max_neb_residual_eV_A"],
        "residual_auc_eV_A_step": trapezoid_auc(points, "max_neb_residual_eV_A"),
        "initial_barrier_proxy_eV": first["barrier_proxy_eV"],
        "final_barrier_proxy_eV": last["barrier_proxy_eV"],
        "final_max_internal_true_force_eV_A": last["max_internal_true_force_eV_A"],
        "final_max_internal_neb_force_eV_A": last["max_internal_neb_force_eV_A"],
        "main_manifest": str(run["main_path"]),
        "main_manifest_sha256": sha256_file(run["main_path"]),
        "history_manifest": str(run["history_path"]),
        "history_manifest_sha256": sha256_file(run["history_path"]),
    }


def compare(args: argparse.Namespace) -> dict:
    config = json.loads(args.config.read_text())
    comparison_policy = json.loads(args.comparison_policy.read_text())
    selection_config = json.loads(args.selection_config.read_text())
    portfolio = json.loads(args.portfolio_manifest.read_text())
    direct, repaired = load_run(args.direct_manifest), load_run(args.repaired_manifest)
    identity_direct, identity_repaired = protocol_identity(direct), protocol_identity(repaired)
    identity_match = identity_direct == identity_repaired
    expected_identity = {
        "calculator": config["calculator"],
        "model_path_sha256": config["model_path_sha256"],
        "seed": config["seed"],
        "fmax_target_eV_A": config["fmax_target_eV_A"],
        "steps_requested": config["steps"],
        "spring_constant_eV_A2": config["spring_constant_eV_A2"],
    }
    identity_expected = all(identity_direct[key] == value for key, value in expected_identity.items())

    repair_manifest = json.loads(args.repair_manifest.read_text())
    repair_binding = (
        repaired["main"]["source_candidate_manifest_sha256"] == sha256_file(args.repair_manifest)
        and repaired["main"]["input_images_sha256"] == repair_manifest["output_sha256"]
        and repair_manifest["all_intermediate_images_pass"]
        and repair_manifest["endpoints_pass"]
    )

    left = final_path_descriptor(direct["main"], direct["main_path"], args.system_id, selection_config)
    right = final_path_descriptor(repaired["main"], repaired["main_path"], args.system_id, selection_config)
    pair = {"raw": raw_pair_distances(left, right)}
    scales = portfolio["groups"][args.system_id]["distance_scales"]
    finalize_distances([pair], selection_config["distance_weights"], scales)
    similarity = pair["similarity"]
    same_mechanism_threshold = float(
        comparison_policy["minimum_final_path_similarity_for_same_mechanism"]
    )
    same_mechanism = similarity >= same_mechanism_threshold

    threshold = float(config["fmax_target_eV_A"])
    direct_metrics = run_metrics(direct, threshold)
    repaired_metrics = run_metrics(repaired, threshold)
    both_converged = direct_metrics["optimizer_converged"] and repaired_metrics["optimizer_converged"]
    both_internal_neb_force_pass = (
        direct_metrics["final_max_internal_neb_force_eV_A"] <= threshold
        and repaired_metrics["final_max_internal_neb_force_eV_A"] <= threshold
    )
    speedup_eligible = (
        identity_match
        and identity_expected
        and repair_binding
        and both_converged
        and both_internal_neb_force_pass
        and same_mechanism
    )
    if speedup_eligible and repaired_metrics["optimizer_steps_completed"] > 0:
        step_speedup = direct_metrics["optimizer_steps_completed"] / repaired_metrics["optimizer_steps_completed"]
        decision = "same_mechanism_speedup_measurable"
    else:
        step_speedup = None
        if not identity_match or not identity_expected or not repair_binding:
            decision = "invalid_provenance_or_protocol"
        elif not both_converged:
            decision = "pending_or_not_both_converged"
        else:
            decision = "mechanism_changed_no_speedup_claim"

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "path_id": args.path_id,
        "system_id": args.system_id,
        "scientific_role": config["scientific_role"],
        "identity_match": identity_match,
        "identity_matches_config": identity_expected,
        "repair_provenance_binding_pass": repair_binding,
        "protocol_identity": identity_direct,
        "direct": direct_metrics,
        "repaired": repaired_metrics,
        "final_path_distance": pair["distance"],
        "final_path_similarity": similarity,
        "final_path_alignment": pair["alignment"],
        "distance_components": pair["components"],
        "same_mechanism_threshold": same_mechanism_threshold,
        "same_mechanism": same_mechanism,
        "both_internal_neb_force_pass": both_internal_neb_force_pass,
        "speedup_eligible": speedup_eligible,
        "ionic_step_speedup_direct_over_repaired": step_speedup,
        "decision": decision,
        "qe_handoff_allowed": False,
        "qe_handoff_reason": "unverified endpoints; diagnostic repair A/B only",
        "provenance": {
            "config_sha256": sha256_file(args.config),
            "comparison_policy_sha256": sha256_file(args.comparison_policy),
            "selection_config_sha256": sha256_file(args.selection_config),
            "portfolio_manifest_sha256": sha256_file(args.portfolio_manifest),
            "repair_manifest_sha256": sha256_file(args.repair_manifest),
            "direct_final_images_sha256": left["images_sha256"],
            "repaired_final_images_sha256": right["images_sha256"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-id", required=True)
    parser.add_argument("--system-id", default="Sb2Te3Cr_61")
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--repaired-manifest", type=Path, required=True)
    parser.add_argument("--repair-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "mlff_repair_ab_coverage_gaps_s46.json")
    parser.add_argument("--comparison-policy", type=Path, default=ROOT / "configs" / "mlff_repair_ab_comparison_v1.json")
    parser.add_argument("--selection-config", type=Path, default=ROOT / "configs" / "historical_trajectory_portfolio.json")
    parser.add_argument("--portfolio-manifest", type=Path, default=ROOT / "data_processed" / "representative_trajectory_portfolio" / "portfolio_manifest.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / f"{args.path_id}_mlff_repair_ab.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    rows = []
    for variant in ("direct", "repaired"):
        rows.append({"variant": variant, **result[variant]})
    fields = list(rows[0])
    with (args.out_dir / f"{args.path_id}_mlff_repair_ab.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(output), "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
