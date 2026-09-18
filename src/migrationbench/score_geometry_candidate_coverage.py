#!/usr/bin/env python3
"""Score an unlabeled geometry-only path against the historical trajectory portfolio."""

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from ase.io import read

from analyze_neb_path_topology import choose_migrant, sha256_file
from compare_mlff_repair_ab import atoms_dict
from select_representative_trajectories import (
    arc_coordinates,
    displacement_signature,
    facility_coverage,
    local_signature,
    pair_signature,
    raw_pair_distances,
    resample_curve,
    topology_signature,
)


COMPONENTS = ("geometry_frechet", "local_environment", "displacement_field", "bond_topology")


def candidate_descriptor(path, path_id, system_id, config):
    images = [atoms_dict(atoms) for atoms in read(path, index=":")]
    migrant = choose_migrant(images, "Cr")
    coordinates, _ = arc_coordinates(images)
    count = int(config["resample_points"])
    curves = {
        "geometry_frechet": [pair_signature(image) for image in images],
        "local_environment": [
            local_signature(image, migrant, int(config["local_neighbors_per_species"]))
            for image in images
        ],
        "displacement_field": [displacement_signature(image, images[0], migrant) for image in images],
        "bond_topology": [
            topology_signature(image, migrant, list(config["coordination_radii_A"]))
            for image in images
        ],
        "energy_profile": [[0.0] for _ in images],
    }
    return {
        "path_id": path_id,
        "system_id": system_id,
        "minimum_pair_distance_A": min(
            atoms.get_all_distances(mic=True)[
                __import__("numpy").triu_indices(len(atoms), 1)
            ].min()
            for atoms in read(path, index=":")
        ),
        "curves": {name: resample_curve(values, coordinates, count) for name, values in curves.items()},
    }


def geometry_distance(left, right, weights, scales):
    raw = raw_pair_distances(left, right)
    totals, normalized = {}, {}
    for orientation in ("forward", "reversed"):
        normalized[orientation] = {name: raw[orientation][name] / scales[name] for name in COMPONENTS}
        totals[orientation] = sum(weights[name] * normalized[orientation][name] for name in COMPONENTS)
    orientation = min(totals, key=totals.get)
    return {
        "alignment": orientation,
        "distance": totals[orientation],
        "similarity": math.exp(-totals[orientation]),
        "components": normalized[orientation],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--path-id", required=True)
    parser.add_argument("--system-id", default="Sb2Te3Cr_61")
    parser.add_argument("--selection-config", type=Path, required=True)
    parser.add_argument("--portfolio-manifest", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.selection_config.read_text())
    portfolio = json.loads(args.portfolio_manifest.read_text())
    historical = [
        row for row in json.loads(args.descriptors.read_text())["descriptors"]
        if row["system_id"] == args.system_id
    ]
    candidate = candidate_descriptor(args.images.resolve(), args.path_id, args.system_id, config)
    scales = portfolio["groups"][args.system_id]["distance_scales"]
    original_weights = config["distance_weights"]
    weight_sum = sum(original_weights[name] for name in COMPONENTS)
    weights = {name: original_weights[name] / weight_sum for name in COMPONENTS}
    all_rows = historical + [candidate]
    pair_rows = []
    similarity = {(row["path_id"], row["path_id"]): 1.0 for row in all_rows}
    for index, left in enumerate(all_rows):
        for right in all_rows[index + 1:]:
            result = geometry_distance(left, right, weights, scales)
            row = {"path_a": left["path_id"], "path_b": right["path_id"], **result}
            pair_rows.append(row)
            similarity[(left["path_id"], right["path_id"])] = result["similarity"]
            similarity[(right["path_id"], left["path_id"])] = result["similarity"]
    historical_ids = [row["path_id"] for row in historical]
    valid_centers = [row["path_id"] for row in historical if row["minimum_pair_distance_A"] >= 1.8]
    baseline = facility_coverage(historical_ids, valid_centers, similarity)
    augmented = facility_coverage(historical_ids, valid_centers + [candidate["path_id"]], similarity)
    candidate_pairs = [row for row in pair_rows if candidate["path_id"] in (row["path_a"], row["path_b"])]
    nearest = max(candidate_pairs, key=lambda row: row["similarity"])
    source_1_5 = next(row for row in candidate_pairs if "1-5" in (row["path_a"], row["path_b"]))
    result = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "geometry_only_diagnostic_no_energy_force_or_endpoint_acceptance",
        "candidate_path_id": candidate["path_id"],
        "candidate_images": str(args.images.resolve()),
        "candidate_images_sha256": sha256_file(args.images.resolve()),
        "system_id": args.system_id,
        "excluded_component": "energy_profile",
        "renormalized_weights": weights,
        "distance_scales": {name: scales[name] for name in COMPONENTS},
        "nearest_historical_path": nearest,
        "similarity_to_1-5": source_1_5["similarity"],
        "valid_center_geometry_coverage_before": baseline,
        "valid_center_geometry_coverage_after": augmented,
        "marginal_geometry_coverage_gain": augmented - baseline,
        "candidate_minimum_pair_distance_A": candidate["minimum_pair_distance_A"],
        "production_eligible": False,
        "next_gate": "MACE NEB pre-relaxation followed by same-mechanism comparison",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "geometry_coverage_score.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.out_dir / "candidate_pair_distances.csv").open("w", newline="") as handle:
        flat = []
        for row in candidate_pairs:
            flat.append({
                "path_a": row["path_a"], "path_b": row["path_b"],
                "alignment": row["alignment"], "distance": row["distance"],
                "similarity": row["similarity"],
                **{"component_" + key: value for key, value in row["components"].items()},
            })
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
