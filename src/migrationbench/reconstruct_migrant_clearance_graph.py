#!/usr/bin/env python3
"""Reconstruct a continuous migrant path as a shortest path through periodic free space."""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.io import read, write

from repair_neb_images import (
    attach_cell_from_qe_template,
    choose_migrant,
    min_pair_distance,
    path_stats,
    resample_images_by_arc_length,
    unwrap_images,
)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fibonacci_directions(count, seed):
    index = np.arange(count, dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phase = (int(seed) % 1009) * (2.0 * math.pi / 1009.0)
    theta = index * (math.pi * (3.0 - math.sqrt(5.0))) + phase
    directions = np.column_stack((radius * np.cos(theta), radius * np.sin(theta), z))
    axes = np.vstack((np.eye(3), -np.eye(3)))
    return np.vstack((directions, axes))


def cr_host_clearances(candidates, hosts, cell):
    inverse = np.linalg.inv(cell)
    delta = hosts[None, :, :] - candidates[:, None, :]
    scaled = np.matmul(delta, inverse)
    scaled -= np.rint(scaled)
    distances = np.linalg.norm(np.matmul(scaled, cell), axis=2)
    return distances.min(axis=1)


def candidate_layer(image, migrant, directions, radii, clearance, max_candidates):
    origin = image.positions[migrant].copy()
    offsets = [np.zeros(3)]
    offsets.extend(radius * direction for radius in radii for direction in directions)
    offsets = np.asarray(offsets)
    positions = origin + offsets
    hosts = np.delete(image.positions, migrant, axis=0)
    clearances = cr_host_clearances(positions, hosts, image.cell.array)
    valid = np.where(clearances >= clearance - 1.0e-10)[0]
    if not len(valid):
        raise RuntimeError("No clearance-valid migrant candidates for one image")
    ranking = sorted(valid, key=lambda i: (float(np.dot(offsets[i], offsets[i])), -float(clearances[i]), int(i)))
    selected = np.asarray(ranking[:max_candidates], dtype=int)
    return {
        "positions": positions[selected],
        "reference_offsets_A": np.linalg.norm(offsets[selected], axis=1),
        "clearances_A": clearances[selected],
        "generated_count": int(len(positions)),
        "valid_count": int(len(valid)),
    }


def transition_distances(left, right, cell):
    inverse = np.linalg.inv(cell)
    delta = right[None, :, :] - left[:, None, :]
    scaled = np.matmul(delta, inverse)
    scaled -= np.rint(scaled)
    return np.linalg.norm(np.matmul(scaled, cell), axis=2)


def shortest_layered_path(layers, cell, max_step, reference_weight, step_weight):
    costs = np.array([0.0])
    backpointers = []
    for layer_index in range(1, len(layers)):
        left, right = layers[layer_index - 1], layers[layer_index]
        distances = transition_distances(left["positions"], right["positions"], cell)
        transition = costs[:, None] + step_weight * distances * distances
        transition[distances > max_step + 1.0e-10] = np.inf
        parent = np.argmin(transition, axis=0)
        next_costs = transition[parent, np.arange(len(right["positions"]))]
        next_costs += reference_weight * right["reference_offsets_A"] ** 2
        if not np.isfinite(next_costs).any():
            raise RuntimeError(f"No connected candidate path at layer {layer_index} under max step {max_step} A")
        backpointers.append(parent)
        costs = next_costs
    selected = [int(np.argmin(costs))]
    for parent in reversed(backpointers):
        selected.append(int(parent[selected[-1]]))
    selected.reverse()
    return selected, float(np.min(costs))


def endpoint_layer(image, migrant, clearance):
    position = image.positions[migrant][None, :]
    hosts = np.delete(image.positions, migrant, axis=0)
    value = float(cr_host_clearances(position, hosts, image.cell.array)[0])
    if value < clearance - 1.0e-10:
        raise RuntimeError(f"Endpoint Cr-host clearance {value} A is below {clearance} A")
    return {
        "positions": position,
        "reference_offsets_A": np.array([0.0]),
        "clearances_A": np.array([value]),
        "generated_count": 1,
        "valid_count": 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--cell-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--migrant-element", default="Cr")
    parser.add_argument("--n-images", type=int, default=13)
    parser.add_argument("--clearance-A", type=float, default=1.8)
    parser.add_argument("--max-step-A", type=float, default=2.0)
    parser.add_argument("--max-offset-A", type=float, default=2.5)
    parser.add_argument("--radial-step-A", type=float, default=0.25)
    parser.add_argument("--directions", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--reference-weight", type=float, default=1.0)
    parser.add_argument("--step-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=48)
    args = parser.parse_args()
    if args.n_images < 3 or args.clearance_A <= 0 or args.max_step_A <= 0:
        raise ValueError("Invalid path dimensions or geometry gates")
    if args.radial_step_A <= 0 or args.max_offset_A < args.radial_step_A:
        raise ValueError("Invalid candidate-shell settings")

    source = args.images.resolve()
    images = read(source, index=":")
    images = attach_cell_from_qe_template(images, args.cell_template.resolve())
    migrant = choose_migrant(images, args.migrant_element)
    original = path_stats(images, migrant)
    reference = resample_images_by_arc_length(unwrap_images(images), args.n_images)
    directions = fibonacci_directions(args.directions, args.seed)
    radii = np.arange(args.radial_step_A, args.max_offset_A + 0.5 * args.radial_step_A, args.radial_step_A)
    layers = [endpoint_layer(reference[0], migrant, args.clearance_A)]
    layers.extend(
        candidate_layer(image, migrant, directions, radii, args.clearance_A, args.max_candidates)
        for image in reference[1:-1]
    )
    layers.append(endpoint_layer(reference[-1], migrant, args.clearance_A))
    selected, objective = shortest_layered_path(
        layers, reference[0].cell.array, args.max_step_A, args.reference_weight, args.step_weight
    )
    output_images = [image.copy() for image in reference]
    selection_rows = []
    for image_index, (image, layer, candidate_index) in enumerate(zip(output_images, layers, selected)):
        image.positions[migrant] = layer["positions"][candidate_index]
        image.info.update({
            "path_transform": "periodic_clearance_graph",
            "path_image_index": image_index,
            "migrant_element": args.migrant_element,
        })
        selection_rows.append({
            "image_index_zero_based": image_index,
            "generated_candidates": layer["generated_count"],
            "clearance_valid_candidates": layer["valid_count"],
            "selected_candidate_index": candidate_index,
            "selected_reference_offset_A": float(layer["reference_offsets_A"][candidate_index]),
            "selected_cr_host_clearance_A": float(layer["clearances_A"][candidate_index]),
        })
    final_stats = path_stats(output_images, migrant)
    host_indices = [index for index in range(len(output_images[0])) if index != migrant]
    minimum_host_only = min(min_pair_distance(image[host_indices])[0] for image in output_images)
    gates = {
        "all_pair_clearance": final_stats["min_pair_distance_A"] >= args.clearance_A - 1.0e-8,
        "host_only_clearance": minimum_host_only >= args.clearance_A - 1.0e-8,
        "maximum_migrant_step": final_stats["max_migrant_step_A"] <= args.max_step_A + 1.0e-8,
        "endpoints_exact": bool(
            np.allclose(output_images[0].positions, reference[0].positions, atol=1.0e-10)
            and np.allclose(output_images[-1].positions, reference[-1].positions, atol=1.0e-10)
        ),
        "periodic_cell": bool(all(image.pbc.all() and image.cell.rank == 3 for image in output_images)),
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    write(args.output.resolve(), output_images)
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "geometry_only_periodic_clearance_graph_initializer",
        "seed": args.seed,
        "source_images": str(source),
        "source_images_sha256": sha256_file(source),
        "cell_template": str(args.cell_template.resolve()),
        "cell_template_sha256": sha256_file(args.cell_template.resolve()),
        "output_images": str(args.output.resolve()),
        "output_images_sha256": sha256_file(args.output.resolve()),
        "migrant_index_zero_based": migrant,
        "parameters": {
            "n_images": args.n_images,
            "clearance_A": args.clearance_A,
            "max_step_A": args.max_step_A,
            "max_offset_A": args.max_offset_A,
            "radial_step_A": args.radial_step_A,
            "directions": args.directions,
            "max_candidates": args.max_candidates,
            "reference_weight": args.reference_weight,
            "step_weight": args.step_weight,
        },
        "objective": "reference_weight*sum(|r_i-r_i_ref|^2)+step_weight*sum(|MIC(r_i-r_(i-1))|^2)",
        "objective_value": objective,
        "before": original,
        "after": final_stats,
        "minimum_host_only_distance_A": minimum_host_only,
        "selection": selection_rows,
        "gates": gates,
        "accepted_geometry_initializer": all(gates.values()),
        "energy_force_labels": "absent_by_design",
        "warning": "Geometry-only initializer; MACE and independent QE validation are required.",
    }
    args.manifest.resolve().write_text(json.dumps(manifest, indent=2) + "\n")
    if not manifest["accepted_geometry_initializer"]:
        raise SystemExit("Clearance-graph path failed geometry gates; see manifest")
    print(json.dumps({"output": str(args.output.resolve()), "before": original, "after": final_stats, "objective": objective}, indent=2))


if __name__ == "__main__":
    main()
