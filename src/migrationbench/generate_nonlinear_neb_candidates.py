#!/usr/bin/env python3
"""Generate traceable nonlinear NEB initial paths without fabricated labels."""

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from ase.io import read as ase_read

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_neb_path_topology import (  # noqa: E402
    displacement,
    geometry_descriptors,
    norm,
    read_qe_image,
)
from parse_qe_neb_path_history import parse_path_file  # noqa: E402


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def scale(a, value):
    return tuple(value * x for x in a)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def unit(a):
    length = norm(a)
    if length <= 1.0e-12:
        raise ValueError("Cannot normalize a zero vector")
    return scale(a, 1.0 / length)


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def validate_endpoints(initial, final):
    if initial["symbols"] != final["symbols"]:
        raise ValueError("Endpoint atom symbols/order differ")
    if initial["cell"] is None or final["cell"] is None:
        raise ValueError("Both endpoints require CELL_PARAMETERS")
    maximum_cell_delta = max(
        abs(a - b)
        for left, right in zip(initial["cell"], final["cell"])
        for a, b in zip(left, right)
    )
    if maximum_cell_delta > 1.0e-6:
        raise ValueError(f"Endpoint cells differ by {maximum_cell_delta:.3e} A")


def choose_migrant(image, element):
    indices = [index for index, symbol in enumerate(image["symbols"]) if symbol == element]
    if len(indices) != 1:
        raise ValueError(f"Expected exactly one {element} migrant, found {len(indices)}")
    return indices[0]


def linear_images(initial, final, n_images):
    cell = initial["cell"]
    vectors = [
        displacement(a, b, cell)
        for a, b in zip(initial["positions"], final["positions"])
    ]
    images = []
    for index in range(n_images):
        s = index / (n_images - 1)
        positions = [add(position, scale(vector, s)) for position, vector in zip(initial["positions"], vectors)]
        images.append({"symbols": initial["symbols"], "positions": positions, "cell": cell})
    images[0]["positions"] = list(initial["positions"])
    # Keep the last frame in the same unwrapped branch as the path. It is exactly
    # equivalent to the requested final endpoint under PBC and avoids a fake NEB
    # spring jump when the input endpoint was wrapped across a cell boundary.
    return images


def perpendicular_directions(chord, cell, minimum_angle_degrees=15.0):
    chord_unit = unit(chord)
    maximum_abs_cosine = math.cos(math.radians(minimum_angle_degrees))
    candidates = list(cell) + [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    directions = []
    for axis in candidates:
        candidate = subtract(axis, scale(chord_unit, dot(axis, chord_unit)))
        if norm(candidate) <= 1.0e-8:
            continue
        candidate = unit(candidate)
        for sign in (1.0, -1.0):
            signed = scale(candidate, sign)
            if all(abs(dot(signed, old)) < maximum_abs_cosine for old in directions):
                directions.append(signed)
    return directions


def clearance_arc(initial, final, n_images, migrant_index, amplitude, direction):
    images = linear_images(initial, final, n_images)
    for index, image in enumerate(images[1:-1], start=1):
        s = index / (n_images - 1)
        offset = scale(direction, amplitude * math.sin(math.pi * s))
        image["positions"][migrant_index] = add(image["positions"][migrant_index], offset)
    return images


def unwrap_images(images):
    cell = images[0]["cell"]
    unwrapped = [list(images[0]["positions"])]
    for previous, current in zip(images, images[1:]):
        unwrapped.append([
            add(old, displacement(prev, now, cell))
            for old, prev, now in zip(unwrapped[-1], previous["positions"], current["positions"])
        ])
    return unwrapped


def configuration_arc_grid(images):
    cell = images[0]["cell"]
    cumulative = [0.0]
    for left, right in zip(images, images[1:]):
        vectors = [displacement(a, b, cell) for a, b in zip(left["positions"], right["positions"])]
        cumulative.append(cumulative[-1] + math.sqrt(sum(norm(v) ** 2 for v in vectors) / len(vectors)))
    if cumulative[-1] <= 1.0e-12:
        return [index / (len(images) - 1) for index in range(len(images))]
    return [value / cumulative[-1] for value in cumulative]


def interpolate_frames(frames, grid, s):
    if s <= 0.0:
        return frames[0]
    if s >= 1.0:
        return frames[-1]
    right = next(index for index, value in enumerate(grid) if value >= s)
    left = right - 1
    fraction = (s - grid[left]) / max(grid[right] - grid[left], 1.0e-12)
    return [
        add(a, scale(subtract(b, a), fraction))
        for a, b in zip(frames[left], frames[right])
    ]


def historical_warp(initial, final, historical, n_images):
    validate_endpoints(historical[0], historical[-1])
    if historical[0]["symbols"] != initial["symbols"]:
        raise ValueError("Historical path atom symbols/order differ from target endpoints")
    target = linear_images(initial, final, n_images)
    unwrapped = unwrap_images(historical)
    grid = configuration_arc_grid(historical)
    hist_start, hist_end = unwrapped[0], unwrapped[-1]
    for index, image in enumerate(target[1:-1], start=1):
        s = index / (n_images - 1)
        sampled = interpolate_frames(unwrapped, grid, s)
        hist_linear = [add(a, scale(subtract(b, a), s)) for a, b in zip(hist_start, hist_end)]
        image["positions"] = [
            add(position, subtract(curved, straight))
            for position, curved, straight in zip(image["positions"], sampled, hist_linear)
        ]
    return target


def geometry_gate(images, migrant_index, args):
    metrics = geometry_descriptors(images, migrant_index)
    max_migrant_step = max(
        (
            norm(displacement(left["positions"][migrant_index], right["positions"][migrant_index], left["cell"]))
            for left, right in zip(images, images[1:])
        ),
        default=0.0,
    )
    reasons = []
    if metrics["minimum_pair_distance_A"] < args.min_pair_distance_A:
        reasons.append("minimum_pair_distance")
    if metrics["minimum_migrant_host_distance_A"] < args.min_migrant_host_distance_A:
        reasons.append("minimum_migrant_host_distance")
    if max_migrant_step > args.max_migrant_step_A:
        reasons.append("maximum_migrant_inter_image_step")
    return {
        "status": "pass" if not reasons else "fail",
        "reasons": reasons,
        "thresholds": {
            "min_pair_distance_A": args.min_pair_distance_A,
            "min_migrant_host_distance_A": args.min_migrant_host_distance_A,
            "max_migrant_step_A": args.max_migrant_step_A,
        },
        "metrics": metrics,
        "max_migrant_inter_image_step_A": max_migrant_step,
    }


def endpoint_pbc_deviation(generated, requested):
    return max(
        norm(displacement(a, b, generated["cell"]))
        for a, b in zip(generated["positions"], requested["positions"])
    )


def path_rms_distance(left, right, atom_indices=None):
    if len(left) != len(right) or len(left[0]["positions"]) != len(right[0]["positions"]):
        raise ValueError("Path shapes differ")
    atom_indices = list(atom_indices) if atom_indices is not None else list(range(len(left[0]["positions"])))
    squared = []
    for left_image, right_image in zip(left, right):
        squared.extend(
            norm(displacement(left_image["positions"][index], right_image["positions"][index], left_image["cell"])) ** 2
            for index in atom_indices
        )
    return math.sqrt(sum(squared) / len(squared))


def write_extxyz(path, images, branch_id):
    chunks = []
    for index, image in enumerate(images):
        lattice = " ".join(f"{value:.12f}" for row in image["cell"] for value in row)
        chunks.extend([
            str(len(image["symbols"])),
            f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T" branch_id={branch_id} image_index={index}',
        ])
        chunks.extend(
            f"{symbol} {position[0]:.12f} {position[1]:.12f} {position[2]:.12f}"
            for symbol, position in zip(image["symbols"], image["positions"])
        )
    path.write_text("\n".join(chunks) + "\n")


def historical_images(path_file, template):
    template_image = read_qe_image(template)
    parsed = parse_path_file(path_file, len(template_image["symbols"]))
    return [
        {
            "symbols": template_image["symbols"],
            "positions": [tuple(atom["position_A"]) for atom in image["atoms"]],
            "cell": template_image["cell"],
        }
        for image in parsed["images"]
    ]


def historical_ase_images(images_file, template, start_image_qe=None, end_image_qe=None):
    template_image = read_qe_image(template)
    frames = ase_read(images_file, index=":")
    if not frames:
        raise ValueError(f"Historical ASE trajectory has no frames: {images_file}")
    start = 1 if start_image_qe is None else int(start_image_qe)
    end = len(frames) if end_image_qe is None else int(end_image_qe)
    if start < 1 or end > len(frames) or start >= end:
        raise ValueError(
            f"Invalid inclusive historical image range {start}:{end} for {len(frames)} frames"
        )
    selected = frames[start - 1 : end]
    rows = []
    for offset, atoms in enumerate(selected, start=start):
        if atoms.get_chemical_symbols() != template_image["symbols"]:
            raise ValueError(f"Historical ASE atom order differs from QE template at image {offset}")
        rows.append({
            "symbols": template_image["symbols"],
            "positions": [tuple(float(value) for value in row) for row in atoms.get_positions()],
            "cell": template_image["cell"],
        })
    return rows


def read_endpoint_structure(path):
    atoms = ase_read(path, index=-1)
    cell = [tuple(float(value) for value in row) for row in atoms.get_cell()]
    if not any(abs(value) > 0 for row in cell for value in row):
        raise ValueError(f"Endpoint structure has no periodic cell: {path}")
    return {
        "symbols": atoms.get_chemical_symbols(),
        "positions": [tuple(float(value) for value in row) for row in atoms.get_positions()],
        "cell": cell,
    }


def validate_production_endpoint_acceptance(
    endpoint_status, acceptance_path, initial_source, final_source
):
    if endpoint_status != "accepted_local_minima":
        return None
    if not acceptance_path or not acceptance_path.is_file():
        raise ValueError("Production candidates require an endpoint-acceptance artifact")
    acceptance = json.loads(acceptance_path.read_text())
    expected = {
        "initial_structure_sha256": sha256_file(initial_source),
        "final_structure_sha256": sha256_file(final_source),
    }
    failed = [key for key, value in expected.items() if acceptance.get(key) != value]
    if acceptance.get("status") != "accepted" or failed:
        raise ValueError("Endpoint acceptance gate failed: " + ", ".join(failed or ["status"]))
    return acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-qe", type=Path)
    parser.add_argument("--final-qe", type=Path)
    parser.add_argument("--initial-structure", type=Path)
    parser.add_argument("--final-structure", type=Path)
    parser.add_argument("--endpoint-acceptance", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-images", type=int, default=7)
    parser.add_argument("--migrant-element", default="Cr")
    parser.add_argument("--arc-amplitudes-A", default="0.25,0.50")
    parser.add_argument("--max-arc-directions", type=int, default=4)
    parser.add_argument("--direction-min-angle-deg", type=float, default=15.0)
    parser.add_argument("--historical-path-file", type=Path)
    parser.add_argument("--historical-images", type=Path)
    parser.add_argument("--historical-template-qe", type=Path)
    parser.add_argument("--historical-start-image-qe", type=int)
    parser.add_argument("--historical-end-image-qe", type=int)
    parser.add_argument("--min-pair-distance-A", type=float, default=1.7)
    parser.add_argument("--min-migrant-host-distance-A", type=float, default=1.4)
    parser.add_argument("--max-migrant-step-A", type=float, default=3.0)
    parser.add_argument("--duplicate-all-atom-rms-A", type=float, default=0.02)
    parser.add_argument("--duplicate-migrant-rms-A", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--endpoint-status", choices=["accepted_local_minima", "unverified_smoke"],
        default="unverified_smoke",
    )
    args = parser.parse_args()
    if args.n_images < 4:
        raise ValueError("NEB requires at least four images")
    historical_sources = sum(bool(value) for value in (args.historical_path_file, args.historical_images))
    if historical_sources > 1:
        raise ValueError("Provide only one historical curvature source")
    if bool(historical_sources) != bool(args.historical_template_qe):
        raise ValueError("A historical path/images source and QE template must be provided together")
    if not args.historical_images and (
        args.historical_start_image_qe is not None or args.historical_end_image_qe is not None
    ):
        raise ValueError("Historical image range is valid only with --historical-images")
    qe_pair = bool(args.initial_qe) and bool(args.final_qe)
    structure_pair = bool(args.initial_structure) and bool(args.final_structure)
    if qe_pair == structure_pair:
        raise ValueError("Provide exactly one complete endpoint pair: QE inputs or ASE structures")
    if args.endpoint_status == "accepted_local_minima" and not structure_pair:
        raise ValueError("Production candidates require final accepted endpoint structures, not QE input geometries")
    if structure_pair:
        initial_source, final_source = args.initial_structure.resolve(), args.final_structure.resolve()
        initial, final = read_endpoint_structure(initial_source), read_endpoint_structure(final_source)
    else:
        initial_source, final_source = args.initial_qe.resolve(), args.final_qe.resolve()
        initial, final = read_qe_image(initial_source), read_qe_image(final_source)
    endpoint_acceptance = validate_production_endpoint_acceptance(
        args.endpoint_status, args.endpoint_acceptance, initial_source, final_source
    )
    validate_endpoints(initial, final)
    migrant_index = choose_migrant(initial, args.migrant_element)
    chord = displacement(initial["positions"][migrant_index], final["positions"][migrant_index], initial["cell"])
    directions = perpendicular_directions(
        chord, initial["cell"], args.direction_min_angle_deg
    )
    amplitudes = [float(value) for value in args.arc_amplitudes_A.split(",") if value.strip()]
    candidates = [("linear_mic", {}, linear_images(initial, final, args.n_images))]
    for amplitude in amplitudes:
        for direction_index, direction in enumerate(directions[: args.max_arc_directions]):
            candidates.append((
                f"clearance_arc_a{amplitude:g}_d{direction_index + 1}",
                {"amplitude_A": amplitude, "direction_unit": direction},
                clearance_arc(initial, final, args.n_images, migrant_index, amplitude, direction),
            ))
    historical_source = args.historical_path_file or args.historical_images
    if historical_source:
        if args.historical_path_file:
            history = historical_images(args.historical_path_file, args.historical_template_qe)
            source_type = "qe_path_restart"
        else:
            history = historical_ase_images(
                args.historical_images,
                args.historical_template_qe,
                args.historical_start_image_qe,
                args.historical_end_image_qe,
            )
            source_type = "ase_multiframe_segment"
        candidates.append((
            "historical_curvature_warp",
            {
                "historical_source_type": source_type,
                "historical_path_file": str(args.historical_path_file.resolve()) if args.historical_path_file else None,
                "historical_path_sha256": sha256_file(args.historical_path_file) if args.historical_path_file else None,
                "historical_images": str(args.historical_images.resolve()) if args.historical_images else None,
                "historical_images_sha256": sha256_file(args.historical_images) if args.historical_images else None,
                "historical_start_image_qe": args.historical_start_image_qe,
                "historical_end_image_qe": args.historical_end_image_qe,
                "historical_template_qe": str(args.historical_template_qe.resolve()),
                "historical_template_sha256": sha256_file(args.historical_template_qe),
            },
            historical_warp(initial, final, history, args.n_images),
        ))
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    accepted_unique_paths = []
    for branch_id, parameters, images in candidates:
        output = out_dir / f"{branch_id}.extxyz"
        write_extxyz(output, images, branch_id)
        gate = geometry_gate(images, migrant_index, args)
        distances = [
            {
                "branch_id": old_branch,
                "all_atom_rms_A": path_rms_distance(images, old_images),
                "migrant_rms_A": path_rms_distance(images, old_images, [migrant_index]),
            }
            for old_branch, old_images in accepted_unique_paths
        ]
        nearest = min(
            (
                row for row in distances
            ),
            key=lambda row: (row["migrant_rms_A"], row["all_atom_rms_A"], row["branch_id"]),
            default=None,
        )
        duplicate_of = (
            nearest["branch_id"]
            if nearest is not None
            and nearest["migrant_rms_A"] <= args.duplicate_migrant_rms_A
            and nearest["all_atom_rms_A"] <= args.duplicate_all_atom_rms_A
            else None
        )
        if gate["status"] == "pass" and duplicate_of is None:
            accepted_unique_paths.append((branch_id, images))
        rows.append({
            "branch_id": branch_id,
            "initializer": "clearance_arc" if branch_id.startswith("clearance_arc") else branch_id,
            "parameters": parameters,
            "geometry_gate": gate,
            "nearest_prior_unique_path": nearest,
            "duplicate_of": duplicate_of,
            "initial_endpoint_pbc_deviation_A": endpoint_pbc_deviation(images[0], initial),
            "final_endpoint_pbc_deviation_A": endpoint_pbc_deviation(images[-1], final),
            "images": output.name,
            "images_sha256": sha256_file(output),
            "contains_energy_labels": False,
            "contains_force_labels": False,
            "production_eligible_before_mace": (
                args.endpoint_status == "accepted_local_minima"
                and gate["status"] == "pass"
                and duplicate_of is None
            ),
        })
    ranked = sorted(
        (
            row for row in rows
            if row["geometry_gate"]["status"] == "pass" and row["duplicate_of"] is None
        ),
        key=lambda row: (
            -row["geometry_gate"]["metrics"]["minimum_migrant_host_distance_A"],
            row["geometry_gate"]["metrics"]["configuration_tortuosity"],
            row["branch_id"],
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["geometry_screen_rank"] = rank
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "nonlinear_geometry_only_neb_candidate_generation",
        "seed": args.seed,
        "endpoint_source_type": "accepted_relaxed_structure" if structure_pair else "qe_input_geometry",
        "initial_qe": str(initial_source) if qe_pair else None,
        "initial_qe_sha256": sha256_file(initial_source) if qe_pair else None,
        "final_qe": str(final_source) if qe_pair else None,
        "final_qe_sha256": sha256_file(final_source) if qe_pair else None,
        "initial_endpoint_source": str(initial_source),
        "initial_endpoint_source_sha256": sha256_file(initial_source),
        "final_endpoint_source": str(final_source),
        "final_endpoint_source_sha256": sha256_file(final_source),
        "endpoint_acceptance": str(args.endpoint_acceptance.resolve()) if args.endpoint_acceptance else None,
        "endpoint_acceptance_sha256": sha256_file(args.endpoint_acceptance) if args.endpoint_acceptance else None,
        "migrant_element": args.migrant_element,
        "migrant_index_zero_based": migrant_index,
        "n_images": args.n_images,
        "direction_min_angle_deg": args.direction_min_angle_deg,
        "endpoint_status": args.endpoint_status,
        "candidates": rows,
        "duplicate_thresholds_A": {
            "all_atom_path_rms": args.duplicate_all_atom_rms_A,
            "migrant_path_rms": args.duplicate_migrant_rms_A,
        },
        "selection_rule": "Run MACE on every geometry-gate-passing nonduplicate candidate only after both endpoints are accepted local minima; linear_mic is a diagnostic baseline, not the preferred initializer.",
        "label_policy": "Geometry-only candidates contain no energies or forces. MACE and DFT labels must be appended only by their calculators with provenance.",
    }
    manifest_path = out_dir / "nonlinear_candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "candidates": len(rows),
        "geometry_gate_pass": sum(row["geometry_gate"]["status"] == "pass" for row in rows),
        "unique_geometry_gate_pass": sum(
            row["geometry_gate"]["status"] == "pass" and row["duplicate_of"] is None
            for row in rows
        ),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
