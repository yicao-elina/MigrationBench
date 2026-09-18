#!/usr/bin/env python3
"""Map a periodic NEB path across dopants and tetradymite host cations.

The mapper is deliberately geometry-only: it preserves atom order and
fractional coordinates, substitutes element labels, optionally rescales the
cell, writes extxyz paths, and records enough provenance for downstream MACE
and QE jobs to decide whether the mapped path is only diagnostic or eligible
for stronger gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_STAGE1_SYSTEMS = [
    {
        "system_id": "tetradymite__Sb2Te3__Cr1__ingap_1-7__reused_path_s42",
        "host_formula": "Sb2Te3",
        "target_dopant": "Cr",
        "target_host_cation": "Sb",
        "cell_scale_ab": 1.0,
        "cell_scale_c": 1.0,
        "role": "original_baseline_pipeline_check",
    },
    {
        "system_id": "tetradymite__Sb2Te3__Mn1__ingap_1-7__reused_path_s42",
        "host_formula": "Sb2Te3",
        "target_dopant": "Mn",
        "target_host_cation": "Sb",
        "cell_scale_ab": 1.0,
        "cell_scale_c": 1.0,
        "role": "dopant_transfer",
    },
    {
        "system_id": "tetradymite__Bi2Te3__Cr1__ingap_1-7__reused_path_s42",
        "host_formula": "Bi2Te3",
        "target_dopant": "Cr",
        "target_host_cation": "Bi",
        "cell_scale_ab": 1.0286116323,
        "cell_scale_c": 1.001280452,
        "role": "host_cation_transfer",
    },
    {
        "system_id": "tetradymite__Bi2Te3__Mn1__ingap_1-7__reused_path_s42",
        "host_formula": "Bi2Te3",
        "target_dopant": "Mn",
        "target_host_cation": "Bi",
        "cell_scale_ab": 1.0286116323,
        "cell_scale_c": 1.001280452,
        "role": "combined_dopant_and_host_transfer",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def subtract(left, right):
    return tuple(a - b for a, b in zip(left, right))


def add(left, right):
    return tuple(a + b for a, b in zip(left, right))


def scale(vector, factor):
    return tuple(factor * value for value in vector)


def norm(vector):
    return math.sqrt(dot(vector, vector))


def det3(matrix):
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def inverse3(matrix):
    determinant = det3(matrix)
    if abs(determinant) <= 1.0e-12:
        raise ValueError("Cell matrix is singular")
    a, b, c = matrix
    cofactor = [
        (b[1] * c[2] - b[2] * c[1], a[2] * c[1] - a[1] * c[2], a[1] * b[2] - a[2] * b[1]),
        (b[2] * c[0] - b[0] * c[2], a[0] * c[2] - a[2] * c[0], a[2] * b[0] - a[0] * b[2]),
        (b[0] * c[1] - b[1] * c[0], a[1] * c[0] - a[0] * c[1], a[0] * b[1] - a[1] * b[0]),
    ]
    return [[value / determinant for value in row] for row in cofactor]


def matvec(vector, matrix):
    return tuple(
        vector[0] * matrix[0][axis]
        + vector[1] * matrix[1][axis]
        + vector[2] * matrix[2][axis]
        for axis in range(3)
    )


def parse_lattice(header: str):
    match = re.search(r'Lattice="([^"]+)"', header)
    if not match:
        raise ValueError("extxyz header is missing Lattice")
    values = [float(value) for value in match.group(1).split()]
    if len(values) != 9:
        raise ValueError("Lattice must contain 9 numeric values")
    return [tuple(values[index : index + 3]) for index in range(0, 9, 3)]


def read_extxyz(path: Path):
    lines = path.read_text().splitlines()
    frames = []
    offset = 0
    while offset < len(lines):
        if not lines[offset].strip():
            offset += 1
            continue
        nat = int(lines[offset].strip())
        header = lines[offset + 1]
        atoms = lines[offset + 2 : offset + 2 + nat]
        if len(atoms) != nat:
            raise ValueError(f"Truncated extxyz frame at line {offset + 1}")
        symbols = []
        positions = []
        for line in atoms:
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Bad atom line: {line}")
            symbols.append(parts[0])
            positions.append(tuple(float(value) for value in parts[1:4]))
        frames.append({"symbols": symbols, "positions": positions, "cell": parse_lattice(header)})
        offset += nat + 2
    if len(frames) < 4:
        raise ValueError("NEB path needs at least four images")
    first_symbols = frames[0]["symbols"]
    for index, frame in enumerate(frames[1:], start=1):
        if frame["symbols"] != first_symbols:
            raise ValueError(f"Atom order changes at image {index}")
    return frames


def fractional_positions(frame):
    inv = inverse3(frame["cell"])
    return [matvec(position, inv) for position in frame["positions"]]


def scale_cell(cell, scale_ab, scale_c):
    return [
        scale(cell[0], scale_ab),
        scale(cell[1], scale_ab),
        scale(cell[2], scale_c),
    ]


def wrap_delta_fractional(left, right):
    raw = subtract(right, left)
    return tuple(value - round(value) for value in raw)


def pbc_displacement(left, right, cell):
    return matvec(wrap_delta_fractional(left, right), cell)


def min_distances(frames, migrant_index):
    min_pair = float("inf")
    min_migrant_host = float("inf")
    max_migrant_step = 0.0
    total_migrant_path = 0.0
    fractional = [fractional_positions(frame) for frame in frames]
    for frame, frac in zip(frames, fractional):
        for i in range(len(frac)):
            for j in range(i + 1, len(frac)):
                distance = norm(pbc_displacement(frac[i], frac[j], frame["cell"]))
                min_pair = min(min_pair, distance)
                if i == migrant_index or j == migrant_index:
                    min_migrant_host = min(min_migrant_host, distance)
    for left, right, frame in zip(fractional, fractional[1:], frames):
        step = norm(pbc_displacement(left[migrant_index], right[migrant_index], frame["cell"]))
        max_migrant_step = max(max_migrant_step, step)
        total_migrant_path += step
    return {
        "min_pair_distance_A": min_pair,
        "min_migrant_host_distance_A": min_migrant_host,
        "max_migrant_step_A": max_migrant_step,
        "total_migrant_path_A": total_migrant_path,
    }


def composition(symbols):
    return dict(sorted(Counter(symbols).items()))


def map_frame(frame, source_dopant, source_host_cation, target_dopant, target_host_cation, scale_ab, scale_c):
    frac = fractional_positions(frame)
    new_cell = scale_cell(frame["cell"], scale_ab, scale_c)
    symbols = []
    for symbol in frame["symbols"]:
        if symbol == source_dopant:
            symbols.append(target_dopant)
        elif symbol == source_host_cation:
            symbols.append(target_host_cation)
        else:
            symbols.append(symbol)
    positions = [matvec(position, new_cell) for position in frac]
    return {"symbols": symbols, "positions": positions, "cell": new_cell}


def write_extxyz(path: Path, frames, branch_id, system):
    chunks = []
    for index, frame in enumerate(frames):
        lattice = " ".join(f"{value:.12f}" for row in frame["cell"] for value in row)
        header = (
            f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 pbc="T T T" '
            f'branch_id={branch_id} image_index={index} system_id={system["system_id"]} '
            "mapping_role=material_transfer_reused_path"
        )
        chunks.append(str(len(frame["symbols"])))
        chunks.append(header)
        for symbol, position in zip(frame["symbols"], frame["positions"]):
            chunks.append(f"{symbol:2s} {position[0]: .12f} {position[1]: .12f} {position[2]: .12f}")
    path.write_text("\n".join(chunks) + "\n")


def load_systems(path: Path | None):
    if not path:
        return DEFAULT_STAGE1_SYSTEMS
    payload = json.loads(path.read_text())
    return payload["systems"] if isinstance(payload, dict) else payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-images", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--systems-json", type=Path)
    parser.add_argument("--source-path-id", default="1-7")
    parser.add_argument("--source-mechanism-id", default="ingap_1-7")
    parser.add_argument("--source-dopant", default="Cr")
    parser.add_argument("--source-host-cation", default="Sb")
    parser.add_argument("--min-pair-distance-A", type=float, default=1.80)
    parser.add_argument("--min-migrant-host-distance-A", type=float, default=1.40)
    parser.add_argument("--max-migrant-step-A", type=float, default=2.50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_images = args.source_images.resolve()
    source_frames = read_extxyz(source_images)
    source_symbols = source_frames[0]["symbols"]
    source_migrants = [index for index, symbol in enumerate(source_symbols) if symbol == args.source_dopant]
    if len(source_migrants) != 1:
        raise ValueError(f"Expected exactly one source dopant {args.source_dopant}, found {len(source_migrants)}")
    migrant_index = source_migrants[0]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    system_manifests = []
    systems = load_systems(args.systems_json)
    for system in systems:
        target_dopant = system["target_dopant"]
        target_host_cation = system["target_host_cation"]
        branch_id = system.get("branch_id") or system["system_id"]
        mapped = [
            map_frame(
                frame,
                args.source_dopant,
                args.source_host_cation,
                target_dopant,
                target_host_cation,
                float(system.get("cell_scale_ab", 1.0)),
                float(system.get("cell_scale_c", 1.0)),
            )
            for frame in source_frames
        ]
        system_dir = args.out_dir / system["system_id"] / "reused_path"
        system_dir.mkdir(parents=True, exist_ok=True)
        images_path = system_dir / "mapped_images.extxyz"
        write_extxyz(images_path, mapped, branch_id, system)
        metrics = min_distances(mapped, migrant_index)
        failed = []
        if metrics["min_pair_distance_A"] < args.min_pair_distance_A:
            failed.append("minimum_pair_distance")
        if metrics["min_migrant_host_distance_A"] < args.min_migrant_host_distance_A:
            failed.append("minimum_migrant_host_distance")
        if metrics["max_migrant_step_A"] > args.max_migrant_step_A:
            failed.append("maximum_migrant_step")
        gate = {
            "status": "pass" if not failed else "fail",
            "failed_checks": failed,
            "thresholds": {
                "min_pair_distance_A": args.min_pair_distance_A,
                "min_migrant_host_distance_A": args.min_migrant_host_distance_A,
                "max_migrant_step_A": args.max_migrant_step_A,
            },
            "metrics": metrics,
        }
        audit = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "system": system,
            "source_images": str(source_images),
            "source_images_sha256": sha256_file(source_images),
            "source_path_id": args.source_path_id,
            "source_mechanism_id": args.source_mechanism_id,
            "source_composition": composition(source_symbols),
            "mapped_composition": composition(mapped[0]["symbols"]),
            "source_dopant": args.source_dopant,
            "source_host_cation": args.source_host_cation,
            "migrant_index_zero_based": migrant_index,
            "mapping_method": "fractional_coordinate_preserving_element_substitution",
            "cell_scaling": {
                "mode": "host_cation_lattice_ratio_for_Bi2Te3_rows_else_identity",
                "ab": float(system.get("cell_scale_ab", 1.0)),
                "c": float(system.get("cell_scale_c", 1.0)),
                "note": "Bi2Te3 rows use tetradymite lattice-ratio scaling; endpoints remain unrelaxed diagnostics.",
            },
            "geometry_gate": gate,
            "outputs": {
                "images": str(images_path),
                "images_sha256": sha256_file(images_path),
            },
        }
        audit_path = system_dir / "geometry_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        row = {
            "branch_id": branch_id,
            "path_id": args.source_path_id,
            "system_id": system["system_id"],
            "host_formula": system["host_formula"],
            "dopant": target_dopant,
            "migrant_element": target_dopant,
            "host_cation": target_host_cation,
            "transfer_role": system["role"],
            "initializer": "reused_path_material_transfer",
            "mapping_method": audit["mapping_method"],
            "images": str(images_path.relative_to(args.out_dir)),
            "images_sha256": sha256_file(images_path),
            "geometry_audit": str(audit_path.relative_to(args.out_dir)),
            "geometry_audit_sha256": sha256_file(audit_path),
            "geometry_gate": gate,
            "duplicate_of": None,
            "contains_energy_labels": False,
            "contains_force_labels": False,
            "production_eligible_before_mace": False,
            "scientific_role": "first_stage_material_transfer_diagnostic_preconditioner",
        }
        candidates.append(row)
        system_manifests.append(audit)

    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "first_stage_MACE_to_DFT_NEB_material_transfer_validation",
        "endpoint_status": "mapped_unrelaxed_unverified_endpoints",
        "source_images": str(source_images),
        "source_images_sha256": sha256_file(source_images),
        "source_path_id": args.source_path_id,
        "source_mechanism_id": args.source_mechanism_id,
        "source_dopant": args.source_dopant,
        "source_host_cation": args.source_host_cation,
        "migrant_index_zero_based": migrant_index,
        "seed": args.seed,
        "systems": systems,
        "candidates": candidates,
        "label_policy": "Mapped paths contain no labels; MACE/DFT must append energies and forces with calculator provenance.",
        "acceptance_boundary": "Diagnostic warm-start only until endpoint-repeat, calculator-identity, and QE-NEB convergence gates pass.",
    }
    manifest_path = args.out_dir / "nonlinear_candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "systems": len(systems),
        "geometry_gate_pass": sum(row["geometry_gate"]["status"] == "pass" for row in candidates),
        "manifest": str(manifest_path.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
