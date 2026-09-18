#!/usr/bin/env python3
"""Select historical trajectory representatives by curve distance and coverage.

The historical candidates are diagnostic inputs only. Selection does not turn
unconverged paths or unverified endpoints into accepted benchmark references.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "migrationbench"))

from analyze_neb_path_topology import (  # noqa: E402
    choose_migrant,
    displacement,
    norm,
    read_images,
    read_qe_image,
    sha256_file,
)
from parse_qe_neb_output import parse_neb_out  # noqa: E402


COMPONENTS = (
    "geometry_frechet",
    "local_environment",
    "displacement_field",
    "energy_profile",
    "bond_topology",
)


def rms_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Descriptor length mismatch: {len(left)} != {len(right)}")
    if not left:
        return 0.0
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / len(left))


def discrete_frechet(left: list[list[float]], right: list[list[float]]) -> float:
    """Discrete Frechet distance for vector-valued curves."""
    cache = [[0.0] * len(right) for _ in left]
    for i in range(len(left)):
        for j in range(len(right)):
            distance = rms_distance(left[i], right[j])
            if i == 0 and j == 0:
                cache[i][j] = distance
            elif i == 0:
                cache[i][j] = max(cache[i][j - 1], distance)
            elif j == 0:
                cache[i][j] = max(cache[i - 1][j], distance)
            else:
                cache[i][j] = max(min(cache[i - 1][j], cache[i - 1][j - 1], cache[i][j - 1]), distance)
    return cache[-1][-1]


def resample_curve(values: list[list[float]], coordinates: list[float], count: int) -> list[list[float]]:
    if len(values) != len(coordinates) or not values:
        raise ValueError("Values and coordinates must be non-empty and have equal length")
    if count < 2:
        raise ValueError("resample_points must be at least 2")
    targets = [index / (count - 1) for index in range(count)]
    result = []
    cursor = 0
    for target in targets:
        while cursor + 1 < len(coordinates) and coordinates[cursor + 1] < target:
            cursor += 1
        if cursor + 1 == len(coordinates):
            result.append(list(values[-1]))
            continue
        lo, hi = coordinates[cursor], coordinates[cursor + 1]
        fraction = 0.0 if hi - lo < 1.0e-14 else (target - lo) / (hi - lo)
        result.append([a + fraction * (b - a) for a, b in zip(values[cursor], values[cursor + 1])])
    return result


def arc_coordinates(images: list[dict]) -> tuple[list[float], float]:
    cell = images[0]["cell"]
    segments = []
    for left, right in zip(images, images[1:]):
        vectors = [displacement(a, b, cell) for a, b in zip(left["positions"], right["positions"])]
        segments.append(math.sqrt(sum(norm(vector) ** 2 for vector in vectors) / len(vectors)))
    total = sum(segments)
    if total < 1.0e-14:
        return [index / (len(images) - 1) for index in range(len(images))], 0.0
    cumulative = [0.0]
    for segment in segments:
        cumulative.append(cumulative[-1] + segment)
    return [value / total for value in cumulative], total


def pair_signature(image: dict) -> list[float]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    symbols, positions, cell = image["symbols"], image["positions"], image["cell"]
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            key = tuple(sorted((symbols[i], symbols[j])))
            groups[key].append(norm(displacement(positions[i], positions[j], cell)))
    signature = []
    for key in sorted(groups):
        signature.extend(sorted(groups[key]))
    return signature


def local_signature(image: dict, migrant: int, neighbors: int) -> list[float]:
    groups: dict[str, list[float]] = defaultdict(list)
    symbols, positions, cell = image["symbols"], image["positions"], image["cell"]
    for index, symbol in enumerate(symbols):
        if index != migrant:
            groups[symbol].append(norm(displacement(positions[migrant], positions[index], cell)))
    signature = []
    for symbol in sorted(groups):
        values = sorted(groups[symbol])
        pad = values[-1] if values else 0.0
        signature.extend((values + [pad] * neighbors)[:neighbors])
    return signature


def displacement_signature(image: dict, initial: dict, migrant: int) -> list[float]:
    groups: dict[str, list[float]] = defaultdict(list)
    cell = image["cell"]
    for index, symbol in enumerate(image["symbols"]):
        value = norm(displacement(initial["positions"][index], image["positions"][index], cell))
        groups[symbol].append(value)
    signature = []
    for symbol in sorted(groups):
        signature.extend(sorted(groups[symbol]))
    signature.append(norm(displacement(initial["positions"][migrant], image["positions"][migrant], cell)))
    return signature


def topology_signature(image: dict, migrant: int, radii: list[float]) -> list[float]:
    groups: dict[str, list[float]] = defaultdict(list)
    symbols, positions, cell = image["symbols"], image["positions"], image["cell"]
    for index, symbol in enumerate(symbols):
        if index != migrant:
            groups[symbol].append(norm(displacement(positions[migrant], positions[index], cell)))
    result = []
    for symbol in sorted(groups):
        values = groups[symbol]
        result.extend(float(sum(distance <= radius for distance in values)) for radius in radii)
        result.append(min(values))
    return result


def normalized_energy_curve(rows: list[dict]) -> list[list[float]]:
    energies = [float(row["energy_eV"]) for row in rows]
    lo, hi = min(energies), max(energies)
    scale = hi - lo
    if scale < 1.0e-12:
        return [[0.0] for _ in energies]
    return [[(energy - lo) / scale] for energy in energies]


def attach_cell(images: list[dict], cell: list[list[float]] | None) -> None:
    if cell is not None:
        for image in images:
            image["cell"] = cell


def build_descriptor(row: dict, config: dict, root: Path) -> dict:
    run_dir = root / row["run_dir"]
    images, image_sources, image_format = read_images(run_dir)
    override = config.get("cell_overrides_by_system", {}).get(row["system_id"])
    cell_source = None
    if override:
        cell_source = root / override
        attach_cell(images, read_qe_image(cell_source)["cell"])
    if any(image["symbols"] != images[0]["symbols"] for image in images):
        raise ValueError(f"Composition/order mismatch in {row['path_id']}")
    migrant = choose_migrant(images, config["migrant_element"])
    parsed = parse_neb_out(run_dir / "neb.out")
    energy_rows = parsed.get("last_images") or []
    if len(energy_rows) != len(images):
        raise ValueError(f"Energy/image count mismatch in {row['path_id']}")
    coordinates, path_length = arc_coordinates(images)
    count = int(config["resample_points"])
    raw_curves = {
        "geometry_frechet": [pair_signature(image) for image in images],
        "local_environment": [local_signature(image, migrant, int(config["local_neighbors_per_species"])) for image in images],
        "displacement_field": [displacement_signature(image, images[0], migrant) for image in images],
        "energy_profile": normalized_energy_curve(energy_rows),
        "bond_topology": [topology_signature(image, migrant, list(config["coordination_radii_A"])) for image in images],
    }
    curves = {name: resample_curve(values, coordinates, count) for name, values in raw_curves.items()}
    max_error = parsed.get("max_image_error_eV_A")
    min_pair = min(
        norm(displacement(image["positions"][i], image["positions"][j], image["cell"]))
        for image in images
        for i in range(len(image["positions"]))
        for j in range(i + 1, len(image["positions"]))
    )
    geometry_score = 1.0 if min_pair >= 1.8 else 0.0
    force_score = math.exp(-float(max_error) / 0.30) if max_error is not None else 0.0
    return {
        "path_id": row["path_id"],
        "system_id": row["system_id"],
        "run_dir": str(run_dir),
        "image_format": image_format,
        "image_sources": [
            {"path": str(path), "sha256": sha256_file(path)} for path in image_sources
        ],
        "cell_source": None if cell_source is None else {"path": str(cell_source), "sha256": sha256_file(cell_source)},
        "neb_out_sha256": sha256_file(run_dir / "neb.out"),
        "n_atoms": len(images[0]["symbols"]),
        "n_images": len(images),
        "migrant_index_zero_based": migrant,
        "path_length_configuration_rms_A": path_length,
        "minimum_pair_distance_A": min_pair,
        "max_image_error_eV_A": max_error,
        "quality_tiebreak_score": 0.6 * geometry_score + 0.4 * force_score,
        "converged": bool(parsed.get("converged_by_default_gate")),
        "curves": curves,
    }


def raw_pair_distances(left: dict, right: dict) -> dict:
    if left["system_id"] != right["system_id"]:
        raise ValueError("Cross-system trajectory distance is undefined")
    result = {"forward": {}, "reversed": {}}
    for name in COMPONENTS:
        curve_left = left["curves"][name]
        curve_right = right["curves"][name]
        if name in {"geometry_frechet", "local_environment", "bond_topology"}:
            metric = discrete_frechet
        else:
            metric = lambda a, b: math.sqrt(sum(rms_distance(x, y) ** 2 for x, y in zip(a, b)) / len(a))
        result["forward"][name] = metric(curve_left, curve_right)
        result["reversed"][name] = metric(curve_left, list(reversed(curve_right)))
    return result


def robust_scales(pair_rows: list[dict]) -> dict:
    scales = {}
    for name in COMPONENTS:
        values = sorted(
            min(row["raw"]["forward"][name], row["raw"]["reversed"][name])
            for row in pair_rows
            if min(row["raw"]["forward"][name], row["raw"]["reversed"][name]) > 1.0e-12
        )
        scales[name] = values[len(values) // 2] if values else 1.0
    return scales


def finalize_distances(pair_rows: list[dict], weights: dict, scales: dict) -> None:
    for row in pair_rows:
        totals = {}
        normalized = {}
        for orientation in ("forward", "reversed"):
            normalized[orientation] = {
                name: row["raw"][orientation][name] / scales[name] for name in COMPONENTS
            }
            totals[orientation] = sum(weights[name] * normalized[orientation][name] for name in COMPONENTS)
        orientation = min(totals, key=totals.get)
        row["alignment"] = orientation
        row["components"] = normalized[orientation]
        row["distance"] = totals[orientation]
        row["similarity"] = math.exp(-row["distance"])


def facility_coverage(ids: list[str], selected: list[str], similarity: dict[tuple[str, str], float]) -> float:
    if not selected:
        return 0.0
    return sum(max(similarity[(candidate, representative)] for representative in selected) for candidate in ids) / len(ids)


def greedy_select(
    descriptors: list[dict],
    pair_rows: list[dict],
    target: float,
    allowed_centers: set[str] | None = None,
) -> list[dict]:
    ids = [row["path_id"] for row in descriptors]
    quality = {row["path_id"]: row["quality_tiebreak_score"] for row in descriptors}
    similarity = {(path_id, path_id): 1.0 for path_id in ids}
    for row in pair_rows:
        similarity[(row["path_a"], row["path_b"])] = row["similarity"]
        similarity[(row["path_b"], row["path_a"])] = row["similarity"]
    selected, records = [], []
    coverage = 0.0
    allowed = set(ids) if allowed_centers is None else set(allowed_centers)
    while len(selected) < len(allowed) and coverage + 1.0e-12 < target:
        choices = []
        for candidate in ids:
            if candidate in selected or candidate not in allowed:
                continue
            trial = facility_coverage(ids, selected + [candidate], similarity)
            choices.append((trial - coverage, quality[candidate], candidate, trial))
        gain, _, chosen, new_coverage = max(choices)
        selected.append(chosen)
        records.append(
            {
                "rank": len(selected),
                "path_id": chosen,
                "marginal_coverage_gain": gain,
                "cumulative_coverage": new_coverage,
                "quality_tiebreak_score": quality[chosen],
            }
        )
        coverage = new_coverage
    return records


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_outputs(groups: dict[str, dict], out_dir: Path) -> None:
    for system_id, result in groups.items():
        ids = result["path_ids"]
        matrix = [[1.0 if a == b else result["similarity"][(a, b)] for b in ids] for a in ids]
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="cividis")
        ax.set_xticks(range(len(ids)), ids, rotation=45, ha="right")
        ax.set_yticks(range(len(ids)), ids)
        ax.set_title(f"Trajectory similarity: {system_id}")
        fig.colorbar(image, ax=ax, label="exp(-D_traj)")
        fig.savefig(out_dir / f"{system_id}_similarity.png", dpi=180, facecolor="white")
        plt.close(fig)

        selected = result["selected"]
        valid_selected = result["valid_only_selected"]
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        ranks = [row["rank"] for row in selected]
        coverages = [row["cumulative_coverage"] for row in selected]
        ax.plot(ranks, coverages, marker="o", color="#2563eb", linewidth=2, label="all historical centers")
        if valid_selected:
            ax.plot(
                [row["rank"] for row in valid_selected],
                [row["cumulative_coverage"] for row in valid_selected],
                marker="s",
                color="#7c3aed",
                linewidth=2,
                label="geometry-valid centers only",
            )
        ax.axhline(result["target"], color="#b45309", linestyle="--", label=f"target={result['target']:.2f}")
        ax.set_xticks(ranks)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("Number of selected representatives")
        ax.set_ylabel("Weighted facility-location coverage")
        ax.set_title(f"Coverage curve: {system_id}")
        ax.grid(axis="y", color="#e5e7eb")
        ax.legend()
        fig.savefig(out_dir / f"{system_id}_coverage.png", dpi=180, facecolor="white")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "historical_trajectory_portfolio.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data_processed" / "representative_trajectory_portfolio")
    parser.add_argument("--seed", type=int, default=42, help="Recorded for job identity; the selector is deterministic.")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    descriptors = [build_descriptor(row, config, ROOT) for row in config["candidate_paths"]]

    grouped: dict[str, list[dict]] = defaultdict(list)
    for descriptor in descriptors:
        grouped[descriptor["system_id"]].append(descriptor)
    group_results = {}
    flat_pairs, flat_selected, coverage_gaps = [], [], []
    target = float(config["coverage"]["target"])
    for system_id, rows in sorted(grouped.items()):
        pairs = []
        for index, left in enumerate(rows):
            for right in rows[index + 1 :]:
                pairs.append({"system_id": system_id, "path_a": left["path_id"], "path_b": right["path_id"], "raw": raw_pair_distances(left, right)})
        scales = robust_scales(pairs)
        finalize_distances(pairs, config["distance_weights"], scales)
        selected = greedy_select(rows, pairs, target)
        similarity = {(row["path_a"], row["path_b"]): row["similarity"] for row in pairs}
        similarity.update({(b, a): value for (a, b), value in list(similarity.items())})
        similarity.update({(row["path_id"], row["path_id"]): 1.0 for row in rows})
        valid_ids = {
            row["path_id"] for row in rows if row["minimum_pair_distance_A"] >= 1.8
        }
        valid_only_selected = greedy_select(rows, pairs, target, allowed_centers=valid_ids)
        valid_only_ceiling = facility_coverage(
            [row["path_id"] for row in rows], sorted(valid_ids), similarity
        )
        for row in rows:
            if row["path_id"] in valid_ids:
                continue
            nearest = max(valid_ids, key=lambda candidate: similarity[(row["path_id"], candidate)])
            coverage_gaps.append({
                "system_id": system_id,
                "uncovered_invalid_path": row["path_id"],
                "nearest_geometry_valid_path": nearest,
                "similarity_to_nearest_valid": similarity[(row["path_id"], nearest)],
                "residual_gap": 1.0 - similarity[(row["path_id"], nearest)],
                "required_action": "generate_repaired_or_new_valid_path_in_this_descriptor_region",
            })
        group_results[system_id] = {
            "path_ids": [row["path_id"] for row in rows],
            "distance_scales": scales,
            "selected": selected,
            "valid_center_path_ids": sorted(valid_ids),
            "valid_only_selected": valid_only_selected,
            "valid_only_coverage_ceiling": valid_only_ceiling,
            "target_reachable_with_current_valid_centers": valid_only_ceiling >= target,
            "similarity": similarity,
            "target": target,
        }
        for pair in pairs:
            flat_pairs.append({
                "system_id": system_id,
                "path_a": pair["path_a"],
                "path_b": pair["path_b"],
                "alignment": pair["alignment"],
                "D_traj": pair["distance"],
                "similarity": pair["similarity"],
                **{f"D_{name}": pair["components"][name] for name in COMPONENTS},
            })
        descriptor_by_id = {descriptor["path_id"]: descriptor for descriptor in rows}
        for selected_row in selected:
            descriptor = descriptor_by_id[selected_row["path_id"]]
            geometry_valid = descriptor["minimum_pair_distance_A"] >= 1.8
            flat_selected.append({
                "system_id": system_id,
                **selected_row,
                "minimum_pair_distance_A": descriptor["minimum_pair_distance_A"],
                "max_image_error_eV_A": descriptor["max_image_error_eV_A"],
                "selection_role": (
                    "representative_candidate_requires_endpoint_validation"
                    if geometry_valid
                    else "coverage_witness_requires_repair_or_replacement"
                ),
            })

    write_csv(args.out_dir / "pairwise_trajectory_distances.csv", flat_pairs, [
        "system_id", "path_a", "path_b", "alignment", "D_traj", "similarity", *[f"D_{name}" for name in COMPONENTS]
    ])
    write_csv(args.out_dir / "representative_paths.csv", flat_selected, [
        "system_id", "rank", "path_id", "marginal_coverage_gain", "cumulative_coverage",
        "quality_tiebreak_score", "minimum_pair_distance_A", "max_image_error_eV_A", "selection_role"
    ])
    write_csv(args.out_dir / "coverage_gaps.csv", coverage_gaps, [
        "system_id", "uncovered_invalid_path", "nearest_geometry_valid_path",
        "similarity_to_nearest_valid", "residual_gap", "required_action"
    ])
    descriptor_payload = {"descriptors": descriptors}
    (args.out_dir / "trajectory_descriptors.json").write_text(json.dumps(descriptor_payload, indent=2) + "\n")
    serializable_groups = {
        system_id: {key: value for key, value in result.items() if key != "similarity"}
        for system_id, result in group_results.items()
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_only_no_endpoint_or_neb_acceptance",
        "seed": args.seed,
        "deterministic_selection": True,
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "distance_contract": "weighted normalized component distance; direction chosen jointly by minimum total forward/reversed alignment",
        "coverage_contract": config["coverage"],
        "lineage_refinements_not_independent_candidates": config["lineage_refinements_not_independent_candidates"],
        "groups": serializable_groups,
        "quality_checks": {
            "candidate_count": len(descriptors),
            "cross_system_pairs_created": 0,
            "all_paths_unconverged_or_historical": all(not row["converged"] for row in descriptors),
            "coverage_monotonic_all": all(
                all(b["cumulative_coverage"] >= a["cumulative_coverage"] for a, b in zip(result["selected"], result["selected"][1:]))
                for result in group_results.values()
            ),
            "coverage_target_reached_all": all(result["selected"][-1]["cumulative_coverage"] >= target for result in group_results.values()),
            "valid_center_target_reached_all": all(
                result["target_reachable_with_current_valid_centers"] for result in group_results.values()
            ),
        },
    }
    (args.out_dir / "portfolio_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot_outputs(group_results, args.out_dir)
    print(json.dumps({"groups": serializable_groups, "quality_checks": manifest["quality_checks"]}, indent=2))


if __name__ == "__main__":
    main()
