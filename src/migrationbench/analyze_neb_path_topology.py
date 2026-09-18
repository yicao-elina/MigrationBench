#!/usr/bin/env python3
"""Detect hidden basins and segment barriers in QE NEB trajectories.

The analyzer is dependency-free so the same audit runs locally and on Rockfish.
It accepts either multi-frame XYZ images or numbered QE pw_*.in images and pairs
them with the last complete image-energy table in neb.out.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_qe_neb_output import parse_neb_out  # noqa: E402


FLOAT_RE = r"[-+0-9.Ee]+"
PW_INDEX_RE = re.compile(r"pw_(\d+)\.in$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_xyz(path: Path) -> list[dict]:
    lines = path.read_text(errors="replace").splitlines()
    images = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        nat = int(lines[cursor].strip())
        cursor += 1
        comment = lines[cursor] if cursor < len(lines) else ""
        cursor += 1
        symbols, positions = [], []
        for _ in range(nat):
            fields = lines[cursor].split()
            cursor += 1
            symbols.append(fields[0])
            positions.append(tuple(float(value) for value in fields[1:4]))
        images.append({"symbols": symbols, "positions": positions, "cell": None, "comment": comment})
    return images


def read_qe_image(path: Path) -> dict:
    lines = path.read_text(errors="replace").splitlines()
    symbols, positions, cell = [], [], None
    for idx, line in enumerate(lines):
        if line.strip().upper().startswith("ATOMIC_POSITIONS"):
            cursor = idx + 1
            while cursor < len(lines):
                fields = lines[cursor].split()
                if len(fields) < 4 or not re.fullmatch(FLOAT_RE, fields[1]):
                    break
                symbols.append(fields[0])
                positions.append(tuple(float(value) for value in fields[1:4]))
                cursor += 1
        if line.strip().upper().startswith("CELL_PARAMETERS"):
            rows = []
            for raw in lines[idx + 1 : idx + 4]:
                fields = raw.split()
                rows.append(tuple(float(value) for value in fields[:3]))
            cell = rows
    if not positions:
        raise ValueError(f"No ATOMIC_POSITIONS found in {path}")
    return {"symbols": symbols, "positions": positions, "cell": cell, "comment": ""}


def read_images(run_dir: Path) -> tuple[list[dict], list[Path], str]:
    xyz = run_dir / "sb2te3.xyz"
    if xyz.exists():
        images = read_xyz(xyz)
        sources = [xyz]
        qe_template = run_dir / "pw_1.in"
        if qe_template.exists():
            template = read_qe_image(qe_template)
            for image in images:
                image["cell"] = template["cell"]
            sources.append(qe_template)
            image_format = "multi_frame_xyz_with_qe_cell"
        else:
            image_format = "multi_frame_xyz"
        return images, sources, image_format
    pw_files = sorted(
        run_dir.glob("pw_*.in"),
        key=lambda path: int(PW_INDEX_RE.search(path.name).group(1)) if PW_INDEX_RE.search(path.name) else 10**9,
    )
    if len(pw_files) >= 2:
        return [read_qe_image(path) for path in pw_files], pw_files, "qe_numbered_inputs"
    raise ValueError(f"No multi-frame sb2te3.xyz or at least two pw_*.in files in {run_dir}")


def determinant3(matrix):
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def inverse3(matrix):
    a, b, c = matrix
    det = determinant3(matrix)
    if abs(det) < 1.0e-12:
        raise ValueError("Singular cell matrix")
    return [
        [(b[1] * c[2] - b[2] * c[1]) / det, (a[2] * c[1] - a[1] * c[2]) / det, (a[1] * b[2] - a[2] * b[1]) / det],
        [(b[2] * c[0] - b[0] * c[2]) / det, (a[0] * c[2] - a[2] * c[0]) / det, (a[2] * b[0] - a[0] * b[2]) / det],
        [(b[0] * c[1] - b[1] * c[0]) / det, (a[1] * c[0] - a[0] * c[1]) / det, (a[0] * b[1] - a[1] * b[0]) / det],
    ]


def row_times_matrix(vector, matrix):
    return tuple(sum(vector[k] * matrix[k][j] for k in range(3)) for j in range(3))


def displacement(a, b, cell):
    delta = tuple(b[i] - a[i] for i in range(3))
    if cell is None:
        return delta
    fractional = row_times_matrix(delta, inverse3(cell))
    wrapped = tuple(value - round(value) for value in fractional)
    return row_times_matrix(wrapped, cell)


def norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def choose_migrant(images, element):
    candidates = [i for i, symbol in enumerate(images[0]["symbols"]) if symbol == element]
    if not candidates:
        raise ValueError(f"No migrant element {element!r}")
    cell = images[0]["cell"]
    return max(candidates, key=lambda i: norm(displacement(images[0]["positions"][i], images[-1]["positions"][i], cell)))


def geometry_descriptors(images, migrant_index):
    cell = images[0]["cell"]
    segment_rms, migrant_steps, host_rms = [], [], []
    minimum_pair = math.inf
    minimum_migrant_host = math.inf
    for image in images:
        pos = image["positions"]
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                distance = norm(displacement(pos[i], pos[j], cell))
                minimum_pair = min(minimum_pair, distance)
                if i == migrant_index or j == migrant_index:
                    minimum_migrant_host = min(minimum_migrant_host, distance)
    for left, right in zip(images, images[1:]):
        vectors = [displacement(a, b, cell) for a, b in zip(left["positions"], right["positions"])]
        segment_rms.append(math.sqrt(sum(norm(v) ** 2 for v in vectors) / len(vectors)))
        migrant_steps.append(norm(vectors[migrant_index]))
        host = [v for i, v in enumerate(vectors) if i != migrant_index]
        host_rms.append(math.sqrt(sum(norm(v) ** 2 for v in host) / len(host)))
    endpoint_vectors = [
        displacement(a, b, cell)
        for a, b in zip(images[0]["positions"], images[-1]["positions"])
    ]
    chord = math.sqrt(sum(norm(v) ** 2 for v in endpoint_vectors) / len(endpoint_vectors))
    migrant_chord = norm(endpoint_vectors[migrant_index])
    total = sum(segment_rms)
    migrant_total = sum(migrant_steps)
    return {
        "configuration_path_length_rms_A": total,
        "configuration_endpoint_chord_rms_A": chord,
        "configuration_tortuosity": total / chord if chord > 1.0e-12 else None,
        "migrant_path_length_A": migrant_total,
        "migrant_endpoint_chord_A": migrant_chord,
        "migrant_tortuosity": migrant_total / migrant_chord if migrant_chord > 1.0e-12 else None,
        "host_step_rms_mean_A": sum(host_rms) / len(host_rms) if host_rms else 0.0,
        "minimum_pair_distance_A": minimum_pair,
        "minimum_migrant_host_distance_A": minimum_migrant_host,
        "cell_available": cell is not None,
    }


def energy_topology(image_rows, config):
    energies = [float(row["energy_eV"]) for row in image_rows]
    errors = [row.get("error_eV_A") for row in image_rows]
    tolerance = float(config["energy_local_minimum_tolerance_eV"])
    basin_indices = [0]
    basin_rows = []
    for idx in range(1, len(energies) - 1):
        depth = min(energies[idx - 1], energies[idx + 1]) - energies[idx]
        if depth >= tolerance:
            basin_indices.append(idx)
            error = errors[idx]
            basin_rows.append(
                {
                    "image_index_zero_based": idx,
                    "image_index_qe": idx + 1,
                    "energy_eV": energies[idx],
                    "energy_relative_global_min_eV": energies[idx] - min(energies),
                    "local_basin_depth_eV": depth,
                    "image_error_eV_A": error,
                    "candidate_status": "candidate_needs_independent_relax" if error is None or error <= float(config["candidate_force_max_eV_A"]) else "weak_candidate_high_neb_force",
                }
            )
    basin_indices.append(len(energies) - 1)
    segments = []
    for start, end in zip(basin_indices, basin_indices[1:]):
        saddle = max(range(start, end + 1), key=lambda i: energies[i])
        endpoint_dominated = saddle in {start, end}
        segments.append(
            {
                "start_image_qe": start + 1,
                "end_image_qe": end + 1,
                "saddle_image_qe": saddle + 1,
                "delta_E_eV": energies[end] - energies[start],
                "barrier_forward_eV": energies[saddle] - energies[start],
                "barrier_reverse_eV": energies[saddle] - energies[end],
                "endpoint_dominated": endpoint_dominated,
                "segment_status": "unresolved_no_interior_saddle" if endpoint_dominated else "candidate_segment_barrier",
            }
        )
    return basin_rows, segments, basin_indices


def parse_spec(spec):
    if "=" not in spec:
        path = Path(spec).resolve()
        return path.name, path
    path_id, raw = spec.split("=", 1)
    return path_id, Path(raw).resolve()


def analyze_one(path_id, run_dir, config):
    images, image_sources, image_format = read_images(run_dir)
    symbols = images[0]["symbols"]
    consistent = all(image["symbols"] == symbols for image in images)
    if not consistent:
        raise ValueError(f"Composition/order changes across images for {path_id}")
    neb_path = run_dir / "neb.out"
    parsed = parse_neb_out(neb_path)
    image_rows = parsed["last_images"]
    energy_count_matches = len(image_rows) == len(images)
    migrant_index = choose_migrant(images, config.get("migrant_element", "Cr"))
    geometry = geometry_descriptors(images, migrant_index)
    basins, segments, basin_indices = ([], [], [0, len(images) - 1])
    if energy_count_matches and image_rows:
        basins, segments, basin_indices = energy_topology(image_rows, config)
    min_pair_ok = geometry["minimum_pair_distance_A"] >= float(config["minimum_pair_distance_A"])
    min_migrant_ok = geometry["minimum_migrant_host_distance_A"] >= float(config["minimum_migrant_host_distance_A"])
    endpoint_stability = "unverified_frozen_neb_endpoints"
    topology_status = "hidden_basin_candidate" if basins else "single_segment_candidate"
    if any(segment["endpoint_dominated"] for segment in segments):
        topology_status += ";endpoint_dominated_segment"
    return {
        "path_id": path_id,
        "run_dir": str(run_dir),
        "image_format": image_format,
        "image_sources": [{"path": str(path), "sha256": sha256_file(path)} for path in image_sources],
        "neb_out": str(neb_path),
        "neb_out_sha256": sha256_file(neb_path),
        "n_images_geometry": len(images),
        "n_images_energy": len(image_rows),
        "n_atoms": len(symbols),
        "migrant_element": symbols[migrant_index],
        "migrant_index_zero_based": migrant_index,
        "composition_order_consistent": consistent,
        "energy_count_matches_geometry": energy_count_matches,
        "endpoint_stability_status": endpoint_stability,
        "internal_basin_candidate_count": len(basins),
        "basin_image_indices_qe": [index + 1 for index in basin_indices],
        "topology_status": topology_status,
        "geometry_gate_pass": min_pair_ok and min_migrant_ok,
        **geometry,
        "qe_summary": parsed,
        "internal_basin_candidates": basins,
        "segments": segments,
    }


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", action="append", required=True, help="PATH_ID=RUN_DIR; repeatable")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [analyze_one(*parse_spec(spec), config) for spec in args.path]

    summary_rows, basin_rows, segment_rows = [], [], []
    for result in results:
        qe = result["qe_summary"]
        summary_rows.append(
            {
                "path_id": result["path_id"],
                "n_atoms": result["n_atoms"],
                "n_images": result["n_images_geometry"],
                "activation_forward_qe_eV": qe.get("activation_forward_eV"),
                "max_image_error_eV_A": qe.get("max_image_error_eV_A"),
                "job_done": qe.get("job_done"),
                "internal_basin_candidate_count": result["internal_basin_candidate_count"],
                "topology_status": result["topology_status"],
                "endpoint_stability_status": result["endpoint_stability_status"],
                "configuration_tortuosity": result["configuration_tortuosity"],
                "migrant_tortuosity": result["migrant_tortuosity"],
                "minimum_pair_distance_A": result["minimum_pair_distance_A"],
                "geometry_gate_pass": result["geometry_gate_pass"],
                "cell_available": result["cell_available"],
            }
        )
        for row in result["internal_basin_candidates"]:
            basin_rows.append({"path_id": result["path_id"], **row})
        for row in result["segments"]:
            segment_rows.append({"path_id": result["path_id"], **row})

    write_csv(out_dir / "path_topology_summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(out_dir / "basin_candidates.csv", basin_rows, ["path_id", "image_index_zero_based", "image_index_qe", "energy_eV", "energy_relative_global_min_eV", "local_basin_depth_eV", "image_error_eV_A", "candidate_status"])
    write_csv(out_dir / "segmented_barriers.csv", segment_rows, ["path_id", "start_image_qe", "end_image_qe", "saddle_image_qe", "delta_E_eV", "barrier_forward_eV", "barrier_reverse_eV", "endpoint_dominated", "segment_status"])
    quality = {
        "path_id_unique": len({row["path_id"] for row in results}) == len(results),
        "composition_order_consistent_all": all(row["composition_order_consistent"] for row in results),
        "energy_count_matches_geometry_all": all(row["energy_count_matches_geometry"] for row in results),
        "geometry_gate_pass_count": sum(row["geometry_gate_pass"] for row in results),
        "cell_missing_count": sum(not row["cell_available"] for row in results),
        "unverified_endpoint_count": 2 * len(results),
    }
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality_checks": quality,
        "results": results,
    }
    (out_dir / "path_topology_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    report = [
        "# NEB Path Topology Audit",
        "",
        "All endpoint stability labels remain unverified because the inspected NEB endpoints are frozen. Internal minima are candidates for standalone endpoint relaxation, not accepted minima.",
        "",
        "| Path | QE forward (eV) | Hidden basin candidates | Topology | Max error (eV/A) |",
        "|---|---:|---:|---|---:|",
    ]
    for row in summary_rows:
        report.append(
            f"| {row['path_id']} | {row['activation_forward_qe_eV']} | {row['internal_basin_candidate_count']} | {row['topology_status']} | {row['max_image_error_eV_A']} |"
        )
    report.extend(["", "Segment barriers are diagnostic until endpoint relaxations and converged QE NEB calculations pass their gates.", ""])
    (out_dir / "README.md").write_text("\n".join(report))
    print(json.dumps({"paths": len(results), "basin_candidates": len(basin_rows), "segments": len(segment_rows), "quality_checks": quality}, indent=2))


if __name__ == "__main__":
    main()
