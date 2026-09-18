#!/usr/bin/env python3
"""Classify migration paths with continuous geometric descriptors.

The labels are intentionally secondary. The durable dataset should store the
continuous descriptors first, then derive labels from documented thresholds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.io import read


ACT_FWD_RE = re.compile(r"activation energy \(->\)\s*=\s*([-+0-9.Ee]+)\s*eV")
ACT_REV_RE = re.compile(r"activation energy \(<-\)\s*=\s*([-+0-9.Ee]+)\s*eV")
ERR_RE = re.compile(r"^\s*\d+\s+[-+0-9.Ee]+\s+([-+0-9.Ee]+)\s+[FT]\s*$")
ITER_RE = re.compile(r"-+ iteration\s+(\d+)\s+-+")
IMAGE_ROW_RE = re.compile(r"^\s*(\d+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([FT])\s*$")


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_cell_from_qe_input(path: Path) -> np.ndarray:
    lines = path.read_text(errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.strip().upper().startswith("CELL_PARAMETERS"):
            if "angstrom" not in line.lower():
                raise ValueError(f"Only CELL_PARAMETERS angstrom is supported: {path}")
            return np.asarray(
                [[float(value) for value in lines[index + offset].split()[:3]] for offset in (1, 2, 3)],
                dtype=float,
            )
    raise ValueError(f"No CELL_PARAMETERS found in {path}")


def mic_vector(start: np.ndarray, end: np.ndarray, cell: np.ndarray | None) -> np.ndarray:
    delta = np.asarray(end) - np.asarray(start)
    if cell is not None:
        fractional = delta @ np.linalg.inv(cell)
        fractional -= np.round(fractional)
        delta = fractional @ cell
    return delta


def parse_neb_tail(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    iterations: list[dict] = []
    current: dict | None = None
    in_image_table = False
    job_done = False
    warnings = 0
    for line in path.read_text(errors="replace").splitlines():
        if "JOB DONE" in line:
            job_done = True
        if "WARNING" in line or "convergence NOT achieved" in line:
            warnings += 1
        iter_match = ITER_RE.search(line)
        if iter_match:
            current = {"iteration": int(iter_match.group(1)), "images": []}
            iterations.append(current)
            in_image_table = False
            continue
        m = ACT_FWD_RE.search(line)
        if m and current is not None:
            current["activation_forward_eV"] = float(m.group(1))
            continue
        m = ACT_REV_RE.search(line)
        if m and current is not None:
            current["activation_reverse_eV"] = float(m.group(1))
            continue
        if line.strip().startswith("image") and "energy" in line and "error" in line:
            in_image_table = True
            continue
        if in_image_table and current is not None:
            row = IMAGE_ROW_RE.match(line)
            if row:
                current["images"].append(
                    {
                        "image_index": int(row.group(1)),
                        "energy_eV": float(row.group(2)),
                        "error_eV_A": float(row.group(3)),
                        "frozen": row.group(4) == "T",
                    }
                )
            elif current.get("images") and not line.strip():
                in_image_table = False
    complete = [
        row
        for row in iterations
        if "activation_forward_eV" in row and "activation_reverse_eV" in row and row.get("images")
    ]
    last = complete[-1] if complete else {}
    errors = [float(row["error_eV_A"]) for row in last.get("images", [])]
    return {
        "activation_forward_eV": last.get("activation_forward_eV"),
        "activation_reverse_eV": last.get("activation_reverse_eV"),
        "job_done": job_done,
        "warning_count": warnings,
        "max_reported_image_error_eV_A": max(errors) if errors else None,
        "last_complete_iteration": last.get("iteration"),
    }


def choose_migrant(images, migrant_element: str) -> int:
    symbols = images[0].get_chemical_symbols()
    candidates = [i for i, sym in enumerate(symbols) if sym == migrant_element]
    if not candidates:
        raise ValueError(f"No atoms with migrant element {migrant_element!r}.")
    if len(candidates) == 1:
        return candidates[0]
    start = images[0].get_positions()
    end = images[-1].get_positions()
    cell = np.asarray(images[0].cell) if images[0].cell.rank == 3 and any(images[0].pbc) else None
    displacements = [np.linalg.norm(mic_vector(start[i], end[i], cell)) for i in candidates]
    return candidates[int(np.argmax(displacements))]


def min_host_distances(atoms, migrant_index: int, host_indices: list[int]) -> np.ndarray:
    pos = atoms.get_positions()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    distances = []
    for j in host_indices:
        if cell.rank == 3 and any(pbc):
            distances.append(atoms.get_distance(migrant_index, j, mic=True))
        else:
            distances.append(float(np.linalg.norm(pos[migrant_index] - pos[j])))
    return np.asarray(distances, dtype=float)


def z_plane_metrics(host_z_ref: np.ndarray, cr_z: float, sigma: float) -> dict:
    dz = np.abs(host_z_ref - cr_z)
    nearest = float(dz.min())
    density = float(np.mean(np.exp(-0.5 * (dz / sigma) ** 2)))
    return {"nearest_host_plane_dz_A": nearest, "z_density_score": density}


def classify_from_features(features: dict) -> tuple[str, float]:
    penetration = features["penetration_score"]
    gap = features["gap_score"]
    coordination = features["mean_coordination"]
    min_distance = features["min_host_distance_A"]

    if penetration >= 0.62 and coordination >= 3.0:
        return "deep_penetration", min(0.98, 0.55 + 0.35 * penetration + 0.03 * coordination)
    if gap >= 0.62 and coordination <= 2.2 and min_distance >= 2.2:
        return "in_gap_or_open_channel", min(0.98, 0.55 + 0.35 * gap)
    if penetration >= 0.50 and gap >= 0.45:
        return "mixed_transition", 0.65
    if penetration >= gap:
        return "probable_deep_penetration", 0.55 + 0.25 * penetration
    return "probable_in_gap_or_open_channel", 0.55 + 0.25 * gap


def analyze_path(path: Path, args: argparse.Namespace) -> dict:
    images = read(path, index=":")
    if len(images) < 2:
        raise ValueError(f"{path} has fewer than two images.")
    cell_source = None
    if args.cell_input:
        cell_source = args.cell_input.resolve()
    elif images[0].cell.rank != 3:
        cell_source = next(
            (candidate for candidate in (path.parent / "neb.in", path.parent / "neb_reconstructed.in", path.parent / "pw_1.in") if candidate.exists()),
            None,
        )
    if cell_source:
        cell = read_cell_from_qe_input(cell_source)
        for atoms in images:
            atoms.set_cell(cell)
            atoms.set_pbc(True)

    migrant_index = choose_migrant(images, args.migrant_element)
    symbols = images[0].get_chemical_symbols()
    host_indices = [i for i, sym in enumerate(symbols) if i != migrant_index and sym != args.migrant_element]
    host_z_ref = images[0].get_positions()[host_indices, 2]

    rows = []
    cr_positions = []
    for image_index, atoms in enumerate(images):
        cr_pos = atoms.get_positions()[migrant_index]
        cr_positions.append(cr_pos)
        distances = min_host_distances(atoms, migrant_index, host_indices)
        coord = int(np.sum(distances <= args.coord_cutoff_A))
        z_metrics = z_plane_metrics(host_z_ref, float(cr_pos[2]), args.z_density_sigma_A)
        rows.append(
            {
                "image_index": image_index,
                "cr_x_A": float(cr_pos[0]),
                "cr_y_A": float(cr_pos[1]),
                "cr_z_A": float(cr_pos[2]),
                "nearest_host_distance_A": float(distances.min()),
                "coordination": coord,
                **z_metrics,
            }
        )

    positions = np.asarray(cr_positions, dtype=float)
    cell = np.asarray(images[0].cell) if images[0].cell.rank == 3 and any(images[0].pbc) else None
    segment_vectors = np.asarray(
        [mic_vector(positions[index], positions[index + 1], cell) for index in range(len(positions) - 1)]
    )
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    endpoint_vector = mic_vector(positions[0], positions[-1], cell)
    nearest = np.asarray([row["nearest_host_distance_A"] for row in rows])
    coord = np.asarray([row["coordination"] for row in rows], dtype=float)
    z_density = np.asarray([row["z_density_score"] for row in rows])
    nearest_z = np.asarray([row["nearest_host_plane_dz_A"] for row in rows])

    coordination_score = float(np.clip(coord.mean() / args.deep_coordination_scale, 0.0, 1.0))
    density_score = float(np.clip(z_density.mean() / args.deep_density_scale, 0.0, 1.0))
    close_contact_score = float(np.clip((args.open_channel_distance_A - nearest.min()) / args.open_channel_distance_A, 0.0, 1.0))
    penetration_score = float(np.clip(0.45 * coordination_score + 0.35 * density_score + 0.20 * close_contact_score, 0.0, 1.0))

    open_distance_score = float(np.clip((nearest.mean() - args.close_contact_distance_A) / (args.open_channel_distance_A - args.close_contact_distance_A), 0.0, 1.0))
    open_z_score = float(np.clip(nearest_z.mean() / args.open_channel_z_A, 0.0, 1.0))
    low_coord_score = float(np.clip(1.0 - coord.mean() / args.deep_coordination_scale, 0.0, 1.0))
    gap_score = float(np.clip(0.45 * open_distance_score + 0.35 * open_z_score + 0.20 * low_coord_score, 0.0, 1.0))

    feature_summary = {
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "cell_source": str(cell_source) if cell_source else None,
        "cell_source_sha256": sha256_file(cell_source) if cell_source else None,
        "pbc_applied": cell is not None,
        "n_images": len(images),
        "n_atoms": len(images[0]),
        "migrant_element": args.migrant_element,
        "migrant_index_zero_based": migrant_index,
        "path_length_A": float(segment_lengths.sum()),
        "endpoint_distance_A": float(np.linalg.norm(endpoint_vector)),
        "min_host_distance_A": float(nearest.min()),
        "mean_host_distance_A": float(nearest.mean()),
        "mean_coordination": float(coord.mean()),
        "max_coordination": int(coord.max()),
        "mean_nearest_host_plane_dz_A": float(nearest_z.mean()),
        "mean_z_density_score": float(z_density.mean()),
        "penetration_score": penetration_score,
        "gap_score": gap_score,
        "descriptor_notes": [
            "penetration_score combines coordination, z-plane density, and closest contact.",
            "gap_score combines open nearest-neighbor distance, separation from host z-planes, and low coordination.",
            "Use continuous scores in the dataset; class_label is a documented bin, not a hidden hand label.",
        ],
        "per_image": rows,
    }
    label, confidence = classify_from_features(feature_summary)
    feature_summary["class_label"] = label
    feature_summary["class_confidence"] = round(float(confidence), 3)

    neb_out = path.with_name("neb.out")
    feature_summary["historical_qe"] = parse_neb_tail(neb_out)
    return feature_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="ASE-readable NEB image paths.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--migrant-element", default="Cr")
    parser.add_argument("--cell-input", type=Path, help="QE input providing a shared CELL_PARAMETERS angstrom block.")
    parser.add_argument("--coord-cutoff-A", type=float, default=3.2)
    parser.add_argument("--z-density-sigma-A", type=float, default=0.8)
    parser.add_argument("--deep-coordination-scale", type=float, default=5.0)
    parser.add_argument("--deep-density-scale", type=float, default=0.16)
    parser.add_argument("--close-contact-distance-A", type=float, default=2.0)
    parser.add_argument("--open-channel-distance-A", type=float, default=3.0)
    parser.add_argument("--open-channel-z-A", type=float, default=1.6)
    args = parser.parse_args()

    results = [analyze_path(path.resolve(), args) for path in args.paths]
    parameters = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "paths"
    }
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": parameters | {"paths": [str(p.resolve()) for p in args.paths]},
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")

    fields = [
        "path_id",
        "class_label",
        "class_confidence",
        "penetration_score",
        "gap_score",
        "n_images",
        "n_atoms",
        "path_length_A",
        "endpoint_distance_A",
        "min_host_distance_A",
        "mean_host_distance_A",
        "mean_coordination",
        "max_coordination",
        "mean_nearest_host_plane_dz_A",
        "mean_z_density_score",
        "activation_forward_eV",
        "activation_reverse_eV",
        "job_done",
        "warning_count",
        "max_reported_image_error_eV_A",
        "source_path",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in results:
            hist = row.get("historical_qe", {})
            flat = {
                "path_id": Path(row["source_path"]).parent.name,
                **{key: row.get(key) for key in fields if key in row},
                "activation_forward_eV": hist.get("activation_forward_eV"),
                "activation_reverse_eV": hist.get("activation_reverse_eV"),
                "job_done": hist.get("job_done"),
                "warning_count": hist.get("warning_count"),
                "max_reported_image_error_eV_A": hist.get("max_reported_image_error_eV_A"),
            }
            writer.writerow(flat)
    print(json.dumps({"wrote_json": str(args.output_json), "wrote_csv": str(args.output_csv), "n_paths": len(results)}, indent=2))


if __name__ == "__main__":
    main()
