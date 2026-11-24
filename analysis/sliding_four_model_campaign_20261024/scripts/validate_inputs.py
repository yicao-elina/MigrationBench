#!/usr/bin/env python3
"""Validate the recovered fixed-path inputs and emit hash-bound M0 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import file_record, load_config, root_path, sha256_file, write_json, write_tsv


def read_xyz(path: Path) -> list[tuple[list[str], np.ndarray]]:
    lines = path.read_text().splitlines()
    frames = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        count = int(lines[cursor].strip())
        cursor += 2
        symbols = []
        coords = []
        for _ in range(count):
            fields = lines[cursor].split()
            symbols.append(fields[0])
            coords.append([float(value) for value in fields[1:4]])
            cursor += 1
        frames.append((symbols, np.asarray(coords, dtype=float)))
    return frames


def parse_latest_neb_energies(path: Path) -> list[float]:
    text = path.read_text()
    blocks = re.split(r"image\s+energy\s+\(eV\)\s+error\s+\(eV/A\)\s+frozen", text)
    parsed = []
    for block in blocks[1:]:
        values = []
        for line in block.splitlines():
            fields = line.split()
            if len(fields) < 4:
                if values:
                    break
                continue
            try:
                int(fields[0])
                values.append(float(fields[1]))
            except ValueError:
                if values:
                    break
        if values:
            parsed.append(values)
    if not parsed:
        raise ValueError(f"no NEB energy block in {path}")
    return parsed[-1]


def read_dft_csv(path: Path, column: str) -> tuple[list[float], list[float]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    reaction = [float(row["Reaction_Coord"]) for row in rows]
    values = [float(row[column]) for row in rows]
    return reaction, values


def canonical_endpoint_hash(symbols: list[str], coords: np.ndarray) -> str:
    payload = {
        "symbols": symbols,
        "coords": np.asarray(coords, dtype=float).round(10).tolist(),
    }
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config, root = load_config(args.config)
    checks = []
    hash_rows = []
    all_pass = True
    for system_id, spec in config["systems"].items():
        xyz = root_path(root, spec["xyz"])
        dft_csv = root_path(root, spec["dft_csv"])
        neb_out = root_path(root, spec["dft_neb_out"])
        reaction, csv_values = read_dft_csv(dft_csv, spec["dft_column"])
        expected = np.asarray(spec["expected_dft_relative_eV"], dtype=float)
        csv_values_array = np.asarray(csv_values, dtype=float)
        neb_values = np.asarray(parse_latest_neb_energies(neb_out), dtype=float)
        neb_relative = neb_values - neb_values[0]
        frames = read_xyz(xyz)
        frame_counts = [len(coords) for _, coords in frames]
        csv_delta = float(np.max(np.abs(csv_values_array - expected))) if len(csv_values_array) == len(expected) else float("inf")
        neb_delta = float(np.max(np.abs(neb_relative - expected))) if len(neb_relative) == len(expected) else float("inf")
        shape_pass = len(frames) == config["protocol"]["fixed_images"] and all(count == spec["atoms"] for count in frame_counts)
        csv_pass = len(csv_values_array) == len(expected) and np.isfinite(csv_values_array).all() and csv_delta < 1e-6
        neb_pass = len(neb_relative) == len(expected) and np.isfinite(neb_relative).all() and neb_delta < 1e-4
        system_pass = bool(shape_pass and csv_pass and neb_pass)
        all_pass = all_pass and system_pass
        checks.append({
            "system_id": system_id,
            "status": "PASS" if system_pass else "FAIL",
            "frames": len(frames),
            "frame_atom_counts": frame_counts,
            "expected_atoms": spec["atoms"],
            "csv_max_abs_delta_eV": csv_delta,
            "neb_max_abs_delta_eV": neb_delta,
            "csv_reference_status": config["protocol"]["dft_reference_status"],
            "endpoint_hash_initial": canonical_endpoint_hash(*frames[0]),
            "endpoint_hash_final": canonical_endpoint_hash(*frames[-1]),
        })
        for role, path in (
            ("xyz_10_frame_path", xyz),
            ("dft_energy_csv", dft_csv),
            ("dft_neb_out", neb_out),
            ("dft_neb_in", root_path(root, spec["dft_neb_in"])),
            ("dft_pw_1", root_path(root, spec["dft_pw_1"])),
            ("dft_pw_2", root_path(root, spec["dft_pw_2"])),
        ):
            record = file_record(path, root)
            record.update({"system_id": system_id, "role": role})
            hash_rows.append(record)
    payload = {
        "schema_version": "sliding_m0_input_validation_v1",
        "campaign_id": config["campaign_id"],
        "status": "PASS" if all_pass else "FAIL",
        "checks": checks,
    }
    write_json(root / "manifests/M0_INPUT_VALIDATION.json", payload)
    fields = ["system_id", "role", "relative_path", "path", "exists", "bytes", "sha256"]
    write_tsv(root / "manifests/INPUT_HASHES.tsv", hash_rows, fields)
    print(json.dumps(payload, indent=2))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

