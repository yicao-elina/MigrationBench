#!/usr/bin/env python3
"""Audit extracted periodic extxyz snapshots before any DFT submission."""

import argparse
import csv
import json
import math
import re
from pathlib import Path


STEP_RE = re.compile(r"(?:^|\s)step=(-?\d+)(?:\s|$)")
LATTICE_RE = re.compile(r'Lattice="([^"]+)"')
PBC_RE = re.compile(r'pbc="([^"]+)"')
STRESS_RE = re.compile(r'stress="([^"]+)"')
PROPS_RE = re.compile(r"Properties=([^\s]+)")


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return float("nan")
    position = (len(values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def read_frame(path):
    with path.open(encoding="utf-8") as handle:
        natoms = int(handle.readline().strip())
        comment = handle.readline().strip()
        rows = [handle.readline().split() for _ in range(natoms)]
    match = PROPS_RE.search(comment)
    if not match or "forces:R:3" not in match.group(1):
        raise ValueError("forces:R:3 property is missing")
    has_momenta = "momenta:R:3" in match.group(1)
    species = []
    forces = []
    positions = []
    momenta = []
    for row in rows:
        species.append(row[0])
        positions.append([float(x) for x in row[1:4]])
        offset = 4
        if has_momenta:
            momenta.append([float(x) for x in row[offset:offset + 3]])
            offset += 3
        forces.append([float(x) for x in row[offset:offset + 3]])
    norms = [math.sqrt(sum(component * component for component in vector)) for vector in forces]
    key_indices = [i for i, symbol in enumerate(species) if symbol == "Cr"]
    result = {
        "file": str(path),
        "natoms": natoms,
        "step": int(STEP_RE.search(comment).group(1)) if STEP_RE.search(comment) else None,
        "pbc": PBC_RE.search(comment).group(1) if PBC_RE.search(comment) else None,
        "lattice": [float(x) for x in LATTICE_RE.search(comment).group(1).split()] if LATTICE_RE.search(comment) else None,
        "stress": [float(x) for x in STRESS_RE.search(comment).group(1).split()] if STRESS_RE.search(comment) else None,
        "properties": match.group(1),
        "species_counts": {symbol: species.count(symbol) for symbol in sorted(set(species))},
        "finite_positions": all(math.isfinite(x) for vector in positions for x in vector),
        "finite_forces": all(math.isfinite(x) for vector in forces for x in vector),
        "finite_momenta": all(math.isfinite(x) for vector in momenta for x in vector) if has_momenta else False,
        "force_max_eV_A": max(norms),
        "force_p50_eV_A": percentile(norms, 0.50),
        "force_p95_eV_A": percentile(norms, 0.95),
        "force_p99_eV_A": percentile(norms, 0.99),
        "cr_force_max_eV_A": max((norms[i] for i in key_indices), default=float("nan")),
        "cr_indices_1based": [i + 1 for i in key_indices],
        "comment": comment,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    rows = [read_frame(path) for path in sorted(args.input_dir.glob("snapshot_step_*.extxyz"))]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    fields = [
        "file", "step", "natoms", "pbc", "species_counts", "finite_positions",
        "finite_forces", "finite_momenta", "force_max_eV_A", "force_p50_eV_A",
        "force_p95_eV_A", "force_p99_eV_A", "cr_force_max_eV_A", "properties",
    ]
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {key: row.get(key) for key in fields}
            record["species_counts"] = json.dumps(record["species_counts"], sort_keys=True)
            writer.writerow(record)


if __name__ == "__main__":
    main()
