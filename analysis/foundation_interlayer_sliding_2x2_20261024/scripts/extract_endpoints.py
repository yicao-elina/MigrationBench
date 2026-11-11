#!/usr/bin/env python3
"""Extract the two endpoint images from the recovered b-path2 XYZ files.

The recovered XYZ files intentionally remain unchanged.  The QE cell is read
from the matching pw_1.in and attached only to the endpoint files used by the
self-relaxed calculation, so the self branch has explicit periodic geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cell(pw_path: Path) -> list[list[float]]:
    lines = pw_path.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("CELL_PARAMETERS"):
            vectors = []
            for row in lines[i + 1 : i + 4]:
                values = row.split()
                if len(values) < 3:
                    raise ValueError(f"Malformed CELL_PARAMETERS row in {pw_path}")
                vectors.append([float(values[0]), float(values[1]), float(values[2])])
            if len(vectors) != 3:
                raise ValueError(f"Missing complete cell in {pw_path}")
            return vectors
    raise ValueError(f"CELL_PARAMETERS not found in {pw_path}")


def read_xyz_frames(xyz_path: Path) -> list[list[tuple[str, float, float, float]]]:
    """Parse this legacy multi-frame XYZ without importing ASE."""
    lines = xyz_path.read_text().splitlines()
    frames = []
    cursor = 0
    while cursor < len(lines):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            break
        try:
            count = int(lines[cursor].strip())
        except ValueError as exc:
            raise ValueError(f"Expected XYZ atom count at line {cursor + 1}") from exc
        cursor += 1
        if cursor >= len(lines):
            raise ValueError("XYZ frame has no comment line")
        cursor += 1
        atoms = []
        for _ in range(count):
            fields = lines[cursor].split()
            if len(fields) < 4:
                raise ValueError(f"Malformed XYZ atom line at line {cursor + 1}")
            atoms.append((fields[0], float(fields[1]), float(fields[2]), float(fields[3])))
            cursor += 1
        frames.append(atoms)
    return frames


def write_extxyz(path: Path, atoms: list[tuple[str, float, float, float]], cell: list[list[float]]) -> None:
    lattice = " ".join(f"{value:.15g}" for row in cell for value in row)
    with path.open("w") as handle:
        handle.write(f"{len(atoms)}\n")
        handle.write(
            f'Properties=species:S:1:pos:R:3 Lattice="{lattice}" pbc="T T T"\n'
        )
        for symbol, x, y, z in atoms:
            handle.write(f"{symbol} {x:.15g} {y:.15g} {z:.15g}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--pw", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    frames = read_xyz_frames(args.xyz)
    if len(frames) != 10:
        raise ValueError(f"Expected 10 frames in {args.xyz}, found {len(frames)}")
    if len(frames[0]) not in (80, 81):
        raise ValueError(f"Unexpected atom count {len(frames[0])} in {args.xyz}")
    if any(len(frame) != len(frames[0]) for frame in frames):
        raise ValueError(f"Atom count changes across frames in {args.xyz}")

    cell = read_cell(args.pw)
    args.outdir.mkdir(parents=True, exist_ok=True)
    endpoints = {"initial": frames[0], "final": frames[-1]}
    for label, atoms in endpoints.items():
        write_extxyz(args.outdir / f"{label}.extxyz", atoms, cell)

    manifest = {
        "system": args.system,
        "source_xyz": str(args.xyz),
        "source_xyz_sha256": sha256(args.xyz),
        "source_pw_1": str(args.pw),
        "source_pw_1_sha256": sha256(args.pw),
        "frame_count": len(frames),
        "atom_count": len(frames[0]),
        "symbols": [atom[0] for atom in frames[0]],
        "cell_angstrom": cell,
        "pbc": [True, True, True],
        "endpoint_files": {
            label: str(args.outdir / f"{label}.extxyz") for label in endpoints
        },
    }
    (args.outdir / "endpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
