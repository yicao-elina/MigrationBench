#!/usr/bin/env python3
"""Export the Cr81 fixed-path NEB images (real coordinates) for the lattice renders.

Reads the three canonical extxyz paths used by the NEB r01 matrix and writes
public/data/geometry.json with every image's atom positions (Å). The video
projects them to a side view, so Cr moves along its actual computed path.

Usage:
    python3 scripts/export_geometry.py [--root PATH]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from export_neb_profiles import DEFAULT_ROOT

INPUT_DIR = "analysis/task3_endpoint_only_neb/neb_fixed_vs_self_matrix_20261019/inputs/fixed_r01"
PATHS = {
    "within_gap": "cr_canonical_in_gap.extxyz",
    "gap_to_ql": "cr_canonical_gap_to_ql.extxyz",
    "within_ql": "cr_canonical_within_ql.extxyz",
}
OUT_PATH = Path(__file__).resolve().parent.parent / "public/data/geometry.json"


def read_extxyz(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    frames, i = [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        header = lines[i + 1]
        lattice = [float(v) for v in header.split('Lattice="')[1].split('"')[0].split()]
        atoms = []
        for row in lines[i + 2 : i + 2 + n]:
            sym, x, y, z = row.split()[:4]
            atoms.append([sym, round(float(x), 3), round(float(y), 3), round(float(z), 3)])
        frames.append({"lattice": lattice, "atoms": atoms})
        i += n + 2
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    out = {"source_dir": str(args.root / INPUT_DIR), "paths": {}}
    for key, name in PATHS.items():
        path = args.root / INPUT_DIR / name
        if not path.is_file():
            sys.exit(f"Missing geometry: {path}")
        frames = read_extxyz(path)
        out["paths"][key] = {
            "file": name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "lattice": frames[0]["lattice"],
            "species": [a[0] for a in frames[0]["atoms"]],
            "images": [[a[1:] for a in f["atoms"]] for f in frames],
        }
        cr = out["paths"][key]["species"].index("Cr")
        start, end = frames[0]["atoms"][cr][1:], frames[-1]["atoms"][cr][1:]
        print(key, len(frames), "images; Cr", start, "->", end)
    OUT_PATH.write_text(json.dumps(out))
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
