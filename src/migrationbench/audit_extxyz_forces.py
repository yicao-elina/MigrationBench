#!/usr/bin/env python3
"""Audit extxyz force columns without relying on ASE result mapping."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np


PROPERTIES_RE = re.compile(r'Properties=("[^"]+"|\S+)')


def parse_properties(header: str) -> tuple[int | None, int | None]:
    match = PROPERTIES_RE.search(header)
    if not match:
        return None, None
    raw = match.group(1).strip('"')
    fields = raw.split(":")
    column = 0
    force_start = None
    force_count = None
    i = 0
    while i + 2 < len(fields):
        name = fields[i]
        try:
            count = int(fields[i + 2])
        except ValueError:
            return None, None
        if name == "forces":
            force_start = column
            force_count = count
            break
        column += count
        i += 3
    return force_start, force_count


def audit_file(path: Path) -> dict[str, object]:
    frames = 0
    frames_with_force_columns = 0
    all_zero_force_frames = 0
    max_abs_force = 0.0
    has_neb_tags = False
    natoms_min = None
    natoms_max = None
    info_keys_seen: set[str] = set()

    with path.open(errors="replace") as handle:
        while True:
            natoms_line = handle.readline()
            if not natoms_line:
                break
            natoms_line = natoms_line.strip()
            if not natoms_line:
                continue
            try:
                natoms = int(natoms_line)
            except ValueError as exc:
                raise ValueError(f"{path}: expected atom count, got {natoms_line!r}") from exc
            header = handle.readline()
            if not header:
                raise ValueError(f"{path}: missing header after frame {frames}")

            frames += 1
            natoms_min = natoms if natoms_min is None else min(natoms_min, natoms)
            natoms_max = natoms if natoms_max is None else max(natoms_max, natoms)
            for key in ("path_name", "path_id", "neb_image", "neb_protocol", "source_neb"):
                if f"{key}=" in header:
                    has_neb_tags = True
            for key in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=", header):
                info_keys_seen.add(key)

            force_start, force_count = parse_properties(header)
            force_values = []
            for _ in range(natoms):
                row = handle.readline()
                if not row:
                    raise ValueError(f"{path}: truncated atom block in frame {frames}")
                parts = row.split()
                if force_start is not None and force_count is not None:
                    force_values.extend(float(x) for x in parts[force_start : force_start + force_count])

            if force_start is not None and force_count:
                frames_with_force_columns += 1
                arr = np.asarray(force_values, dtype=float)
                if arr.size:
                    max_abs_force = max(max_abs_force, float(np.max(np.abs(arr))))
                    if np.allclose(arr, 0.0):
                        all_zero_force_frames += 1
                else:
                    all_zero_force_frames += 1

    return {
        "path": str(path),
        "frames": frames,
        "frames_with_force_columns": frames_with_force_columns,
        "all_zero_force_frames": all_zero_force_frames,
        "max_abs_force_eV_A": max_abs_force,
        "has_neb_tags": has_neb_tags,
        "info_keys_sample": "|".join(sorted(info_keys_seen)[:20]),
        "natoms_min": natoms_min,
        "natoms_max": natoms_max,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [audit_file(path) for path in args.paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
