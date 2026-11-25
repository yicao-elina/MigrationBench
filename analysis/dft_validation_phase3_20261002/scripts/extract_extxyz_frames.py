#!/usr/bin/env python3
"""Extract exact step-labelled extxyz frames without altering the source file."""

import argparse
import hashlib
import json
import re
from pathlib import Path


STEP_RE = re.compile(r"(?:^|\s)step=(-?\d+)(?:\s|$)")
CELL_RE = re.compile(r'Lattice="([^"]+)"')
PBC_RE = re.compile(r'pbc="([^"]+)"')
PROP_RE = re.compile(r"Properties=([^\s]+)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    args = parser.parse_args()

    targets = set(args.steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    found = {}
    natoms_seen = set()
    frame_index = 0

    with args.input.open("r", encoding="utf-8", errors="replace") as source:
        while True:
            first = source.readline()
            if not first:
                break
            first = first.rstrip("\n")
            if not first.strip().isdigit():
                raise ValueError(f"frame {frame_index}: invalid atom count line: {first!r}")
            natoms = int(first.strip())
            comment = source.readline().rstrip("\n")
            if not comment:
                raise ValueError(f"frame {frame_index}: missing comment line")
            atom_lines = [source.readline() for _ in range(natoms)]
            if any(line == "" for line in atom_lines):
                raise ValueError(f"frame {frame_index}: truncated atom block")

            natoms_seen.add(natoms)
            match = STEP_RE.search(comment)
            if not match:
                raise ValueError(f"frame {frame_index}: no exact step metadata in comment")
            step = int(match.group(1))
            if step in targets and step not in found:
                output = args.output_dir / f"snapshot_step_{step:06d}.extxyz"
                with output.open("w", encoding="utf-8") as handle:
                    handle.write(f"{natoms}\n{comment}\n")
                    handle.writelines(atom_lines)
                properties = PROP_RE.search(comment)
                lattice = CELL_RE.search(comment)
                pbc = PBC_RE.search(comment)
                found[step] = {
                    "step": step,
                    "frame_index": frame_index,
                    "natoms": natoms,
                    "output": str(output),
                    "output_sha256": sha256(output),
                    "has_momenta_property": bool(properties and "momenta" in properties.group(1)),
                    "properties_token": properties.group(1) if properties else None,
                    "lattice_token_present": bool(lattice),
                    "pbc_token": pbc.group(1) if pbc else None,
                    "comment": comment,
                }
            frame_index += 1
            if targets.issubset(found):
                break

    manifest = {
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "requested_steps": sorted(targets),
        "found_steps": sorted(found),
        "missing_steps": sorted(targets - set(found)),
        "frames_scanned": frame_index,
        "natoms_seen": sorted(natoms_seen),
        "snapshots": [found[step] for step in sorted(found)],
    }
    with (args.output_dir / "extraction_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if targets - set(found):
        raise SystemExit(f"missing requested steps: {sorted(targets - set(found))}")


if __name__ == "__main__":
    main()
