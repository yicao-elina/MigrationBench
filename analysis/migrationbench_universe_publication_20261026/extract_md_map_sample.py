#!/usr/bin/env python3
"""Extract a small, deterministic MD frame sample from remote EXTXYZ files.

The full trajectories remain on the source host.  This helper copies only
uniformly spaced frames needed for the local Cr-centred SOAP universe map and
records source paths, frame indices, sizes, and SHA256 values in a manifest.
It uses only the Python standard library so it can run on the Rockfish login
environment without depending on the local analysis environment.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def sha256_and_count(path: Path) -> Tuple[str, int, int]:
    digest = hashlib.sha256()
    frames = 0
    natoms_seen = 0
    with path.open("rb") as handle:
        while True:
            first = handle.readline()
            if not first:
                break
            digest.update(first)
            natoms = int(first.strip())
            natoms_seen = natoms if natoms_seen == 0 else natoms_seen
            header = handle.readline()
            if not header:
                raise ValueError(f"truncated EXTXYZ header in {path} frame {frames}")
            digest.update(header)
            for _ in range(natoms):
                row = handle.readline()
                if not row:
                    raise ValueError(f"truncated EXTXYZ atoms in {path} frame {frames}")
                digest.update(row)
            frames += 1
    return digest.hexdigest(), frames, natoms_seen


def selected_indices(nframes: int, keep: int) -> List[int]:
    if nframes <= 0:
        return []
    if nframes <= keep:
        return list(range(nframes))
    if keep == 1:
        return [0]
    values = [round(i * (nframes - 1) / (keep - 1)) for i in range(keep)]
    return sorted(set(int(x) for x in values))


def copy_selected(path: Path, indices: Set[int], output, metadata: Dict[str, str], records: List[dict]) -> None:
    with path.open("rb") as handle:
        frame = 0
        while True:
            first = handle.readline()
            if not first:
                break
            natoms = int(first.strip())
            header = handle.readline()
            atoms = [handle.readline() for _ in range(natoms)]
            if frame in indices:
                output.write(first)
                output.write(header)
                output.writelines(atoms)
                records.append(
                    {
                        "sample_frame": len(records),
                        "model_id": metadata["model_id"],
                        "family": metadata["family"],
                        "source_path": str(path),
                        "source_sha256": metadata["source_sha256"],
                        "source_frame": frame,
                        "natoms": natoms,
                    }
                )
            frame += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--keep", type=int, default=5)
    args = parser.parse_args()

    rows = []
    with args.inputs.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        header = [name.lstrip("# ").strip() for name in header]
        for values in reader:
            if not values or not values[0].strip():
                continue
            rows.append(dict(zip(header, values)))

    selected_rows = []
    seen = set()
    for row in rows:
        model_id = row["model_id"]
        if model_id in seen:
            continue
        seen.add(model_id)
        selected_rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    source_records = []
    with args.output.open("wb") as output:
        for row in selected_rows:
            path = Path(row["trajectory_path"])
            if not path.exists():
                source_records.append({**row, "status": "missing"})
                continue
            source_sha256, nframes, natoms = sha256_and_count(path)
            keep_indices = selected_indices(nframes, args.keep)
            meta = {
                "model_id": row["model_id"],
                "family": row["family"],
                "source_sha256": source_sha256,
            }
            before = len(records)
            copy_selected(path, set(keep_indices), output, meta, records)
            source_records.append(
                {
                    **row,
                    "status": "sampled",
                    "source_sha256": source_sha256,
                    "nframes": nframes,
                    "natoms": natoms,
                    "selected_frames": keep_indices,
                    "sample_count": len(records) - before,
                }
            )

    payload = {
        "keep_per_model": args.keep,
        "sources": source_records,
        "frames": records,
    }
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"sources": len(source_records), "frames": len(records), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
