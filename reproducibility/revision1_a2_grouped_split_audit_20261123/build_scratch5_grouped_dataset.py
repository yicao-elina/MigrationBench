#!/usr/bin/env python3
"""Build an auditable Scratch-5% control from the fixed grouped-split data.

The default interpretation is 5% of the grouped train configurations, while
valid/test remain unchanged.  It never samples from valid/test and writes a
manifest containing the selected train indices and file hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import read, write


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True, help="data_grouped directory")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--selection-seed", type=int, default=20260927)
    args = ap.parse_args()
    if not 0 < args.fraction < 1:
        raise SystemExit("fraction must be between 0 and 1")
    train = read(args.source / "train.xyz", index=":")
    valid = read(args.source / "valid.xyz", index=":")
    test = read(args.source / "test.xyz", index=":")
    n_keep = max(1, round(len(train) * args.fraction))
    rng = np.random.default_rng(args.selection_seed)
    selected = np.sort(rng.choice(len(train), size=n_keep, replace=False))
    trajectory_counts: dict[str, int] = {}
    for i in selected:
        trajectory = str(train[i].info.get("source_trajectory", "UNKNOWN"))
        trajectory_counts[trajectory] = trajectory_counts.get(trajectory, 0) + 1
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "train.xyz", [train[i] for i in selected], format="extxyz")
    write(args.output / "valid.xyz", valid, format="extxyz")
    write(args.output / "test.xyz", test, format="extxyz")
    manifest = {
        "definition": "5% of grouped train configurations; valid/test unchanged",
        "source_dir": str(args.source),
        "fraction": args.fraction,
        "selection_seed": args.selection_seed,
        "source_counts": {"train": len(train), "valid": len(valid), "test": len(test)},
        "output_counts": {"train": n_keep, "valid": len(valid), "test": len(test)},
        "selected_train_indices": selected.tolist(),
        "selected_frames_per_source_trajectory": dict(sorted(trajectory_counts.items())),
    }
    for name in ("train", "valid", "test"):
        manifest.setdefault("sha256", {})[name] = sha256(args.output / f"{name}.xyz")
    (args.output / "scratch5_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
