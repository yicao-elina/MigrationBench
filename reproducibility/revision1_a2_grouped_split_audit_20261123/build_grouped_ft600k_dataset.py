#!/usr/bin/env python3
"""Build an FT-600K dataset without splitting a source trajectory.

This keeps the historical 600 K preprocessing rule (discard 200 frames,
sample every fifth frame, cap each trajectory at 500 frames), but replaces the
old global frame-level shuffle with an explicit source-file manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ase.io import read, write


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--discard", type=int, default=200)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=500)
    p.add_argument("--isolated-file", type=Path, required=True)
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    groups = manifest["groups"]
    allowed = {"train", "valid", "test", "skip"}
    splits = {g["split"] for g in groups}
    unknown = splits - allowed
    if unknown:
        raise SystemExit(f"Unknown split labels: {sorted(unknown)}")
    if not {"train", "valid", "test"}.issubset(splits):
        raise SystemExit("Manifest must assign at least one group to train, valid, and test")

    seen = set()
    by_split = {name: [] for name in ("train", "valid", "test")}
    summary = {name: {"groups": [], "periodic_frames": 0} for name in by_split}

    for group in groups:
        name = group["name"]
        source = Path(group["source"])
        split = group["split"]
        if name in seen:
            raise SystemExit(f"Duplicate group name: {name}")
        seen.add(name)
        if split == "skip":
            continue
        if not source.exists():
            raise SystemExit(f"Missing source file: {source}")
        frames = read(source, index=":")
        total = len(frames)
        sampled = frames[args.discard :: args.stride]
        sampled = sampled[: args.max_frames]
        for i, atoms in enumerate(sampled):
            atoms.info["source_trajectory"] = name
            atoms.info["source_file"] = str(source)
            atoms.info["source_frame_after_equilibration"] = args.discard + i * args.stride
            atoms.info["grouped_split"] = split
        by_split[split].extend(sampled)
        summary[split]["groups"].append(
            {"name": name, "source": str(source), "original_frames": total,
             "sampled_frames": len(sampled), "sha256": sha256(source)}
        )
        summary[split]["periodic_frames"] += len(sampled)

    isolated = read(args.isolated_file, index=":")
    for atoms in isolated:
        atoms.info["source_trajectory"] = "isolated_atoms_SG15"
        atoms.info["source_file"] = str(args.isolated_file)
        atoms.info["grouped_split"] = "train"
    by_split["train"].extend(isolated)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        write(args.output_dir / f"{split}.xyz", by_split[split], format="extxyz")
    summary["parameters"] = {"discard": args.discard, "stride": args.stride,
                              "max_frames": args.max_frames,
                              "isolated_file": str(args.isolated_file),
                              "isolated_frames_added_to_train": len(isolated)}
    summary["output_sha256"] = {
        split: sha256(args.output_dir / f"{split}.xyz")
        for split in ("train", "valid", "test")
    }
    (args.output_dir / "grouped_split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
