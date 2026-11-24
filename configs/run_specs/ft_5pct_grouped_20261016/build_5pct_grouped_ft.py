#!/usr/bin/env python3
"""Build a deterministic, trajectory-stratified 5% FT release."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.io import read, write


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def allocate_quotas(groups: dict[str, list[int]], target: int) -> dict[str, int]:
    names = sorted(groups)
    total = sum(len(groups[name]) for name in names)
    raw = {name: len(groups[name]) * target / total for name in names}
    quota = {name: max(1, int(np.floor(raw[name]))) for name in names}
    while sum(quota.values()) > target:
        candidates = [name for name in names if quota[name] > 1]
        name = min(candidates, key=lambda item: (raw[item] - quota[item], item))
        quota[name] -= 1
    order = sorted(names, key=lambda item: (-(raw[item] - np.floor(raw[item])), item))
    for name in order:
        if sum(quota.values()) >= target:
            break
        quota[name] += 1
    return quota


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--selection-seed", type=int, default=20261016)
    args = parser.parse_args()
    if not 0 < args.fraction < 1:
        raise SystemExit("fraction must be between 0 and 1")

    train = read(args.source / "train.xyz", index=":")
    valid = read(args.source / "valid.xyz", index=":")
    test = read(args.source / "test.xyz", index=":")
    periodic = [i for i, atoms in enumerate(train) if len(atoms) > 1]
    isolated = [i for i, atoms in enumerate(train) if len(atoms) == 1]
    groups: dict[str, list[int]] = defaultdict(list)
    for index in periodic:
        groups[str(train[index].info.get("source_trajectory", "UNKNOWN"))].append(index)
    if "UNKNOWN" in groups:
        raise SystemExit("periodic frame without source_trajectory")

    target = max(len(groups), round(len(periodic) * args.fraction))
    quotas = allocate_quotas(groups, target)
    rng = np.random.default_rng(args.selection_seed)
    selected_periodic: list[int] = []
    selected_per_group: dict[str, list[int]] = {}
    for name in sorted(groups):
        candidates = np.asarray(groups[name], dtype=int)
        picked = np.sort(rng.choice(candidates, size=quotas[name], replace=False))
        selected_per_group[name] = picked.tolist()
        selected_periodic.extend(picked.tolist())
    selected = sorted(selected_periodic + isolated)

    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "train.xyz", [train[i] for i in selected], format="extxyz")
    write(args.output / "valid.xyz", valid, format="extxyz")
    write(args.output / "test.xyz", test, format="extxyz")
    manifest = {
        "schema": "migrationbench_ft_5pct_grouped_dataset_v1",
        "family": args.family,
        "definition": "5% of periodic grouped-train configurations; all isolated E0 anchors retained; valid/test unchanged",
        "source_dir": str(args.source),
        "fraction": args.fraction,
        "selection_seed": args.selection_seed,
        "selection_method": "trajectory-stratified random without replacement; minimum one periodic frame per train trajectory",
        "source_counts": {"train": len(train), "periodic_train": len(periodic), "isolated_train": len(isolated), "valid": len(valid), "test": len(test)},
        "output_counts": {"train": len(selected), "periodic_train": len(selected_periodic), "isolated_train": len(isolated), "valid": len(valid), "test": len(test)},
        "selected_periodic_train_indices": selected_periodic,
        "selected_train_indices_including_isolated": selected,
        "selected_frames_per_source_trajectory": {name: len(selected_per_group[name]) for name in sorted(selected_per_group)},
        "source_sha256": {name: sha256(args.source / f"{name}.xyz") for name in ("train", "valid", "test")},
        "output_sha256": {name: sha256(args.output / f"{name}.xyz") for name in ("train", "valid", "test")},
    }
    (args.output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
