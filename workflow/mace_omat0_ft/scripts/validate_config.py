#!/usr/bin/env python3
"""Validate a configuration-driven OMAT-0 FT campaign without submitting jobs."""
import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"CONFIG_ERROR: {message}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--stage", choices=["train", "md", "neb", "all"], default="train")
    ap.add_argument("--check-files", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    for key in ("campaign_id", "remote_root", "foundation_models", "selected_foundation_model", "datasets", "families", "training", "md", "neb", "analysis", "release"):
        require(key in cfg, f"missing top-level key: {key}")
    selected = cfg["foundation_models"].get(cfg["selected_foundation_model"])
    require(selected is not None, "selected_foundation_model is not defined")
    require(selected["role"] == "public_zero_shot_foundation", "selected foundation must be explicitly tagged public_zero_shot_foundation")
    require("legacy_compiled" not in selected["source"], "legacy compiled checkpoint cannot be selected")

    seeds = [int(x) for x in cfg["training"]["seeds"]]
    require(len(seeds) == len(set(seeds)), "training seeds must be unique")
    require(len(seeds) == 3, "this campaign requires exactly three training seeds")
    require(set(seeds) == {123, 234, 345}, "training seeds must be 123, 234, 345")
    require(len(cfg["families"]) == 2, "campaign must contain exactly two FT families")
    for family_id, family in cfg["families"].items():
        require(family["dataset"] in cfg["datasets"], f"{family_id}: unknown dataset {family['dataset']}")
        require(bool(family.get("multiheads_finetuning")), f"{family_id}: multiheads_finetuning must be explicit")

    if args.stage in ("md", "all"):
        require(cfg["md"].get("enabled") is True, "MD stage is not enabled")
        require(set(cfg["md"]["md_velocity_seed_by_training_seed"]) == {str(s) for s in seeds}, "MD seed map must cover all training seeds")
        require(len(set(cfg["md"]["md_velocity_seed_by_training_seed"].values())) == len(seeds), "MD velocity seeds must be distinct")
    if args.stage in ("neb", "all"):
        require(cfg["neb"].get("enabled") is True, "NEB stage is intentionally gated off until an input manifest is supplied")
        require(cfg["neb"].get("input_manifest"), "NEB input_manifest is required")

    checked = {"model": None, "datasets": {}}
    if args.check_files:
        model_path = Path(selected["source"])
        require(model_path.is_file(), f"missing foundation model: {model_path}")
        actual = sha256(model_path)
        require(actual == selected["sha256"], f"foundation SHA256 mismatch: expected {selected['sha256']} got {actual}")
        checked["model"] = {"path": str(model_path), "sha256": actual}
        for dataset_id, dataset in cfg["datasets"].items():
            root = Path(dataset["root"])
            require(root.is_dir(), f"missing dataset root: {root}")
            rows = {}
            for name, expected in dataset["files"].items():
                path = root / name
                require(path.is_file() and path.stat().st_size > 0, f"missing dataset file: {path}")
                actual = sha256(path)
                require(actual == expected, f"dataset SHA256 mismatch for {path}: expected {expected} got {actual}")
                rows[name] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
            checked["datasets"][dataset_id] = rows

    jobs = [{"family": family_id, "training_seed": seed} for family_id in cfg["families"] for seed in seeds]
    payload = {"campaign_id": cfg["campaign_id"], "selected_foundation": cfg["selected_foundation_model"], "jobs": jobs, "checked": checked}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
