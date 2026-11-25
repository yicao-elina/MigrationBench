#!/usr/bin/env python3
"""Validate the additive three-foundation seed-123 campaign without submitting jobs."""
import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"CONFIG_ERROR: {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--families", help="comma-separated family IDs to validate/report")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    required = ("campaign_id", "remote_root", "foundation_models", "datasets", "families", "training", "md", "neb", "analysis", "release")
    for key in required:
        require(key in cfg, f"missing top-level key: {key}")

    seeds = [int(value) for value in cfg["training"]["seeds"]]
    require(seeds == [123], "this additive campaign is fixed to training seed 123")
    require(set(cfg["foundation_models"]) == {"mpa0", "mp0b3", "polar1m"}, "foundation registry must contain exactly mpa0, mp0b3, polar1m")
    require(len(cfg["families"]) == 6, "campaign must contain six new foundation-by-family jobs")

    for foundation_id, foundation in cfg["foundation_models"].items():
        require(foundation["role"] == "public_zero_shot_foundation", f"{foundation_id}: role must be public_zero_shot_foundation")
        require(foundation["source"].startswith(cfg["remote_root"] + "/models/"), f"{foundation_id}: model must live below the new campaign root")
        if foundation_id == "polar1m":
            require(foundation["model_type"] == "PolarMACE", "polar1m must use model_type PolarMACE")
            require(foundation["comparability"] == "exploratory_not_primary_material_baseline", "PolarMACE comparability label is mandatory")
        else:
            require(foundation["model_type"] == "MACE", f"{foundation_id}: ordinary foundation must use model_type MACE")
            require(foundation["comparability"] == "primary_material_baseline", f"{foundation_id}: material comparability label is mandatory")

    selected_families = list(cfg["families"])
    if args.families:
        selected_families = [item.strip() for item in args.families.split(",") if item.strip()]
        require(selected_families, "--families cannot be empty")
        require(set(selected_families) <= set(cfg["families"]), "--families contains an unknown family")

    for family_id, family in cfg["families"].items():
        require(family["dataset"] in cfg["datasets"], f"{family_id}: unknown dataset {family['dataset']}")
        require(family["foundation_model"] in cfg["foundation_models"], f"{family_id}: unknown foundation")
        foundation = cfg["foundation_models"][family["foundation_model"]]
        require(family["runtime"] == ("polar" if foundation["model_type"] == "PolarMACE" else "core"), f"{family_id}: runtime/model-type mismatch")
        require("{seed}" in family["output_dir_template"], f"{family_id}: output_dir_template must include {{seed}}")
        require(bool(family.get("multiheads_finetuning")) is (foundation["model_type"] == "MACE"), f"{family_id}: multiheads_finetuning is inconsistent with model type")

    require(cfg["md"]["md_velocity_seed_by_training_seed"] == {"123": 1100}, "MD velocity seed map must be exactly training seed 123 -> 1100")
    require(cfg["md"]["required_natoms"] == 2050, "common MD input must contain 2050 atoms")
    require(tuple(cfg["md"]["required_pbc"]) == (True, True, True), "common MD input must be fully periodic")

    checked = {"models": {}, "datasets": {}, "output_dirs": {}}
    if args.check_files:
        root = Path(cfg["remote_root"])
        require(root.is_dir(), f"missing campaign root: {root}")
        for foundation_id, foundation in cfg["foundation_models"].items():
            model_path = Path(foundation["source"])
            require(model_path.is_file() and model_path.stat().st_size > 0, f"missing foundation model: {model_path}")
            actual = sha256(model_path)
            require(actual == foundation["sha256"], f"foundation SHA256 mismatch for {foundation_id}: expected {foundation['sha256']} got {actual}")
            checked["models"][foundation_id] = {"path": str(model_path), "sha256": actual, "bytes": model_path.stat().st_size}
        for dataset_id, dataset in cfg["datasets"].items():
            dataset_root = Path(dataset["root"])
            require(dataset_root.is_dir(), f"missing dataset root: {dataset_root}")
            rows = {}
            for name, expected in dataset["files"].items():
                path = dataset_root / name
                require(path.is_file() and path.stat().st_size > 0, f"missing dataset file: {path}")
                actual = sha256(path)
                require(actual == expected, f"dataset SHA256 mismatch for {path}: expected {expected} got {actual}")
                rows[name] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
            checked["datasets"][dataset_id] = rows
        for family_id in selected_families:
            family = cfg["families"][family_id]
            output_dir = root / family["output_dir_template"].format(seed=123)
            require(not output_dir.exists(), f"output collision: {output_dir}")
            checked["output_dirs"][family_id] = str(output_dir)

    jobs = [{"family": family_id, "training_seed": 123, "foundation": cfg["families"][family_id]["foundation_model"]} for family_id in selected_families]
    print(json.dumps({"campaign_id": cfg["campaign_id"], "families": selected_families, "jobs": jobs, "checked": checked}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
