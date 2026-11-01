#!/usr/bin/env python3
"""Collect selected fixed/self-relaxed outputs into tidy local tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CAMPAIGN_ID = "scratch_600k_migration_canonical_20261101"

def latest_manifest(root: Path, observable: str, model_id: str, path_id: str) -> Path:
    if observable == "FIXED_PATH_SINGLE_POINT":
        pattern = f"runs/fixed/{model_id}/{path_id}/attempt_*/fixed_single_point_manifest.json"
    else:
        pattern = f"runs/idpp/{model_id}/{path_id}/attempt_*/endpoint_only_manifest.json"
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no {observable} manifest for {model_id}/{path_id}")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    models = pd.read_csv(args.raw_root / "MODEL_REGISTRY.tsv", sep="\t")
    paths = pd.read_csv(args.raw_root / "PATH_REGISTRY.tsv", sep="\t")
    model_lookup = models.set_index("model_id").to_dict("index")
    path_lookup = paths.set_index("path_id").to_dict("index")

    results: list[dict[str, object]] = []
    profile_rows: list[pd.DataFrame] = []
    for model_id, model in model_lookup.items():
        for path_id, path in path_lookup.items():
            fixed_manifest_path = latest_manifest(args.raw_root, "FIXED_PATH_SINGLE_POINT", model_id, path_id)
            fixed = json.loads(fixed_manifest_path.read_text())
            fixed_profile_path = fixed_manifest_path.parent / "image_profile.csv"
            profile = pd.read_csv(fixed_profile_path)
            profile["model_family"] = model["model_family"]
            profile["training_seed"] = model["training_seed"]
            profile["model_label"] = model["model_label"]
            profile_rows.append(profile)
            results.append(
                {
                    "campaign_id": CAMPAIGN_ID,
                    "record_id": fixed["record_id"],
                    "observable": fixed["observable"],
                    "initialization_method": fixed["initialization_method"],
                    "model_family": model["model_family"],
                    "model_id": model_id,
                    "model_label": model["model_label"],
                    "training_seed": model["training_seed"],
                    "path_id": path_id,
                    "path_class": path["path_class"],
                    "value_name": "forward_barrier_eV",
                    "value_eV": fixed["forward_barrier_eV"],
                    "scientific_status": fixed["scientific_status"],
                    "publication_status": fixed["publication_status"],
                    "failure_class": fixed.get("failure_class", ""),
                    "model_sha256": fixed["model_sha256"],
                    "fixed_path_sha256": path["fixed_path_sha256"],
                    "raw_manifest_path": str(fixed_manifest_path),
                    "raw_profile_path": str(fixed_profile_path),
                }
            )

            self_manifest_path = latest_manifest(args.raw_root, "SELF_RELAXED_ENDPOINT_ONLY_NEB", model_id, path_id)
            self_manifest = json.loads(self_manifest_path.read_text())
            engine_path = self_manifest_path.parent / "mlff_neb_manifest.json"
            engine = json.loads(engine_path.read_text())
            results.append(
                {
                    "campaign_id": CAMPAIGN_ID,
                    "record_id": self_manifest["record_id"],
                    "observable": self_manifest["observable"],
                    "initialization_method": self_manifest["initialization_method"],
                    "model_family": model["model_family"],
                    "model_id": model_id,
                    "model_label": model["model_label"],
                    "training_seed": model["training_seed"],
                    "path_id": path_id,
                    "path_class": path["path_class"],
                    "value_name": "barrier_proxy_eV",
                    "value_eV": engine["barrier_proxy_eV"],
                    "scientific_status": self_manifest["scientific_status"],
                    "publication_status": self_manifest["publication_status"],
                    "failure_class": self_manifest.get("failure_class", ""),
                    "model_sha256": self_manifest["model_sha256"],
                    "fixed_path_sha256": path["fixed_path_sha256"],
                    "raw_manifest_path": str(self_manifest_path),
                    "raw_profile_path": str(engine_path),
                }
            )

    result = pd.DataFrame(results).sort_values(["observable", "model_id", "path_id"])
    result.to_csv(args.output_dir / "SCRATCH600K_MIGRATION_RESULT_MATRIX.tsv", sep="\t", index=False)
    pd.concat(profile_rows, ignore_index=True).to_csv(
        args.output_dir / "SCRATCH600K_FIXED_IMAGE_PROFILES.tsv", sep="\t", index=False
    )
    print(result[["observable", "scientific_status", "publication_status"]].value_counts().to_string())
    print(f"wrote {len(result)} result rows to {args.output_dir}")


if __name__ == "__main__":
    main()
