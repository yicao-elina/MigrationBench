#!/usr/bin/env python3
"""Build deterministic job tables and the campaign-local model registry."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, write_tsv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config, root = load_config(args.config)
    system_ids = list(config["systems"])
    current_ids = [mid for mid, item in config["models"].items() if item["role"].startswith("A1_") or item["role"].startswith("A2_") or item["role"].startswith("A3_") or item["role"].startswith("A4_")]
    a0_ids = [mid for mid, item in config["models"].items() if item["role"] == "A0_pipeline_self_check"]

    def make_rows(model_ids: list[str], stage: str) -> list[dict[str, str]]:
        rows = []
        index = 0
        for model_id in model_ids:
            for system_id in system_ids:
                rows.append({
                    "row_index": str(index),
                    "stage": stage,
                    "model_id": model_id,
                    "system_id": system_id,
                })
                index += 1
        return rows

    fields = ["row_index", "stage", "model_id", "system_id"]
    write_tsv(root / "jobs/a0_fixed_jobs.tsv", make_rows(a0_ids, "A0_FIXED"), fields)
    write_tsv(root / "jobs/fixed_jobs.tsv", make_rows(current_ids, "T1_FIXED"), fields)
    write_tsv(root / "jobs/dense_jobs.tsv", make_rows(current_ids, "T2_DENSE"), fields)

    registry_rows = []
    for model_id, item in config["models"].items():
        registry_rows.append({
            "canonical_model_id": model_id,
            "role": item["role"],
            "model_family": item["family"],
            "scratch_subfamily": item.get("scratch_subfamily", "none"),
            "model_label": item["label"],
            "training_seed": item.get("training_seed", "NA"),
            "seed_semantics": item.get("seed_semantics", "not_applicable"),
            "model_path": item["model_path"],
            "expected_model_sha256": item["model_sha256"],
            "provenance_status": item.get("provenance_status", "PROVENANCE_PARTIAL"),
            "publication_status": item["publication_status"],
            "old_model_name": item.get("old_model_name", ""),
        })
    registry_fields = list(registry_rows[0])
    write_tsv(root / "manifests/MODEL_REGISTRY_USED.tsv", registry_rows, registry_fields)
    print(f"wrote {len(a0_ids) * len(system_ids)} A0 rows")
    print(f"wrote {len(current_ids) * len(system_ids)} fixed rows")
    print(f"wrote {len(current_ids) * len(system_ids)} dense rows")
    print(f"wrote {root / 'manifests/MODEL_REGISTRY_USED.tsv'}")


if __name__ == "__main__":
    main()

