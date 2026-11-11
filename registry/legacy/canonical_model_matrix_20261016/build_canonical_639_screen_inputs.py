#!/usr/bin/env python3
"""Build immutable input tables for the accepted canonical 639-model screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--model-registry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidate_path = Path(args.candidate_manifest)
    registry_path = Path(args.model_registry)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(candidate_path, dtype=str).fillna("")
    registry = pd.read_csv(registry_path, sep="\t", dtype=str).fillna("")
    required_candidate = {"candidate_id", "candidate_index", "prepared_extxyz", "candidate_group", "window"}
    missing_candidate = sorted(required_candidate - set(candidates.columns))
    if missing_candidate:
        raise SystemExit(f"candidate manifest missing columns: {missing_candidate}")
    if len(candidates) != 639:
        raise SystemExit(f"expected exactly 639 candidates, observed {len(candidates)}")
    if candidates["candidate_id"].duplicated().any():
        raise SystemExit("candidate_id is not unique")
    if candidates["prepared_extxyz"].eq("").any():
        raise SystemExit("candidate manifest contains empty prepared_extxyz")

    required_registry = {
        "canonical_model_id",
        "model_family",
        "training_seed",
        "model_path",
        "model_sha256",
        "computational_status",
        "provenance_status",
    }
    missing_registry = sorted(required_registry - set(registry.columns))
    if missing_registry:
        raise SystemExit(f"model registry missing columns: {missing_registry}")
    accepted_statuses = {"REGISTERED", "ACCEPTED", "COMPLETED_ACCEPTED"}
    selected = registry[registry["computational_status"].isin(accepted_statuses)].copy()
    if selected.empty:
        raise SystemExit("no accepted models are available")
    if selected["canonical_model_id"].duplicated().any():
        raise SystemExit("canonical_model_id is not unique")
    if selected["model_path"].eq("").any() or selected["model_sha256"].eq("").any():
        raise SystemExit("accepted model has an empty path or hash")
    if selected["provenance_status"].str.contains("PROVISIONAL|UNVERIFIED|FAIL", case=False, regex=True).any():
        bad = selected[selected["provenance_status"].str.contains("PROVISIONAL|UNVERIFIED|FAIL", case=False, regex=True)]
        raise SystemExit(f"accepted model has non-final provenance: {bad['canonical_model_id'].tolist()}")

    screen_models = selected.rename(
        columns={
            "canonical_model_id": "model_id",
            "publication_role": "model_label",
        }
    ).copy()
    if "model_label" not in screen_models:
        screen_models["model_label"] = screen_models["model_family"]
    screen_models["model_label"] = screen_models["model_label"].replace("", pd.NA).fillna(screen_models["model_family"])
    columns = ["model_id", "model_label", "model_family", "training_seed", "model_path", "model_sha256"]
    screen_models[columns].to_csv(output / "models_rockfish.tsv", sep="\t", index=False)
    candidates.to_csv(output / "candidate_manifest_639.csv", index=False)

    manifest = {
        "campaign_id": "canonical_model_matrix_20261016",
        "screen_id": "canonical_639_screen_v1",
        "status": "READY_FOR_SUBMISSION",
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": sha256(candidate_path),
        "candidate_count": len(candidates),
        "model_registry": str(registry_path),
        "model_registry_sha256": sha256(registry_path),
        "model_count": len(screen_models),
        "model_ids": screen_models["model_id"].tolist(),
        "row_count_expected": int(len(candidates) * len(screen_models)),
        "selection_rule": "computational_status in REGISTERED, ACCEPTED, COMPLETED_ACCEPTED; non-provisional provenance; nonempty path and hash",
        "historical_models_excluded": True,
        "worker_source": "analysis/multimodel_639_screen_20261014/run_mlip_candidate_chunk.py",
        "collector_source": "analysis/multimodel_639_screen_20261014/collect_mlip_results.py",
    }
    (output / "SCREEN_INPUT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "candidate_count": 639, "model_count": len(screen_models), "row_count_expected": manifest["row_count_expected"]}, indent=2))


if __name__ == "__main__":
    main()
