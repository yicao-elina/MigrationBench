#!/usr/bin/env python3
"""Select a bounded, reproducible DFT-ready candidate set from accepted MLIP output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


GROUP_PRIORITY = {
    "precursor_near_onset": 0,
    "threshold_or_terminal": 1,
    "stable_pre_onset": 2,
    "stable_early_control": 3,
    "terminal_stable_target": 4,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spread", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--analysis-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()
    spread_path = Path(args.spread)
    candidate_path = Path(args.candidate_manifest)
    analysis_manifest_path = Path(args.analysis_manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.count < 1 or args.count > 20:
        raise SystemExit("count must be between 1 and 20")
    analysis_manifest = json.loads(analysis_manifest_path.read_text())
    if analysis_manifest.get("status") != "VERIFIED":
        raise SystemExit("canonical screen analysis is not VERIFIED")

    spread = pd.read_csv(spread_path)
    candidates = pd.read_csv(candidate_path)
    required = {"candidate_id", "candidate_group", "run_key", "prepared_extxyz", "source_snapshot"}
    if not required.issubset(candidates.columns):
        raise SystemExit(f"candidate manifest missing {sorted(required - set(candidates.columns))}")
    if spread["candidate_id"].duplicated().any():
        raise SystemExit("candidate spread must contain exactly one row per candidate")
    if len(spread) != len(candidates):
        raise SystemExit("candidate spread does not cover the full candidate manifest")
    if spread["max_force_range_eV_A"].isna().any():
        raise SystemExit("candidate spread contains missing force-range values")
    merged = candidates.merge(
        spread[["candidate_id", "max_force_range_eV_A", "pairwise_force_rms_max_eV_A", "n_models_success"]],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    merged["group_priority"] = merged["candidate_group"].map(GROUP_PRIORITY).fillna(99)
    merged["score"] = pd.to_numeric(merged["pairwise_force_rms_max_eV_A"], errors="coerce").fillna(0.0)
    merged = merged.sort_values(["group_priority", "score", "max_force_range_eV_A", "candidate_id"], ascending=[True, False, False, True])

    chosen = []
    used_runs = set()
    for _, row in merged.iterrows():
        if row["run_key"] in used_runs:
            continue
        chosen.append(row)
        used_runs.add(row["run_key"])
        if len(chosen) == args.count:
            break
    if len(chosen) < args.count:
        chosen_ids = {row["candidate_id"] for row in chosen}
        for _, row in merged.iterrows():
            if row["candidate_id"] in chosen_ids:
                continue
            chosen.append(row)
            if len(chosen) == args.count:
                break
    selected = pd.DataFrame(chosen).reset_index(drop=True)
    selected["batch_order"] = range(len(selected))
    selected["selection_status"] = "SELECTED_FROM_ACCEPTED_CANONICAL_SCREEN"
    selected["dft_scientific_status"] = "PREPARED_ONLY"
    selected.to_csv(output / "dft_candidate_selection.csv", index=False)
    payload = {
        "schema": "migrationbench_canonical_dft_selection_v1",
        "status": "SELECTED_FROM_ACCEPTED_CANONICAL_SCREEN",
        "analysis_manifest": str(analysis_manifest_path),
        "analysis_manifest_sha256": sha256(analysis_manifest_path),
        "spread": str(spread_path),
        "spread_sha256": sha256(spread_path),
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": sha256(candidate_path),
        "n_selected": len(selected),
        "one_candidate_per_run_first_pass": True,
        "selection_order": "group priority, descending paired force disagreement, descending max-force range, candidate_id",
        "qe_status": "NOT_SUBMITTED",
        "exact_full_cell_parity": "NOT_CLAIMED",
        "candidate_ids": selected["candidate_id"].tolist(),
    }
    (output / "DFT_SELECTION_MANIFEST.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "n_selected": len(selected), "qe_status": payload["qe_status"]}, indent=2))


if __name__ == "__main__":
    main()
