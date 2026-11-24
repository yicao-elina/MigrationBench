#!/usr/bin/env python3
"""Validate the compact provenance capsule for the 639-candidate MLIP screen."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


EXPECTED_GROUPS = {
    "stable_early_control": 207,
    "stable_pre_onset": 216,
    "precursor_near_onset": 208,
    "threshold_or_terminal": 7,
    "terminal_stable_target": 1,
}
EXPECTED_WAVE = {
    "threshold_or_terminal": 7,
    "terminal_stable_target": 1,
    "precursor_near_onset": 9,
    "stable_early_model_control": 3,
}
EXPECTED_MLIP = {
    "threshold_or_terminal": 1,
    "precursor_near_onset": 12,
    "stable_pre_onset": 4,
    "stable_early_control": 2,
    "terminal_stable_target": 1,
}
EXPECTED_MODEL_HASHES = {
    "a26db3575de8484ca3c7fcf7621165e38ce0162a580cde022bccd16135c61351",
    "52fccd0994664618256ec5e986dc5e962050c845d3f05f3d70862ff064009af5",
    "9cadf6fa3c3212c512bc06ad1bb9392bd9b664c40949157ee6184e9c0b05811d",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fail(message: str) -> None:
    raise SystemExit(f"VALIDATION_FAILED: {message}")


def check_unique(rows: list[dict[str, str]], field: str, label: str) -> None:
    values = [row.get(field, "") for row in rows]
    if any(not value for value in values):
        fail(f"{label} contains empty {field}")
    if len(values) != len(set(values)):
        fail(f"{label} contains duplicate {field}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=Path("data/raw_compact/revision1/from_scratch_failure_attribution_20261013"),
    )
    args = parser.parse_args()
    root = args.evidence_dir

    candidates = read_csv(root / "candidate_categorization.csv")
    wave = read_csv(root / "wave_01_manifest.csv")
    mlip = read_csv(root / "mlip_dft_selection.csv")
    mapping = read_csv(root / "candidate_execution_mapping.csv")
    if len(candidates) != 639:
        fail(f"candidate inventory has {len(candidates)} rows, expected 639")
    check_unique(candidates, "candidate_id", "candidate inventory")
    if not all(re.fullmatch(r"dftcand_\d{4}", row["candidate_id"]) for row in candidates):
        fail("candidate IDs do not follow dftcand_#### scheme")
    group_counts = Counter(row.get("candidate_group", "") for row in candidates)
    if dict(group_counts) != EXPECTED_GROUPS:
        fail(f"candidate group counts {dict(group_counts)} != {EXPECTED_GROUPS}")

    candidate_ids = {row["candidate_id"] for row in candidates}
    if len(wave) != 20:
        fail(f"wave_01 has {len(wave)} rows, expected 20")
    check_unique(wave, "candidate_id", "wave_01")
    if not {row["candidate_id"] for row in wave} <= candidate_ids:
        fail("wave_01 contains IDs absent from the 639-row inventory")
    wave_counts = Counter(row.get("selection_role", "") for row in wave)
    if dict(wave_counts) != EXPECTED_WAVE:
        fail(f"wave_01 role counts {dict(wave_counts)} != {EXPECTED_WAVE}")

    if len(mlip) != 20:
        fail(f"MLIP-informed DFT selection has {len(mlip)} rows, expected 20")
    check_unique(mlip, "candidate_id", "MLIP-informed DFT selection")
    if not {row["candidate_id"] for row in mlip} <= candidate_ids:
        fail("MLIP-informed selection contains IDs absent from the 639-row inventory")
    mlip_counts = Counter(row.get("candidate_group", "") for row in mlip)
    if dict(mlip_counts) != EXPECTED_MLIP:
        fail(f"MLIP-informed group counts {dict(mlip_counts)} != {EXPECTED_MLIP}")
    for row in mlip:
        try:
            minimum_distance = float(row["min_distance_A"])
        except (KeyError, TypeError, ValueError) as exc:
            fail(f"MLIP selection has invalid min_distance_A: {exc}")
        if minimum_distance < 0.8:
            fail(f"MLIP selection violates min_distance_A >= 0.8: {row['candidate_id']}")
        if row.get("dft_status") != "prepared_only_mlip_selected":
            fail(f"unexpected DFT status for {row['candidate_id']}: {row.get('dft_status')}")

    if len(mapping) != 639:
        fail(f"execution-to-canonical mapping has {len(mapping)} rows, expected 639")
    check_unique(mapping, "prepared_extxyz", "execution-to-canonical mapping")
    check_unique(mapping, "candidate_id_canonical", "execution-to-canonical mapping canonical IDs")
    if {row["candidate_id_canonical"] for row in mapping} != candidate_ids:
        fail("execution-to-canonical mapping does not cover exactly the 639 canonical IDs")

    model_path = root / "MODEL_REGISTRY.tsv"
    with model_path.open(encoding="utf-8") as handle:
        model_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(model_rows) != 3:
        fail(f"model registry has {len(model_rows)} rows, expected 3")
    model_hashes = {row.get("model_sha256", "") for row in model_rows}
    if model_hashes != EXPECTED_MODEL_HASHES:
        fail("model registry hashes do not match the three registered checkpoints")

    summary = {
        "status": "VERIFIED",
        "candidate_count": len(candidates),
        "candidate_group_counts": dict(group_counts),
        "wave_01_count": len(wave),
        "wave_01_role_counts": dict(wave_counts),
        "mlip_informed_dft_count": len(mlip),
        "mlip_informed_group_counts": dict(mlip_counts),
        "mapped_execution_candidates": len(mapping),
        "registered_models": len(model_rows),
        "scientific_boundary": "selection provenance validated; no DFT correctness or causal mechanism claim",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
