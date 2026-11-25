#!/usr/bin/env python3
"""Check that the compact Revision-1 A1/A2 evidence release is navigable."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
INDEX = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    required = [
        INDEX / "README.md",
        INDEX / "claim_evidence_matrix.tsv",
        INDEX / "artifact_versions.tsv",
        INDEX / "failure_exclusion_registry.tsv",
        INDEX / "revision1_release_manifest.json",
        ROOT / "paper/response_to_reviewers/reviewer_a1_fig3_figs3_update/fig3d_figS3_protocol_side_by_side_audited.pdf",
        ROOT / "paper/response_to_reviewers/reviewer_a2_grouped_retraining/figures/a2_grouped_retraining_audit.pdf",
        ROOT / "paper/response_to_reviewers/reviewer_a2_grouped_retraining/data/overlap_full_leakage_audit.json",
        ROOT / "paper/response_to_reviewers/reviewer_a2_grouped_retraining/data/overlap_scratch5_leakage_audit.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing release files:\n" + "\n".join(missing))

    rows = list(csv.DictReader((INDEX / "artifact_versions.tsv").open(), delimiter="\t"))
    missing_artifacts = []
    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file():
            missing_artifacts.append(row["path"])
    if missing_artifacts:
        raise SystemExit("Missing indexed artifacts:\n" + "\n".join(missing_artifacts))

    with (INDEX / "revision1_release_manifest.json").open() as handle:
        json.load(handle)

    banned = []
    for package in (
        ROOT / "paper/response_to_reviewers/reviewer_a1_fig3_figs3_update",
        ROOT / "paper/response_to_reviewers/reviewer_a2_grouped_retraining",
        ROOT / "paper/response_to_reviewers/reviewer_a2_distribution_shift_task_ladder_20261018",
    ):
        for path in package.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".md", ".csv", ".json", ".py", ".tex"}:
                if "MLFF" in path.read_text(errors="replace"):
                    banned.append(str(path.relative_to(ROOT)))
    if banned:
        raise SystemExit("Legacy MLFF token remains in release text/data:\n" + "\n".join(banned))

    print(f"OK: {len(rows)} indexed artifacts; A1/A2 compact release is navigable")
    for path in required[-4:]:
        print(f"  {path.relative_to(ROOT)}  {sha256(path)[:16]}...")


if __name__ == "__main__":
    main()
