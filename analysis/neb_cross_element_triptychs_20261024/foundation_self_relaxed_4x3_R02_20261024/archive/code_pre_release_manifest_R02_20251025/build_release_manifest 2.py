#!/usr/bin/env python3
"""Assemble a release gate manifest from immutable launch and review evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def review_pass(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(errors="replace")
    first_lines = "\n".join(text.splitlines()[:12]).upper()
    return bool(re.search(r"(?:^|\n)\s*(?:CONCLUSION\s*:\s*)?PASS\b", first_lines))


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    figures = root / "figures"
    launch = json.loads((root / "CAMPAIGN_MANIFEST.json").read_text())
    bookkeeping_path = root / "EXECUTION_BOOKKEEPING_R02.json"
    bookkeeping = json.loads(bookkeeping_path.read_text())
    collection_path = data / "COLLECTION_MANIFEST.json"
    derived_path = data / "DERIVED_PROFILE_MANIFEST.json"
    collection = json.loads(collection_path.read_text()) if collection_path.is_file() else {}
    derived = json.loads(derived_path.read_text()) if derived_path.is_file() else {}
    rows = read_tsv(data / "ROW_RESULTS.tsv")
    failures = read_tsv(data / "FAILURE_REGISTRY.tsv")
    review_data = root / "data_review" / "data_review_R02_final.md"
    review_visual = root / "visual_review" / "visual_review_R02.md"
    expected_stems = ["foundation_self_relaxed_4x3_R02"] + [
        f"panel_{element}_{path}_self_relaxed_R02"
        for element in ("Cr", "Cu", "Li", "Na")
        for path in ("in_gap", "gap_to_ql", "within_ql")
    ]
    formats = {fmt: [figures / f"{stem}.{fmt}" if stem.startswith("foundation_") else figures / "panels" / f"{stem}.{fmt}" for stem in expected_stems] for fmt in ("png", "svg", "pdf")}
    figures_complete = all(path.is_file() and path.stat().st_size > 0 for paths in formats.values() for path in paths)
    final_collection = bool(collection.get("finalized"))
    rows_complete = len(rows) == 108
    all_converged = rows_complete and len([row for row in rows if row.get("scientific_status") == "CONVERGED"]) == 108
    all_primary = rows_complete and len([row for row in rows if row.get("publication_status") == "PRIMARY_EVIDENCE"]) == 108
    data_gate = final_collection and rows_complete and all_converged and all_primary and not failures
    review_gate = review_pass(review_data) and review_pass(review_visual)
    gate = {
        "formal_array_terminal": bookkeeping.get("formal_array_status") in {"COMPLETED", "COMPLETE", "FAILED_WITH_EXCLUSIONS"},
        "final_collection": final_collection,
        "108_rows_present": rows_complete,
        "108_rows_converged": all_converged,
        "108_rows_primary_evidence": all_primary,
        "no_failure_registry_rows": not failures,
        "derived_profile_release_ready": bool(derived.get("release_ready")),
        "scheduler_audit_present": (data / "SCHEDULER_AUDIT.json").is_file() and (data / "SCHEDULER_TASK_MAP.tsv").is_file(),
        "data_review_pass": review_pass(review_data),
        "visual_review_pass": review_pass(review_visual),
        "all_13_png_svg_pdf_outputs_present": figures_complete,
    }
    release_ready = data_gate and bool(derived.get("release_ready")) and gate["scheduler_audit_present"] and gate["data_review_pass"] and gate["visual_review_pass"] and figures_complete
    evidence_names = [
        "CAMPAIGN_MANIFEST.json",
        "EXECUTION_BOOKKEEPING_R02.json",
        "JOB_TABLE_IDPP.tsv",
        "MODEL_REGISTRY.tsv",
        "PATH_REGISTRY.tsv",
        "data/SCHEDULER_TASK_MAP.tsv",
        "data/SCHEDULER_AUDIT.json",
        "data/COLLECTION_MANIFEST.json",
        "data/ROW_RESULTS.tsv",
        "data/IMAGE_PROFILES.tsv",
        "data/FAILURE_REGISTRY.tsv",
        "data/ATTEMPT_LEDGER.tsv",
        "data/RAW_ARTIFACT_INDEX.tsv",
        "data/RAW_ARTIFACT_INDEX.json",
        "data/DERIVED_PROFILE_MANIFEST.json",
        "data/CLASS_PROFILES.tsv",
        "data/CLASS_SUMMARY.tsv",
        "data/PROFILE_INCLUSION_AUDIT.tsv",
        "figures/VISUAL_QA_R02.json",
        "data_review/data_review_R02_final.md",
        "visual_review/visual_review_R02.md",
    ]
    evidence = {}
    for name in evidence_names:
        path = root / name
        if path.is_file():
            evidence[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    output = root / "RELEASE_MANIFEST_R02.json"
    payload = {
        "schema_version": "foundation_self_relaxed_release_manifest_v1",
        "campaign_id": launch.get("campaign_id"),
        "campaign_base_id": bookkeeping.get("campaign_base_id"),
        "campaign_revision_id": bookkeeping.get("campaign_revision_id"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_ready": release_ready,
        "target_rows": 108,
        "collected_rows": len(rows),
        "converged_rows": len([row for row in rows if row.get("scientific_status") == "CONVERGED"]),
        "failure_rows": len(failures),
        "immutable_formal_launch_manifest_sha256": bookkeeping.get("formal_launch_manifest_sha256"),
        "remote_raw_root": launch.get("remote_root"),
        "observable_boundary": "self-relaxed MLFF diagnostic/proposal profiles; not unconditional DFT ground truth",
        "fixed_path_profiles_included": False,
        "blender_included": False,
        "gate": gate,
        "evidence": evidence,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
