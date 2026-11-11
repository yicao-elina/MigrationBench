#!/usr/bin/env python3
"""Build the final auditable release manifest for Foundation self-relaxed R02."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_ROWS = 108
EXPECTED_ATTEMPTS = 125
REFERENCE_SHA256 = "59d6202451f9f94ff21f3fba9924e025001c0af6c406c2bdf2ca1fc947023f84"
EXPECTED_MAP_ROWS = {
    "SCHEDULER_TASK_MAP.tsv": 108,
    "SCHEDULER_TASK_MAP_RETRY_R02A.tsv": 10,
    "SCHEDULER_TASK_MAP_RETRY_R02B.tsv": 5,
    "SCHEDULER_TASK_MAP_RETRY_R02C.tsv": 1,
    "SCHEDULER_TASK_MAP_SMOKE.tsv": 1,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def review_pass(path: Path) -> bool:
    return path.is_file() and path.read_text().lstrip().splitlines()[0].strip() == "PASS"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    figures = root / "figures"
    bookkeeping = read_json(root / "EXECUTION_BOOKKEEPING_R02.json")
    collection = read_json(data / "COLLECTION_MANIFEST.json")
    derived = read_json(data / "DERIVED_PROFILE_MANIFEST.json")
    visual_qa = read_json(figures / "VISUAL_QA_R02.json")
    rows = read_tsv(data / "ROW_RESULTS.tsv")
    profiles = read_tsv(data / "IMAGE_PROFILES.tsv")
    attempts = read_tsv(data / "ATTEMPT_LEDGER.tsv")
    raw_index = read_tsv(data / "RAW_ARTIFACT_INDEX.tsv")

    checks: dict[str, bool] = {}
    checks["collection_finalized_108_rows"] = (
        collection.get("finalized") is True
        and collection.get("target_rows") == EXPECTED_ROWS
        and collection.get("rows_with_manifests") == EXPECTED_ROWS
        and collection.get("converged_rows") == EXPECTED_ROWS
        and collection.get("failure_rows") == 0
        and collection.get("profile_rows") == EXPECTED_ROWS * 10
        and collection.get("attempt_rows") == EXPECTED_ATTEMPTS
    )
    checks["collection_output_hashes"] = all(
        (data / name).is_file() and sha256(data / name) == digest
        for name, digest in collection.get("outputs", {}).items()
    )
    checks["row_level_selection"] = (
        len(rows) == EXPECTED_ROWS
        and all(row.get("scientific_status") == "CONVERGED" for row in rows)
        and all(row.get("publication_status") == "PRIMARY_EVIDENCE" for row in rows)
        and all(row.get("optimizer_converged") == "true" for row in rows)
        and all(row.get("target_model_convergence") == "true" for row in rows)
        and all(row.get("n_profile_images") == "10" for row in rows)
    )
    checks["retry_manifest_role_consistency"] = all(
        row.get("manifest_role") == row.get("selected_attempt_manifest_role")
        and (
            not row.get("manifest_role", "").startswith("RETRY_ARRAY_")
            or row.get("retry_round") == row.get("manifest_role", "").removeprefix("RETRY_ARRAY_")
        )
        for row in rows
    )
    checks["attempt_and_raw_index_complete"] = (
        len(attempts) == EXPECTED_ATTEMPTS
        and len(raw_index) == EXPECTED_ATTEMPTS
        and {
            (row.get("record_id"), row.get("attempt_id")) for row in attempts
        }
        == {
            (row.get("record_id"), row.get("attempt_id")) for row in raw_index
        }
        and read_json(data / "RAW_ARTIFACT_INDEX.json").get("attempt_manifest_count") == EXPECTED_ATTEMPTS
        and read_json(data / "RAW_ARTIFACT_INDEX.json").get("output_sha256") == sha256(data / "RAW_ARTIFACT_INDEX.tsv")
    )
    checks["scheduler_maps_machine_readable"] = True
    scheduler_summary = {}
    for name, expected in EXPECTED_MAP_ROWS.items():
        path = data / name
        try:
            task_rows = read_tsv(path)
            ok = (
                len(task_rows) == expected
                and all(row.get("State") == "COMPLETED" for row in task_rows)
                and all(row.get("ExitCode") == "0:0" for row in task_rows)
                and all(len(row) == 7 for row in task_rows)
            )
        except Exception:
            task_rows = []
            ok = False
        checks["scheduler_maps_machine_readable"] = checks["scheduler_maps_machine_readable"] and ok
        scheduler_summary[name] = {"rows": len(task_rows), "sha256": sha256(path) if path.is_file() else "", "pass": ok}
    checks["derived_profiles_bound_and_complete"] = (
        derived.get("release_ready") is True
        and derived.get("target_rows") == EXPECTED_ROWS
        and derived.get("accepted_rows") == EXPECTED_ROWS
        and derived.get("profile_rows") == 360
        and derived.get("inputs", {}).get("ROW_RESULTS.tsv") == sha256(data / "ROW_RESULTS.tsv")
        and derived.get("inputs", {}).get("IMAGE_PROFILES.tsv") == sha256(data / "IMAGE_PROFILES.tsv")
        and all(row.get("status") == "PASS" for row in read_tsv(data / "CLASS_SUMMARY.tsv"))
    )
    checks["visual_export_set_and_style"] = (
        visual_qa.get("release_ready_input") is True
        and visual_qa.get("composite_axes") == 12
        and visual_qa.get("standalone_panel_count") == 12
        and len(visual_qa.get("outputs", [])) == 39
        and visual_qa.get("style", {}).get("grid") is False
        and visual_qa.get("style", {}).get("blender") is False
        and visual_qa.get("style", {}).get("independent_y_axes") is True
        and visual_qa.get("style", {}).get("intended_composite_width_mm") == 180
        and visual_qa.get("style", {}).get("intended_standalone_width_mm") == 100
        and visual_qa.get("style", {}).get("minimum_text_size_pt_at_intended_width") >= 11
        and all(Path(item["path"]).is_file() and sha256(Path(item["path"])) == item["sha256"] for item in visual_qa["outputs"])
    )
    reference = visual_qa.get("reference_standard", {})
    reference_path = Path(reference.get("resolved_authoritative_path", ""))
    checks["authoritative_reference_bound"] = (
        reference.get("requested_path_exists") is False
        and reference_path.is_file()
        and sha256(reference_path) == REFERENCE_SHA256
        and reference.get("resolved_authoritative_sha256") == REFERENCE_SHA256
    )
    checks["bookkeeping_final_hashes"] = (
        bookkeeping.get("formal_launch_manifest_is_immutable") is True
        and bookkeeping.get("formal_launch_manifest_sha256") == "e226280ce571cc7df286824a627772a1b8c3ebe7808fde5ca7805e82d1f0bd4c"
        and bookkeeping.get("formal_array_status") == "COMPLETED_ALL_ROWS"
        and bookkeeping.get("final_converged_rows") == EXPECTED_ROWS
        and bookkeeping.get("final_failure_rows") == 0
        and bookkeeping.get("final_attempt_rows") == EXPECTED_ATTEMPTS
        and bookkeeping.get("final_collection_manifest_sha256") == sha256(data / "COLLECTION_MANIFEST.json")
        and bookkeeping.get("final_row_results_sha256") == sha256(data / "ROW_RESULTS.tsv")
        and bookkeeping.get("final_raw_artifact_index_sha256") == sha256(data / "RAW_ARTIFACT_INDEX.tsv")
        and bookkeeping.get("final_derived_profile_manifest_sha256") == sha256(data / "DERIVED_PROFILE_MANIFEST.json")
    )
    reviews = {
        "data": root / "data_review" / "data_review_R02_final.md",
        "visual": root / "visual_review" / "visual_review_R02.md",
    }
    checks["independent_reviews_pass"] = all(review_pass(path) for path in reviews.values())
    checks["caption_present"] = (root / "CAPTION_FOUNDATION_SELF_RELAXED_4x3_R02.md").is_file()
    checks["failure_registry_schema"] = data.joinpath("FAILURE_REGISTRY.tsv").read_text().splitlines()[:1] == [
        "record_id\telement\tpath_id\tmodel_id\tscientific_status\tfailure_class\tfailure_reason\tattempt_dir"
    ]

    release = {
        "schema_version": "foundation_self_relaxed_4x3_release_manifest_v1",
        "campaign_id": bookkeeping.get("campaign_id"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_ready": all(checks.values()),
        "checks": checks,
        "evidence": {
            "bookkeeping": {"path": str(root / "EXECUTION_BOOKKEEPING_R02.json"), "sha256": sha256(root / "EXECUTION_BOOKKEEPING_R02.json")},
            "collection_manifest": {"path": str(data / "COLLECTION_MANIFEST.json"), "sha256": sha256(data / "COLLECTION_MANIFEST.json")},
            "derived_profile_manifest": {"path": str(data / "DERIVED_PROFILE_MANIFEST.json"), "sha256": sha256(data / "DERIVED_PROFILE_MANIFEST.json")},
            "visual_qa": {"path": str(figures / "VISUAL_QA_R02.json"), "sha256": sha256(figures / "VISUAL_QA_R02.json")},
            "scheduler_maps": scheduler_summary,
            "reviews": {name: {"path": str(path), "sha256": sha256(path) if path.is_file() else ""} for name, path in reviews.items()},
            "reference_standard": reference,
        },
        "scientific_boundary": "SELF_RELAXED_ENDPOINT_ONLY_NEB; Foundation-MLFF diagnostic observable, not DFT ground truth or an unconditional physical barrier claim.",
        "no_blender": True,
    }
    out = root / "RELEASE_MANIFEST_R02.json"
    out.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
    print(json.dumps(release, indent=2, sort_keys=True))
    if not release["release_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
