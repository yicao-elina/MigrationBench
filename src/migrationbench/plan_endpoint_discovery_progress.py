#!/usr/bin/env python3
"""Plan the next fail-closed stage for a prospective endpoint-discovery batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from compare_site_proposer_dft_validation import (
    FINAL_CLASSIFICATIONS,
    RESTARTABLE_CLASSIFICATIONS,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoint_key(row: dict) -> Tuple[str, int]:
    return row["path_id"], int(row["image_index_qe"])


def plan(
    batch_path: Path,
    first_status_path: Path,
    repeat_status_path: Optional[Path] = None,
    acceptance_path: Optional[Path] = None,
) -> dict:
    batch = json.loads(batch_path.read_text())
    first_status = json.loads(first_status_path.read_text())
    expected_branches = {row["branch_id"] for row in batch["jobs"]}
    first_by_branch = {row["branch_id"]: row for row in first_status["jobs"]}
    if set(first_by_branch) != expected_branches:
        raise ValueError("First-relax status does not exactly match the frozen batch")

    first_rows = list(first_by_branch.values())
    first_restartable = [row for row in first_rows if row["classification"] in RESTARTABLE_CLASSIFICATIONS]
    first_unresolved = [
        row for row in first_rows
        if row["classification"] not in FINAL_CLASSIFICATIONS | RESTARTABLE_CLASSIFICATIONS
    ]
    first_accepted = [row for row in first_rows if row["classification"] == "accepted_local_minimum"]
    first_rejected = [row for row in first_rows if row["classification"] in FINAL_CLASSIFICATIONS and row not in first_accepted]

    stage = "first_relax_resolved"
    actions = []
    if first_restartable:
        stage = "first_relax_continuation_required"
        actions.append("continue_first_relax_clean_parents")
    elif first_unresolved:
        active = all(row["classification"] in {"pending", "first_scf_in_progress", "ionic_relax_in_progress"} for row in first_unresolved)
        stage = "waiting_for_first_relax" if active else "first_relax_needs_review"
        actions.append("wait_for_cadence" if active else "review_first_relax_failures")
    elif not first_accepted:
        stage = "complete_no_first_relax_successes"
        actions.append("run_final_comparator_without_repeat")
    elif repeat_status_path is None:
        stage = "prepare_repeat_relaxations"
        actions.append("prepare_and_submit_repeat_relaxations")
    else:
        repeat_status = json.loads(repeat_status_path.read_text())
        repeat_by_key: Dict[Tuple[str, int], dict] = {endpoint_key(row): row for row in repeat_status["jobs"]}
        accepted_keys = {endpoint_key(row) for row in first_accepted}
        if set(repeat_by_key) != accepted_keys:
            raise ValueError("Repeat-relax status does not exactly match first-relax successes")
        repeat_rows = list(repeat_by_key.values())
        repeat_restartable = [row for row in repeat_rows if row["classification"] in RESTARTABLE_CLASSIFICATIONS]
        repeat_unresolved = [
            row for row in repeat_rows
            if row["classification"] not in FINAL_CLASSIFICATIONS | RESTARTABLE_CLASSIFICATIONS
        ]
        if repeat_restartable:
            stage = "repeat_relax_continuation_required"
            actions.append("continue_repeat_relax_clean_parents")
        elif repeat_unresolved:
            active = all(row["classification"] in {"pending", "first_scf_in_progress", "ionic_relax_in_progress"} for row in repeat_unresolved)
            stage = "waiting_for_repeat_relax" if active else "repeat_relax_needs_review"
            actions.append("wait_for_cadence" if active else "review_repeat_relax_failures")
        elif acceptance_path is None:
            stage = "build_repeat_acceptance"
            actions.append("build_endpoint_acceptance_batch")
        else:
            acceptance = json.loads(acceptance_path.read_text())
            if acceptance.get("first_status_sha256") != sha256_file(first_status_path):
                raise ValueError("Acceptance artifact is not bound to the first-relax status")
            acceptance_keys = {endpoint_key(row) for row in acceptance.get("endpoint_records", [])}
            if acceptance_keys != accepted_keys:
                raise ValueError("Acceptance artifact does not exactly cover first-relax successes")
            accepted_repeat = sum(row["status"] == "accepted" for row in acceptance["endpoint_records"])
            stage = "ready_for_basin_assignment_and_final_comparison"
            actions.extend(["assign_endpoint_basins"] if accepted_repeat else [])
            actions.append("run_repeat_validated_prospective_comparator")

    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "prospective_endpoint_discovery_state_plan",
        "batch_manifest": str(batch_path.resolve()),
        "batch_manifest_sha256": sha256_file(batch_path),
        "first_status": str(first_status_path.resolve()),
        "first_status_sha256": sha256_file(first_status_path),
        "repeat_status": str(repeat_status_path.resolve()) if repeat_status_path else None,
        "repeat_status_sha256": sha256_file(repeat_status_path) if repeat_status_path else None,
        "endpoint_acceptance": str(acceptance_path.resolve()) if acceptance_path else None,
        "endpoint_acceptance_sha256": sha256_file(acceptance_path) if acceptance_path else None,
        "stage": stage,
        "actions": actions,
        "counts": {
            "batch_jobs": len(batch["jobs"]),
            "first_accepted": len(first_accepted),
            "first_rejected": len(first_rejected),
            "first_restartable": len(first_restartable),
            "first_unresolved": len(first_unresolved),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--first-status", type=Path, required=True)
    parser.add_argument("--repeat-status", type=Path)
    parser.add_argument("--endpoint-acceptance", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plan(args.batch_manifest, args.first_status, args.repeat_status, args.endpoint_acceptance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"stage": result["stage"], "actions": result["actions"]}, indent=2))


if __name__ == "__main__":
    main()
