#!/usr/bin/env python3
"""Materialize endpoint-bound nonlinear candidates from an audited remap plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def resolve_artifact(owner: Path, recorded: str, expected_hash: str) -> Path:
    path = Path(recorded)
    candidates = [path, owner.parent / path.name]
    matches = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    matches = [candidate for candidate in matches if sha256(candidate) == expected_hash]
    if len({str(path) for path in matches}) != 1:
        raise ValueError(f"Cannot resolve one hash-matching artifact for {recorded}")
    return matches[0]


def endpoint_record(acceptance: dict, image_index_qe: int) -> dict:
    matches = [
        row for row in acceptance.get("endpoint_records", [])
        if int(row.get("image_index_qe", -1)) == int(image_index_qe)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one accepted endpoint record for image {image_index_qe}")
    if matches[0].get("status") != "accepted":
        raise ValueError(f"Endpoint image {image_index_qe} is not accepted")
    return matches[0]


def validate_segment_acceptance(segment: dict, acceptance_path: Path) -> dict:
    acceptance = json.loads(acceptance_path.read_text())
    if acceptance.get("status") != "accepted":
        raise ValueError(f"Segment acceptance is not accepted: {acceptance_path}")
    initial_index = int(segment["historical_start_image_qe"])
    final_index = int(segment["historical_end_image_qe"])
    initial_record = endpoint_record(acceptance, initial_index)
    final_record = endpoint_record(acceptance, final_index)
    if initial_record.get("calculator_identity") != final_record.get("calculator_identity"):
        raise ValueError("Segment endpoints use different calculator identities")
    initial = resolve_artifact(
        acceptance_path,
        acceptance["initial_structure"],
        acceptance["initial_structure_sha256"],
    )
    final = resolve_artifact(
        acceptance_path,
        acceptance["final_structure"],
        acceptance["final_structure_sha256"],
    )
    return {
        "acceptance": acceptance,
        "acceptance_path": acceptance_path.resolve(),
        "acceptance_sha256": sha256(acceptance_path),
        "initial_structure": initial,
        "final_structure": final,
        "calculator_identity": initial_record["calculator_identity"],
    }


def parse_acceptances(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--acceptance must be SEGMENT_ID=PATH")
        segment_id, raw_path = value.split("=", 1)
        if segment_id in result:
            raise ValueError(f"Duplicate acceptance for {segment_id}")
        result[segment_id] = Path(raw_path).resolve()
    return result


def materialize(args) -> tuple[Path, dict]:
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    acceptances = parse_acceptances(args.acceptance)
    expected_ids = {row["segment_id"] for row in plan["segments"]}
    if set(acceptances) != expected_ids:
        raise ValueError(f"Acceptance IDs must equal planned segments: {sorted(expected_ids)}")
    donor = plan["historical_curvature_donor"]
    project_root = plan_path.parents[1]
    historical_images = (project_root / donor["images"]).resolve()
    historical_template = (project_root / donor["qe_template"]).resolve()
    if sha256(historical_images) != donor["images_sha256"]:
        raise ValueError("Historical donor images hash mismatch")
    if sha256(historical_template) != donor["qe_template_sha256"]:
        raise ValueError("Historical donor template hash mismatch")

    outputs = []
    for segment in plan["segments"]:
        segment_id = segment["segment_id"]
        evidence = validate_segment_acceptance(segment, acceptances[segment_id])
        out_dir = args.out_root.resolve() / segment_id
        command = [
            sys.executable,
            str(args.generator.resolve()),
            "--initial-structure", str(evidence["initial_structure"]),
            "--final-structure", str(evidence["final_structure"]),
            "--endpoint-acceptance", str(evidence["acceptance_path"]),
            "--endpoint-status", "accepted_local_minima",
            "--historical-images", str(historical_images),
            "--historical-template-qe", str(historical_template),
            "--historical-start-image-qe", str(segment["historical_start_image_qe"]),
            "--historical-end-image-qe", str(segment["historical_end_image_qe"]),
            "--n-images", str(segment["n_output_images"]),
            "--seed", str(plan["seed"]),
            "--out-dir", str(out_dir),
        ]
        subprocess.run(command, check=True)
        candidate_path = out_dir / "nonlinear_candidate_manifest.json"
        candidate = json.loads(candidate_path.read_text())
        if candidate.get("endpoint_status") != "accepted_local_minima":
            raise ValueError(f"Generated candidate is not endpoint accepted: {segment_id}")
        if not any(row.get("production_eligible_before_mace") for row in candidate.get("candidates", [])):
            raise ValueError(f"No production-eligible candidate for {segment_id}")
        outputs.append({
            "segment_id": segment_id,
            "endpoint_acceptance": str(evidence["acceptance_path"]),
            "endpoint_acceptance_sha256": evidence["acceptance_sha256"],
            "calculator_identity": evidence["calculator_identity"],
            "candidate_manifest": str(candidate_path),
            "candidate_manifest_sha256": sha256(candidate_path),
            "status": "materialized_for_mace_selection",
        })

    identities = {row["calculator_identity"] for row in outputs}
    if len(identities) != 1:
        raise ValueError("Remapped segments do not share one endpoint calculator identity")
    result = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "historical_donor_reused_as_labels": False,
        "accepted_endpoint_coordinates_preserved_exactly": True,
        "segments": outputs,
        "status": "ready_for_new_mace_runs",
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    output = args.out_root / "endpoint_remap_materialization_manifest.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    return output, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--acceptance", action="append", default=[], help="SEGMENT_ID=endpoint_pair_acceptance.json")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--generator", type=Path, default=Path(__file__).with_name("generate_nonlinear_neb_candidates.py"))
    args = parser.parse_args()
    output, result = materialize(args)
    print(json.dumps({"output": str(output), "segments": len(result["segments"])}, indent=2))


if __name__ == "__main__":
    main()
