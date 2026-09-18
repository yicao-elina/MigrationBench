#!/usr/bin/env python3
"""Validate a portable matched direct/MLFF QE NEB handoff bundle."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from audit_qe_calculator_identity import identity as qe_calculator_identity


def sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def image_counts(text):
    declared = re.search(r"num_of_images\s*=\s*(\d+)", text, re.I)
    return {
        "declared": int(declared.group(1)) if declared else None,
        "first": len(re.findall(r"^\s*FIRST_IMAGE\s*$", text, re.M)),
        "intermediate": len(re.findall(r"^\s*INTERMEDIATE_IMAGE\s*$", text, re.M)),
        "last": len(re.findall(r"^\s*LAST_IMAGE\s*$", text, re.M)),
    }


def validate(pair_path):
    pair_path = pair_path.resolve()
    pair = json.loads(pair_path.read_text())
    rows = []
    schema_1_1 = pair.get("schema_version") == "1.1"
    for role, path_key, hash_key, manifest_key, manifest_hash_key in (
        (
            "direct_baseline", "direct_input", "direct_input_sha256",
            "direct_handoff_manifest", "direct_handoff_manifest_sha256",
        ),
        (
            "mlff_preconditioned", "mlff_input", "mlff_input_sha256",
            "mlff_handoff_manifest", "mlff_handoff_manifest_sha256",
        ),
    ):
        path = (pair_path.parent / pair[path_key]).resolve()
        text = path.read_text()
        counts = image_counts(text)
        identity = qe_calculator_identity(path, role)
        checks = {
            "sha256": sha256(path) == pair[hash_key],
            "one_first": counts["first"] == 1,
            "one_last": counts["last"] == 1,
            "image_count": counts["declared"] == counts["intermediate"] + 2,
            "identity_complete": identity["identity_complete"],
            "identity_matches_pair": identity["calculator_identity"] == pair["shared_calculator_identity"],
        }
        handoff_path = None
        if schema_1_1:
            handoff_path = (pair_path.parent / pair[manifest_key]).resolve()
            handoff = json.loads(handoff_path.read_text())
            expected_initialization_role = (
                "direct_baseline" if role == "direct_baseline"
                else pair.get("mlff_initialization_role", "mlff_preconditioned")
            )
            checks.update({
                "handoff_manifest_sha256": sha256(handoff_path) == pair[manifest_hash_key],
                "handoff_output_binding": handoff.get("output_sha256") == pair[hash_key],
                "handoff_initialization_role": (
                    handoff.get("initialization_role") == expected_initialization_role
                ),
                "handoff_candidate_binding": (
                    handoff.get("source_candidate_manifest_sha256")
                    == pair.get("source_candidate_manifest_sha256")
                ),
            })
            if role == "mlff_preconditioned":
                checks["handoff_mlff_binding"] = (
                    handoff.get("source_mlff_manifest_sha256")
                    == pair.get("source_mlff_manifest_sha256")
                )
                if expected_initialization_role == "trust_region_mlff_preconditioned":
                    checks["handoff_trust_acceptance_binding"] = (
                        handoff.get("mlff_preconditioner_acceptance_sha256")
                        == pair.get("mlff_preconditioner_acceptance_sha256")
                    )
            if pair.get("production_submission_eligible"):
                checks["production_handoff_gate"] = (
                    handoff.get("production_handoff_gate", {}).get("status") == "pass"
                )
        rows.append(
            {
                "role": role,
                "path": str(path),
                "sha256": sha256(path),
                "image_counts": counts,
                "calculator_identity": identity["calculator_identity"],
                "handoff_manifest": str(handoff_path) if handoff_path else None,
                "checks": checks,
                "status": "pass" if all(checks.values()) else "fail",
            }
        )
    status = "pass" if all(row["status"] == "pass" for row in rows) else "fail"
    result = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pair_manifest": str(pair_path),
        "pair_manifest_sha256": sha256(pair_path),
        "status": status,
        "rows": rows,
    }
    if status != "pass":
        raise ValueError(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.pair_manifest)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "roles": len(result["rows"])}, indent=2))
