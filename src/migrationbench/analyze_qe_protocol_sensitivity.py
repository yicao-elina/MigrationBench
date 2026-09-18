#!/usr/bin/env python3
"""Compare paired fixed-geometry QE SCF energy differences across protocols."""

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


RY_TO_EV = 13.605693122994


def evaluate_gate(results, reference_profile, tolerance):
    """Return an unambiguous scientific/execution gate for a protocol stage."""
    reference = next(
        (row for row in results if row["profile_id"] == reference_profile), None
    )
    if reference is None:
        return "invalid", "reference_profile_missing"
    incomplete = [row for row in results if not row["both_scf_accepted"]]
    if incomplete:
        classifications = {
            value
            for row in incomplete
            for value in row.get("job_classifications", [])
            if value
        }
        terminal = {
            value for value in classifications
            if value in {
                "terminal_needs_review", "cancelled", "failed", "out_of_memory",
                "missing_remote_run", "unknown_terminal",
            }
        }
        if terminal:
            return "invalid", "one_or_more_scf_jobs_terminal_without_accepted_energy"
        return "pending", "waiting_for_all_profile_image_pairs"
    non_reference = [
        row for row in results if row["profile_id"] != reference_profile
    ]
    if not non_reference:
        return "invalid", "no_test_profile_present"
    failed = [
        row for row in non_reference
        if abs(row["delta_E_difference_from_reference_eV"]) > tolerance
    ]
    if failed:
        return "fail", "energy_contrast_is_protocol_sensitive"
    return "pass", "all_test_profiles_within_tolerance"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--reference-profile", required=True)
    parser.add_argument("--low-image", type=int, required=True)
    parser.add_argument("--high-image", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads(args.status.read_text())
    batch = json.loads(args.batch_manifest.read_text())
    job_manifest = {row["branch_id"]: row for row in batch["jobs"]}
    grouped = {}
    for row in status["jobs"]:
        profile_id = row["profile_id"]
        image_index = int(row["image_index_qe"])
        parsed = row.get("parsed", {})
        energy_ry = parsed.get("final_energy_Ry")
        accepted = (
            row.get("classification") == "accepted_scf_warmstart"
            and isinstance(energy_ry, (int, float))
            and math.isfinite(energy_ry)
        )
        grouped.setdefault(profile_id, {})[image_index] = {
            "accepted": accepted,
            "energy_Ry": energy_ry,
            "job_id": row["job_id"],
            "branch_id": row["branch_id"],
            "classification": row.get("classification"),
            "magnetic_observables": {
                key: parsed.get(key) for key in (
                    "total_magnetization_Bohr_magneton_cell",
                    "total_magnetization_components_Bohr_magneton_cell",
                    "absolute_magnetization_Bohr_magneton_cell",
                    "site_magnetic_moments_last",
                )
            },
            "calculator_identity": job_manifest[row["branch_id"]]["calculator_identity"]["calculator_identity"],
        }
    results = []
    for profile_id, images in sorted(grouped.items()):
        low, high = images.get(args.low_image), images.get(args.high_image)
        complete = bool(low and high and low["accepted"] and high["accepted"])
        delta = (high["energy_Ry"] - low["energy_Ry"]) * RY_TO_EV if complete else None
        results.append({
            "profile_id": profile_id,
            "calculator_identity": low["calculator_identity"] if low else None,
            "low_image_index": args.low_image,
            "high_image_index": args.high_image,
            "low_job_id": low["job_id"] if low else None,
            "high_job_id": high["job_id"] if high else None,
            "both_scf_accepted": complete,
            "job_classifications": [
                item["classification"] if item else None for item in (low, high)
            ],
            "low_image_magnetic_observables": low["magnetic_observables"] if low else None,
            "high_image_magnetic_observables": high["magnetic_observables"] if high else None,
            "delta_E_high_minus_low_eV": delta,
        })
    reference = next((row for row in results if row["profile_id"] == args.reference_profile), None)
    reference_delta = reference["delta_E_high_minus_low_eV"] if reference else None
    tolerance = next(iter(batch["jobs"]))["acceptance"]["pair_delta_E_tolerance_eV"]
    for row in results:
        difference = (
            row["delta_E_high_minus_low_eV"] - reference_delta
            if row["delta_E_high_minus_low_eV"] is not None and reference_delta is not None
            else None
        )
        row["delta_E_difference_from_reference_eV"] = difference
        row["passes_tolerance"] = abs(difference) <= tolerance if difference is not None else None
    gate, gate_reason = evaluate_gate(results, args.reference_profile, tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_profile": args.reference_profile,
        "pair_delta_E_tolerance_eV": tolerance,
        "results": results,
        "gate": gate,
        "gate_reason": gate_reason,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
