#!/usr/bin/env python3
"""Compare a frozen prospective site-proposer DFT validation batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from validate_runtime_provenance import validate_runtime_provenance


FINAL_CLASSIFICATIONS = {
    "accepted_local_minimum",
    "relax_done_force_gate_failed",
    "terminal_without_ionic_step",
}
RESTARTABLE_CLASSIFICATIONS = {
    "clean_max_seconds_restartable",
    "clean_max_ionic_steps_restartable",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance_gate(local_dir: Path, expected_input_hash: str, expected_job_id: str) -> dict:
    input_path = local_dir / "relax.in"
    runtime_path = local_dir / "runtime_provenance.json"
    observed_input_hash = sha256_file(input_path) if input_path.is_file() else None
    runtime = validate_runtime_provenance(
        runtime_path, {"relax_input": expected_input_hash}, ["pw.x"], expected_job_id
    )
    passed = observed_input_hash == expected_input_hash and runtime["passed"]
    return {
        "passed": passed,
        "expected_input_sha256": expected_input_hash,
        "observed_input_sha256": observed_input_hash,
        "runtime_input_sha256": runtime["observed_input_hashes"]["relax_input"],
        "runtime_provenance": str(runtime_path),
        "runtime_provenance_sha256": runtime["sha256"],
        "runtime_validation": runtime,
    }


def outcome(job: dict, status: dict, endpoint_acceptance: Optional[dict]) -> dict:
    classification = status["classification"]
    provenance = provenance_gate(
        Path(status["local_dir"]), job["relax_input_sha256"], str(status["job_id"])
    )
    if classification in RESTARTABLE_CLASSIFICATIONS:
        result = "restart_required"
    elif classification not in FINAL_CLASSIFICATIONS:
        result = "pending_or_needs_review"
    elif classification == "accepted_local_minimum" and provenance["passed"] and endpoint_acceptance is None:
        result = "repeat_required"
    elif classification == "accepted_local_minimum" and provenance["passed"]:
        result = (
            "accepted_repeat_validated_local_minimum"
            if endpoint_acceptance["status"] == "accepted"
            else "rejected_repeat_validation"
        )
    elif classification == "accepted_local_minimum":
        result = "provenance_gate_failed"
    else:
        result = "rejected_local_minimum"
    parsed = status.get("parsed", {})
    return {
        "arm": job["arm"],
        "pair_index": int(job["pair_index"]),
        "raw_site_id": job["raw_site_id"],
        "job_id": status["job_id"],
        "classification": classification,
        "result": result,
        "accepted": result == "accepted_repeat_validated_local_minimum",
        "lineage_total_ionic_steps": parsed.get("lineage_total_ionic_steps"),
        "lineage_total_scf_iterations": parsed.get("lineage_total_scf_iterations"),
        "final_max_atom_force_eV_A": parsed.get("final_max_atom_force_eV_A"),
        "final_energy_eV": parsed.get("final_energy_eV"),
        "provenance": provenance,
        "repeat_acceptance_status": endpoint_acceptance.get("status") if endpoint_acceptance else None,
        "repeat_job_id": endpoint_acceptance.get("repeat_job_id") if endpoint_acceptance else None,
        "repeat_failed_checks": endpoint_acceptance.get("failed_checks", []) if endpoint_acceptance else [],
    }


def exact_paired_binomial_pvalue(proposal_only: int, control_only: int) -> float | None:
    discordant = proposal_only + control_only
    if not discordant:
        return None
    smaller = min(proposal_only, control_only)
    tail = sum(math.comb(discordant, value) for value in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_yield_metrics(pairs: list[dict], finalized: bool) -> dict:
    counts = {key: 0 for key in ("both_accepted", "proposal_only", "control_only", "neither_accepted")}
    for pair in pairs:
        proposal = pair["proposal"]["accepted"]
        control = pair["control"]["accepted"]
        key = (
            "both_accepted" if proposal and control else
            "proposal_only" if proposal else
            "control_only" if control else
            "neither_accepted"
        )
        counts[key] += 1
    if not finalized:
        return {**counts, "proposal_yield": None, "control_yield": None, "paired_yield_difference": None, "exact_paired_pvalue": None}
    total = len(pairs)
    proposal_accepted = counts["both_accepted"] + counts["proposal_only"]
    control_accepted = counts["both_accepted"] + counts["control_only"]
    return {
        **counts,
        "proposal_yield": proposal_accepted / total,
        "control_yield": control_accepted / total,
        "paired_yield_difference": (proposal_accepted - control_accepted) / total,
        "exact_paired_pvalue": exact_paired_binomial_pvalue(counts["proposal_only"], counts["control_only"]),
    }


def compare(args) -> tuple[Path, dict]:
    batch = json.loads(args.batch_manifest.read_text())
    status = json.loads(args.status.read_text())
    audit = json.loads(args.calculator_audit.read_text())
    if audit["counts"] != {
        "inputs": len(batch["jobs"]),
        "parse_errors": 0,
        "calculator_identities": 1,
        "incomplete_identities": 0,
        "soc_enabled_inputs": 0,
        "spin_polarized_inputs": len(batch["jobs"]),
    }:
        raise ValueError("Calculator audit does not accept the complete batch")
    status_index = {row["branch_id"]: row for row in status["jobs"]}
    if set(status_index) != {row["branch_id"] for row in batch["jobs"]}:
        raise ValueError("Status jobs do not exactly match the frozen batch")
    acceptance_path = getattr(args, "endpoint_acceptance", None)
    acceptance_payload = json.loads(acceptance_path.read_text()) if acceptance_path else None
    acceptance_index = {}
    if acceptance_payload:
        if acceptance_payload.get("first_status_sha256") != sha256_file(args.status):
            raise ValueError("Endpoint acceptance is not bound to this first-relax status")
        for record in acceptance_payload.get("endpoint_records", []):
            key = (record["path_id"], int(record["image_index_qe"]))
            if key in acceptance_index:
                raise ValueError(f"Duplicate endpoint acceptance record: {key}")
            if record.get("status") == "accepted" and not all(
                record.get("checks", {}).get(check)
                for check in (
                    "parent_runtime_provenance",
                    "repeat_runtime_provenance",
                    "calculator_identity_unchanged",
                )
            ):
                raise ValueError(f"Accepted endpoint record lacks strict provenance checks: {key}")
            acceptance_index[key] = record
    expected_keys = {
        (job["source_path_id"], int(job["source_image_index_qe"]))
        for job in batch["jobs"]
        if status_index[job["branch_id"]]["classification"] == "accepted_local_minimum"
    }
    unknown_acceptance = sorted(set(acceptance_index) - expected_keys)
    if unknown_acceptance:
        raise ValueError(f"Endpoint acceptance contains rows outside accepted first relaxations: {unknown_acceptance}")
    job_by_key = {
        (job["source_path_id"], int(job["source_image_index_qe"])): job
        for job in batch["jobs"]
    }
    for key, record in acceptance_index.items():
        first_row = status_index[job_by_key[key]["branch_id"]]
        if str(record.get("parent_job_id")) != str(first_row["job_id"]):
            raise ValueError(f"Endpoint acceptance parent job mismatch: {key}")
        if record.get("calculator_identity") != audit["profiles"][0]["calculator_identity"]:
            raise ValueError(f"Endpoint acceptance calculator mismatch: {key}")
    rows = [
        outcome(
            job,
            status_index[job["branch_id"]],
            acceptance_index.get((job["source_path_id"], int(job["source_image_index_qe"]))),
        )
        for job in batch["jobs"]
    ]
    by_pair = {}
    for row in rows:
        by_pair.setdefault(row["pair_index"], {})[row["arm"]] = row
    pairs = []
    for pair in batch["pairs"]:
        arms = by_pair[int(pair["pair_index"])]
        if set(arms) != {"proposal", "control"}:
            raise ValueError(f"Incomplete arms for pair {pair['pair_index']}")
        pairs.append({**pair, "proposal": arms["proposal"], "control": arms["control"]})
    finalized = all(row["result"] in {
        "accepted_repeat_validated_local_minimum",
        "rejected_repeat_validation",
        "rejected_local_minimum",
        "provenance_gate_failed",
    } for row in rows)
    metrics = paired_yield_metrics(pairs, finalized)
    result = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "prospective_site_proposer_endpoint_yield_comparison",
        "batch_manifest": str(args.batch_manifest.resolve()),
        "batch_manifest_sha256": sha256_file(args.batch_manifest),
        "status": str(args.status.resolve()),
        "status_sha256": sha256_file(args.status),
        "endpoint_acceptance": str(acceptance_path.resolve()) if acceptance_path else None,
        "endpoint_acceptance_sha256": sha256_file(acceptance_path) if acceptance_path else None,
        "calculator_audit": str(args.calculator_audit.resolve()),
        "calculator_audit_sha256": sha256_file(args.calculator_audit),
        "calculator_identity": audit["profiles"][0]["calculator_identity"],
        "all_pairs_finalized": finalized,
        "formal_result_available": finalized,
        "same_structure_speedup_claim_allowed": False,
        "metrics": metrics,
        "pairs": pairs,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "prospective_dft_validation_comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    with (args.out_dir / "prospective_dft_validation_rows.csv").open("w", newline="") as handle:
        fields = [
            "pair_index", "arm", "raw_site_id", "job_id", "classification", "result",
            "accepted", "lineage_total_ionic_steps", "lineage_total_scf_iterations",
            "final_max_atom_force_eV_A", "final_energy_eV", "repeat_acceptance_status",
            "repeat_job_id",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    return output, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument(
        "--endpoint-acceptance", type=Path,
        help="Repeat-relax endpoint_acceptance_batch.json; required before first-relax successes can be final",
    )
    parser.add_argument("--calculator-audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    output, result = compare(args)
    print(json.dumps({"output": str(output), "formal_result_available": result["formal_result_available"]}, indent=2))


if __name__ == "__main__":
    main()
