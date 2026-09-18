#!/usr/bin/env python3
"""Compare an unrestrained MACE NEB with its reference-tethered control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def status_row(path: Path, job_id: str) -> dict:
    payload = json.loads(path.read_text())
    matches = [row for row in payload["jobs"] if str(row["job_id"]) == str(job_id)]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for job {job_id} in {path}; found {len(matches)}")
    return matches[0]


def manifest(row: dict) -> tuple[dict, Path]:
    path = Path(row["manifest"])
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text()), path


def physical_identity(payload: dict, spring: float | None = None) -> dict:
    return {
        "input_images_sha256": payload.get("input_images_sha256"),
        "source_candidate_manifest_sha256": payload.get("source_candidate_manifest_sha256"),
        "calculator": payload.get("calculator"),
        "model_path_sha256": payload.get("model_path_sha256"),
        "seed": payload.get("seed"),
        "n_images": payload.get("n_images"),
        "fmax_target_eV_A": payload.get("fmax_target_eV_A"),
        "spring_constant_eV_A2": spring,
    }


def stopping_budget_comparison(baseline: dict, trust: dict) -> dict:
    """Separate a numerical stopping budget from the physical Hamiltonian."""
    baseline_budget = int(baseline.get("steps_requested", -1))
    trust_budget = int(trust.get("steps_requested", -1))
    common_budget = min(baseline_budget, trust_budget)
    baseline_steps = int(baseline.get("optimizer_steps_completed", -1))
    trust_steps = int(trust.get("optimizer_steps_completed", -1))
    baseline_natural_stop = baseline.get("optimizer_converged") is True
    trust_natural_stop = trust.get(
        "optimizer_converged_under_optimization_potential",
        trust.get("optimizer_converged"),
    ) is True
    comparable = (
        common_budget > 0
        and baseline_natural_stop
        and trust_natural_stop
        and baseline_steps < common_budget
        and trust_steps < common_budget
    )
    return {
        "baseline_steps_requested": baseline_budget,
        "trust_steps_requested": trust_budget,
        "common_budget_steps": common_budget,
        "baseline_steps_completed": baseline_steps,
        "trust_steps_completed": trust_steps,
        "baseline_natural_stop": baseline_natural_stop,
        "trust_natural_stop": trust_natural_stop,
        "comparable_before_common_budget": comparable,
    }


def history_spring(manifest_path: Path) -> float:
    history_path = manifest_path.with_name("mlff_neb_iteration_history_manifest.json")
    history = json.loads(history_path.read_text())
    return float(history["spring_constant_eV_A2"])


def concise_metrics(row: dict, payload: dict) -> dict:
    mechanism = row.get("mechanism_metrics", {})
    geometry = mechanism.get("final_geometry", {})
    return {
        "classification": row.get("classification"),
        "optimizer_steps_completed": payload.get("optimizer_steps_completed"),
        "optimizer_converged_base_neb": payload.get("optimizer_converged"),
        "optimizer_converged_optimization_potential": payload.get(
            "optimizer_converged_under_optimization_potential",
            payload.get("optimizer_converged"),
        ),
        "base_barrier_proxy_eV": payload.get("barrier_proxy_eV"),
        "max_abs_internal_base_energy_change_eV": mechanism.get(
            "max_abs_internal_relaxation_energy_delta_eV"
        ),
        "migrant_internal_coordinate_rms_A": mechanism.get(
            "migrant_internal_coordinate_rms_A"
        ),
        "all_atom_internal_coordinate_rms_A": mechanism.get(
            "all_atom_internal_coordinate_rms_A"
        ),
        "final_min_pair_distance_A": geometry.get("min_pair_distance_A"),
        "final_max_migrant_step_A": geometry.get("max_migrant_step_A"),
        "final_total_migrant_path_A": geometry.get("total_migrant_path_A"),
        "checks": row.get("checks", {}),
    }


def candidate_evidence(path: Path, trust_manifest: dict) -> dict:
    payload = json.loads(path.read_text())
    manifest_hash_match = (
        trust_manifest.get("source_candidate_manifest_sha256") == sha256_file(path)
    )
    matches = [
        row for row in payload.get("candidates", [])
        if row.get("images_sha256") == trust_manifest.get("input_images_sha256")
    ]
    row = matches[0] if len(matches) == 1 else {}
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "manifest_hash_match": manifest_hash_match,
        "unique_input_row_match": len(matches) == 1,
        "endpoint_status": payload.get("endpoint_status"),
        "endpoints_accepted": payload.get("endpoint_status") == "accepted_local_minima",
        "candidate_production_eligible": row.get("production_eligible_before_mace") is True,
        "candidate_branch_id": row.get("branch_id"),
    }


def comparison_decision(
    identity_match: bool,
    trust_row: dict,
    trust_manifest: dict,
    candidate: dict,
    stopping_budget_comparable: bool = True,
) -> str:
    if not identity_match:
        return "invalid_physical_identity_mismatch"
    if not stopping_budget_comparable:
        return "invalid_stopping_budget_mismatch_or_censoring"
    if trust_manifest.get("reference_tether", {}).get("enabled") is not True:
        return "invalid_missing_reference_tether"
    if trust_row.get("checks", {}).get("dual_potential_history") is not True:
        return "invalid_incomplete_dual_potential_history"
    if not candidate.get("manifest_hash_match") or not candidate.get("unique_input_row_match"):
        return "invalid_candidate_provenance_binding"
    if trust_row.get("classification") == "ready_for_dft_preconditioner_review":
        if candidate.get("endpoints_accepted") and candidate.get("candidate_production_eligible"):
            return "trust_region_candidate_ready_for_dft_ab_design"
        return "trust_region_diagnostic_ready_as_curvature_donor_requires_endpoint_remap"
    return "trust_region_candidate_rejected"


def compare(args: argparse.Namespace) -> dict:
    baseline_row = status_row(args.baseline_status, args.baseline_job_id)
    trust_row = status_row(args.trust_status, args.trust_job_id)
    baseline, baseline_path = manifest(baseline_row)
    trust, trust_path = manifest(trust_row)
    candidate = candidate_evidence(args.source_candidate_manifest.resolve(), trust)
    baseline_identity = physical_identity(baseline, history_spring(baseline_path))
    trust_identity = physical_identity(trust, history_spring(trust_path))
    identity_match = baseline_identity == trust_identity
    budget = stopping_budget_comparison(baseline, trust)
    decision = comparison_decision(
        identity_match,
        trust_row,
        trust,
        candidate,
        budget["comparable_before_common_budget"],
    )
    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "path_id": args.path_id,
        "baseline_job_id": str(args.baseline_job_id),
        "trust_job_id": str(args.trust_job_id),
        "physical_identity_match": identity_match,
        "physical_identity": baseline_identity if identity_match else {
            "baseline": baseline_identity,
            "trust": trust_identity,
        },
        "stopping_budget_comparison": budget,
        "intended_experimental_difference": trust.get("reference_tether"),
        "source_candidate": candidate,
        "baseline": concise_metrics(baseline_row, baseline),
        "trust_region": concise_metrics(trust_row, trust),
        "decision": decision,
        "qe_handoff_allowed": False,
        "qe_handoff_reason": (
            "Historical unaccepted endpoints require a newly hash-bound endpoint remap. "
            "Only a trust run on that remapped production candidate may enter a paired QE design."
        ),
        "required_next_stage": (
            "generate_nonlinear_candidates_from_accepted_endpoints_then_rerun_mace"
            if decision == "trust_region_diagnostic_ready_as_curvature_donor_requires_endpoint_remap"
            else "paired_qe_design_gates" if decision == "trust_region_candidate_ready_for_dft_ab_design"
            else "reject_or_revise_preconditioner"
        ),
        "provenance": {
            "baseline_status": str(args.baseline_status.resolve()),
            "baseline_status_sha256": sha256_file(args.baseline_status),
            "trust_status": str(args.trust_status.resolve()),
            "trust_status_sha256": sha256_file(args.trust_status),
            "baseline_manifest": str(baseline_path.resolve()),
            "baseline_manifest_sha256": sha256_file(baseline_path),
            "trust_manifest": str(trust_path.resolve()),
            "trust_manifest_sha256": sha256_file(trust_path),
            "source_candidate_manifest": str(args.source_candidate_manifest.resolve()),
            "source_candidate_manifest_sha256": sha256_file(args.source_candidate_manifest),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-id", required=True)
    parser.add_argument("--baseline-status", type=Path, required=True)
    parser.add_argument("--trust-status", type=Path, required=True)
    parser.add_argument("--baseline-job-id", required=True)
    parser.add_argument("--trust-job-id", required=True)
    parser.add_argument("--source-candidate-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "trust_region_comparison.json"
    csv_path = args.out_dir / "trust_region_comparison.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    rows = []
    for variant in ("baseline", "trust_region"):
        row = {"variant": variant, **result[variant]}
        row["checks"] = json.dumps(row["checks"], sort_keys=True)
        rows.append(row)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(json_path), "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
