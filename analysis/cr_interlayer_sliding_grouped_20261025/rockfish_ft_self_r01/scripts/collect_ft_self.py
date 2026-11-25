#!/usr/bin/env python3
"""Collect the FT-600K / FT-MultiT interlayer self-relaxed campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_CAMPAIGN = "ft_interlayer_sliding_self_20261025_r01"
EXPECTED_SCHEMA = "ft_interlayer_self_v1"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def finite_profile(engine: object, expected_images: int) -> tuple[bool, str]:
    if not isinstance(engine, dict):
        return False, "missing_engine_manifest"
    profile = engine.get("profile")
    if not isinstance(profile, list) or len(profile) != expected_images:
        return False, "profile_length_mismatch"
    values: list[float] = []
    try:
        for item in profile:
            values.extend(float(item[key]) for key in ("energy_eV", "rel_energy_eV", "fmax_eV_A"))
        residual = float(engine["final_max_neb_residual_eV_A"])
        internal_force = float(engine["final_max_internal_neb_force_eV_A"])
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"profile_schema_error:{exc}"
    if not all(math.isfinite(value) for value in values + [residual, internal_force]):
        return False, "nonfinite_profile_or_residual"
    if residual > 0.05 or internal_force > 0.05:
        return False, f"residual_gate_failed:{residual}:{internal_force}"
    if engine.get("optimizer_converged") is not True:
        return False, "optimizer_not_converged"
    if engine.get("scientific_gate") is not True:
        return False, "engine_scientific_gate_failed"
    return True, ""


def candidate_manifests(root: Path, system: str, model_id: str) -> list[Path]:
    pattern = root / "runs" / "self" / system / model_id
    return sorted(pattern.glob("*/endpoint_only_manifest.json"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    jobs = read_tsv(root / "jobs" / "JOBS_SELF.tsv")
    registry = {row["model_id"]: row for row in read_tsv(root / "MODEL_REGISTRY.tsv")}
    helper = root / "code" / "self" / "export_mlff_neb_iteration_history.py"
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for job in jobs:
        key = (job["system"], job["model_id"])
        candidates = candidate_manifests(root, *key)
        manifest_path = candidates[-1] if candidates else None
        rec: dict[str, object] = {
            "record_id": job["record_id"],
            "system": job["system"],
            "model_id": job["model_id"],
            "manifest_path": str(manifest_path) if manifest_path else None,
            "manifest_sha256": sha256(manifest_path) if manifest_path else None,
            "candidate_count": len(candidates),
            "campaign_gate": False,
            "schema_gate": False,
            "model_sha_gate": False,
            "endpoint_gate": False,
            "runtime_gate": False,
            "profile_gate": False,
            "iteration_history_gate": False,
            "scientific_gate": False,
            "scientific_status": "MISSING",
            "publication_status": "CANDIDATE_ONLY",
            "failure_reason": "MISSING_MANIFEST" if not manifest_path else "",
        }
        reasons: list[str] = []
        if len(candidates) != 1:
            reasons.append("MISSING_MANIFEST" if not candidates else "DUPLICATE_ATTEMPTS")
        payload: dict[str, object] | None = None
        if manifest_path:
            try:
                payload = json.loads(manifest_path.read_text())
            except Exception as exc:
                reasons.append(f"MANIFEST_PARSE_ERROR:{type(exc).__name__}")
        if payload is not None:
            rec["campaign_gate"] = payload.get("campaign_id") == EXPECTED_CAMPAIGN
            rec["schema_gate"] = payload.get("schema_version") == EXPECTED_SCHEMA
            if not rec["campaign_gate"]:
                reasons.append("CAMPAIGN_ID_MISMATCH")
            if not rec["schema_gate"]:
                reasons.append("SCHEMA_VERSION_MISMATCH")
            expected_model = registry[job["model_id"]]
            rec["model_sha_gate"] = (
                payload.get("model_id") == job["model_id"]
                and payload.get("model_sha256_observed") == expected_model["model_sha256"]
                and payload.get("model_sha256_registry") == expected_model["model_sha256"]
            )
            if not rec["model_sha_gate"]:
                reasons.append("MODEL_SHA_MISMATCH")
            rec["endpoint_gate"] = (
                payload.get("endpoint_input_gate") == "PASS"
                and payload.get("endpoint_output_gate") == "PASS"
                and int(payload.get("atoms", -1)) == int(job["atoms"])
                and float(payload.get("endpoint_max_abs_displacement_A", math.inf)) <= 1.0e-8
            )
            if not rec["endpoint_gate"]:
                reasons.append("ENDPOINT_GATE_FAILED")
            rec["runtime_gate"] = payload.get("runtime_provenance_gate") == "PASS"
            if not rec["runtime_gate"]:
                reasons.append("RUNTIME_PROVENANCE_GATE_FAILED")
            profile_ok, profile_reason = finite_profile(payload.get("engine"), int(job["n_images"]))
            rec["profile_gate"] = profile_ok
            if not profile_ok:
                reasons.append(profile_reason)
            attempt_dir = manifest_path.parent
            history = attempt_dir / "mlff_neb_iteration_history.extxyz"
            history_csv = attempt_dir / "mlff_neb_iteration_history.csv"
            rec["iteration_history_gate"] = helper.is_file() and helper.stat().st_size > 0 and history.is_file() and history.stat().st_size > 0 and history_csv.is_file() and history_csv.stat().st_size > 0
            if not rec["iteration_history_gate"]:
                reasons.append("ITERATION_HISTORY_OR_HELPER_MISSING")
            rec["scientific_status"] = payload.get("scientific_status", "UNVERIFIED")
            rec["publication_status"] = payload.get("publication_status", "CANDIDATE_ONLY")
            rec["scientific_gate"] = payload.get("scientific_gate") == "PASS" and rec["scientific_status"] == "CONVERGED"
            if not rec["scientific_gate"]:
                reasons.append("SCIENTIFIC_GATE_FAILED")
        if reasons:
            rec["failure_reason"] = ";".join(dict.fromkeys(reasons))
            failures.append(rec)
        else:
            rec["failure_reason"] = ""
        records.append(rec)

    output = root / "FT_SELF_R01_COLLECT.tsv"
    failure = root / "FT_SELF_R01_FAILURE_LEDGER.tsv"
    fields = list(records[0]) if records else ["record_id"]
    for destination, payload in ((output, records), (failure, failures)):
        with destination.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(payload)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": EXPECTED_CAMPAIGN,
        "expected_count": len(jobs),
        "observed_count": sum(bool(item["manifest_path"]) for item in records),
        "gate_pass_count": len(records) - len(failures),
        "failure_or_missing_count": len(failures),
        "status": "PASS" if len(records) == 12 and not failures else "PARTIAL",
        "publication_status": "CANDIDATE_ONLY",
        "output_tsv": str(output),
        "output_tsv_sha256": sha256(output),
        "failure_ledger": str(failure),
        "failure_ledger_sha256": sha256(failure),
        "campaign_boundary": "FT-600K and FT-MultiT endpoint-only MLIP NEB proxy; not a DFT MEP barrier.",
    }
    (root / "FT_SELF_R01_COLLECT.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
