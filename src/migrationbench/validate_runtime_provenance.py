#!/usr/bin/env python3
"""Validate hash-bound runtime provenance for a scientific calculation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_runtime_provenance(
    path: Path,
    expected_inputs: dict[str, str],
    required_binaries: list[str],
    expected_job_id: str | None = None,
) -> dict:
    path = Path(path).resolve()
    payload = json.loads(path.read_text()) if path.is_file() else {}
    checks = {"runtime_provenance_exists": path.is_file()}
    observed_inputs = payload.get("inputs", {})
    for name, expected_hash in sorted(expected_inputs.items()):
        record = observed_inputs.get(name, {})
        checks[f"input_{name}_exists"] = bool(record.get("exists"))
        checks[f"input_{name}_hash_matches"] = record.get("sha256") == expected_hash
    binaries = payload.get("binaries", {})
    for name in sorted(required_binaries):
        record = binaries.get(name, {})
        checks[f"binary_{name}_exists"] = bool(record.get("exists"))
        checks[f"binary_{name}_hash_recorded"] = bool(record.get("sha256"))
    observed_job_id = payload.get("environment", {}).get("SLURM_JOB_ID")
    if expected_job_id is not None:
        checks["slurm_job_id_matches"] = str(observed_job_id) == str(expected_job_id)
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
        "path": str(path),
        "sha256": sha256_file(path) if path.is_file() else None,
        "observed_slurm_job_id": observed_job_id,
        "observed_input_hashes": {
            name: observed_inputs.get(name, {}).get("sha256") for name in sorted(expected_inputs)
        },
        "required_input_hashes": expected_inputs,
        "required_binaries": sorted(required_binaries),
    }
