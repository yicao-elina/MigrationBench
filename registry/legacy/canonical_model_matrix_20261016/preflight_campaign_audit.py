#!/usr/bin/env python3
"""Audit canonical campaign structure without promoting scientific status."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output", default="PRELIMINARY_PREFLIGHT_AUDIT.json")
    args = parser.parse_args()
    root = Path(args.root)
    failures = []
    required = [
        "EXPERIMENT_MANIFEST.json",
        "MODEL_REGISTRY_CANONICAL_PRELIM.tsv",
        "ACCEPTANCE_MATRIX_PRELIMINARY.tsv",
        "DATASET_GROUP_AUDIT_20261016.json",
        "SCREEN_SOURCE_MANIFEST.json",
        "FINAL_RELEASE_CHECKLIST.md",
        "MODEL_GATE_SUBMISSION_32075596.json",
        "scratch_600k_grouped_v1/SUBMISSION_RECORD_32075474.json",
    ]
    missing = [path for path in required if not (root / path).is_file()]
    failures.extend(f"missing:{path}" for path in missing)
    registry = pd.read_csv(root / "MODEL_REGISTRY_CANONICAL_PRELIM.tsv", sep="\t", dtype=str).fillna("")
    acceptance = pd.read_csv(root / "ACCEPTANCE_MATRIX_PRELIMINARY.tsv", sep="\t", dtype=str).fillna("")
    if registry["canonical_model_id"].duplicated().any():
        failures.append("duplicate canonical_model_id")
    if acceptance["canonical_model_id"].duplicated().any():
        failures.append("duplicate acceptance canonical_model_id")
    if set(registry["canonical_model_id"]) != set(acceptance["canonical_model_id"]):
        failures.append("registry_acceptance_id_mismatch")
    scripts = sorted(root.glob("*.py"))
    compile_result = subprocess.run(["python", "-m", "py_compile", *map(str, scripts)], capture_output=True, text=True)
    if compile_result.returncode:
        failures.append("python_compile_failure")
    group_audit = json.loads((root / "DATASET_GROUP_AUDIT_20261016.json").read_text()) if not missing else {}
    if group_audit.get("status") != "VERIFIED":
        failures.append("dataset_group_audit_not_verified")
    source_manifest = json.loads((root / "SCREEN_SOURCE_MANIFEST.json").read_text()) if (root / "SCREEN_SOURCE_MANIFEST.json").is_file() else {}
    tool_hashes = {}
    for name, record in source_manifest.get("downstream_reproducibility_tools", {}).items():
        path = Path(record["path"])
        observed = sha256(path) if path.is_file() else "MISSING"
        tool_hashes[name] = {"path": str(path), "expected": record["sha256"], "observed": observed, "match": observed == record["sha256"]}
        if observed != record["sha256"]:
            failures.append(f"tool_hash_mismatch:{name}")
    pending = registry[registry["computational_status"].eq("PENDING")]["canonical_model_id"].tolist()
    payload = {
        "schema": "migrationbench_canonical_preflight_audit_v1",
        "status": "PRELIMINARY_PASS" if not failures else "PRELIMINARY_FAILURE",
        "scientific_acceptance": "NOT_YET_EVALUATED",
        "root": str(root),
        "required_file_count": len(required),
        "missing_required_files": missing,
        "model_count": len(registry),
        "acceptance_row_count": len(acceptance),
        "models_pending_training_or_gate": pending,
        "python_compile_status": "PASS" if compile_result.returncode == 0 else "FAIL",
        "downstream_tool_hashes": tool_hashes,
        "failure_reasons": failures,
        "boundary": "This preflight proves structural reproducibility only; it does not accept models, jobs, static outputs, DFT, or publication claims.",
    }
    output = root / args.output
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "model_count": len(registry), "pending_count": len(pending), "failure_reasons": failures}, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
