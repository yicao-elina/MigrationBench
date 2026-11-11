#!/usr/bin/env python3
"""Audit grain, completeness, validity, and coverage of the 639 MLIP table."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CORE_COLUMNS = [
    "candidate_id", "model_id", "model_family", "evaluation_status",
    "energy_eV", "forces_finite", "max_force_eV_A", "rms_force_eV_A",
    "candidate_group", "window", "input_sha256", "model_sha256",
    "model_path", "natoms", "species", "pbc",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_health(frame: pd.DataFrame, column: str) -> dict[str, object]:
    values = pd.to_numeric(frame[column], errors="coerce")
    return {
        "null": int(values.isna().sum()),
        "nonfinite": int((~np.isfinite(values.dropna())).sum()),
        "min": float(values.min()) if values.notna().any() else None,
        "max": float(values.max()) if values.notna().any() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--categorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = pd.read_csv(args.results)
    manifest = pd.read_csv(args.candidate_manifest)
    categorization = pd.read_csv(args.categorization)
    args.output.mkdir(parents=True, exist_ok=True)

    result_ids = set(results["candidate_id"].astype(str))
    manifest_ids = set(manifest["candidate_id"].astype(str))
    pair_counts = results.groupby(["candidate_id", "model_id"], dropna=False).size()
    candidate_counts = results.groupby("candidate_id", dropna=False).size()
    core_missing = {
        column: int(results[column].isna().sum())
        for column in CORE_COLUMNS
        if column in results
    }
    numeric = {
        column: numeric_health(results, column)
        for column in ["energy_eV", "energy_per_atom_eV", "max_force_eV_A", "rms_force_eV_A", "p95_force_eV_A", "p99_force_eV_A"]
        if column in results
    }
    duplicate_lineage_columns = {
        column: int(results[column].isna().sum())
        for column in results.columns
        if column.endswith("_x") or column.endswith("_y") or column in {
            "source_trajectory", "source_run_key", "source_onset_ps", "source_trigger"
        }
    }
    stage_counts = (
        manifest.groupby(["candidate_group", "window"], dropna=False)["candidate_id"]
        .nunique().sort_index().to_dict()
    )
    result_stage_counts = (
        results.groupby(["candidate_group", "window"], dropna=False)["candidate_id"]
        .nunique().sort_index().to_dict()
    )
    checks = {
        "status": "VERIFIED",
        "results_sha256": sha256(args.results),
        "candidate_manifest_sha256": sha256(args.candidate_manifest),
        "categorization_sha256": sha256(args.categorization),
        "results_rows": int(len(results)),
        "results_columns": int(len(results.columns)),
        "candidate_count": int(results["candidate_id"].nunique()),
        "manifest_candidate_count": int(manifest["candidate_id"].nunique()),
        "categorization_candidate_count": int(categorization["candidate_id"].nunique()),
        "model_count": int(results["model_id"].nunique()),
        "model_candidate_counts": {str(k): int(v) for k, v in results.groupby("model_id")["candidate_id"].nunique().items()},
        "candidate_rows_per_model_counts": {str(k): int(v) for k, v in candidate_counts.value_counts().items()},
        "duplicate_candidate_model_pairs": int((pair_counts > 1).sum()),
        "duplicate_rows_exact": int(results.duplicated().sum()),
        "missing_candidate_ids_in_results": sorted(manifest_ids - result_ids),
        "unexpected_candidate_ids_in_results": sorted(result_ids - manifest_ids),
        "core_missing": core_missing,
        "numeric_health": numeric,
        "success_rows": int((results["evaluation_status"] == "success").sum()),
        "failed_rows": int((results["evaluation_status"] != "success").sum()),
        "finite_force_rows": int(results["forces_finite"].astype(bool).sum()),
        "manifest_stage_counts": {str(k): int(v) for k, v in stage_counts.items()},
        "result_stage_counts": {str(k): int(v) for k, v in result_stage_counts.items()},
        "model_family_counts": {str(k): int(v) for k, v in results.groupby("model_family")["model_id"].nunique().items()},
        "duplicate_lineage_columns_null_counts": duplicate_lineage_columns,
        "schema_warning": "The reconciled table contains merge-carried duplicate lineage columns with all-null or partially-null values; canonical candidate/model/stage columns are complete and should be used for analysis.",
    }
    hard_fail = [
        checks["candidate_count"] != 639,
        checks["manifest_candidate_count"] != 639,
        checks["categorization_candidate_count"] != 639,
        checks["results_rows"] != checks["candidate_count"] * checks["model_count"],
        checks["duplicate_candidate_model_pairs"] != 0,
        bool(checks["missing_candidate_ids_in_results"]),
        bool(checks["unexpected_candidate_ids_in_results"]),
        checks["failed_rows"] != 0,
        any(v for v in core_missing.values()),
        any(v["nonfinite"] for v in numeric.values()),
    ]
    if any(hard_fail):
        checks["status"] = "PROVENANCE_PARTIAL"
    (args.output / "DATA_QUALITY_AUDIT.json").write_text(json.dumps(checks, indent=2, default=str) + "\n", encoding="utf-8")

    lines = [
        "# 639 MLIP result data-quality audit",
        "",
        f"Status: `{checks['status']}`. This audit checks the reconciled static-evaluation table before canonical-family merging.",
        "",
        "## Dataset and grain",
        "",
        f"The intended grain is one row per candidate/model pair. The table has {checks['results_rows']} rows, {checks['candidate_count']} candidates, and {checks['model_count']} model IDs.",
        f"Candidate manifest and categorization each contain {checks['manifest_candidate_count']} and {checks['categorization_candidate_count']} unique candidates, respectively.",
        "",
        "## Checks",
        "",
        f"- Duplicate candidate/model pairs: {checks['duplicate_candidate_model_pairs']}",
        f"- Exact duplicate rows: {checks['duplicate_rows_exact']}",
        f"- Successful rows: {checks['success_rows']}; failed rows: {checks['failed_rows']}",
        f"- Finite force rows: {checks['finite_force_rows']}",
        f"- Missing candidate IDs: {len(checks['missing_candidate_ids_in_results'])}; unexpected IDs: {len(checks['unexpected_candidate_ids_in_results'])}",
        "",
        "## Findings and risk",
        "",
        "The canonical candidate/model grain is complete for the historical 639×3 table when the status is VERIFIED. The wide reconciled CSV also retains merge-carried `_x`/`_y` and legacy source columns; these are schema contamination, not evidence of missing core evaluations. Downstream analysis must use the canonical candidate_id, model_id, candidate_group, window, energy_eV, force metrics, and evaluation_status columns, and the issue must be repaired or explicitly carried into the final manifest.",
        "",
        "The 639 candidates are mechanism-focused, event-driven diagnostics rather than an unbiased random test set. Their stage labels come from the candidate manifest and must not be reinterpreted as newly observed DFT onset labels.",
        "",
        "## Reproduction",
        "",
        f"`python {Path(__file__).resolve()} --results {args.results} --candidate-manifest {args.candidate_manifest} --categorization {args.categorization} --output {args.output}`",
        f"Script SHA256: `{sha256(Path(__file__))}`",
    ]
    (args.output / "DATA_QUALITY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: checks[k] for k in ("status", "results_rows", "candidate_count", "model_count", "duplicate_candidate_model_pairs", "failed_rows")}, indent=2))


if __name__ == "__main__":
    main()
