#!/usr/bin/env python3
"""Validate and summarize the accepted-model 639-candidate screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_ORDER = [
    "stable_early_control",
    "stable_pre_onset",
    "precursor_near_onset",
    "threshold_or_terminal",
    "terminal_stable_target",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantile(values: pd.Series, q: float) -> float:
    numbers = pd.to_numeric(values, errors="coerce").dropna()
    return float(numbers.quantile(q)) if len(numbers) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analysis = Path(args.analysis)
    manifest_path = Path(args.manifest)
    models_path = Path(args.models)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(analysis / "mlip_candidate_results_all.csv", dtype={"candidate_id": str, "model_id": str})
    candidates = pd.read_csv(manifest_path, dtype=str).fillna("")
    models = pd.read_csv(models_path, sep="\t", dtype=str).fillna("")
    expected_pairs = len(candidates) * len(models)
    pair_counts = results.groupby(["candidate_id", "model_id"], dropna=False).size()
    duplicate_pairs = int((pair_counts > 1).sum())
    expected_ids = set(candidates["candidate_id"].astype(str))
    expected_models = set(models["model_id"].astype(str))
    observed_ids = set(results["candidate_id"].astype(str))
    observed_models = set(results["model_id"].astype(str))
    finite_rows = (
        results["evaluation_status"].eq("success")
        & pd.to_numeric(results["energy_eV"], errors="coerce").map(np.isfinite)
        & pd.to_numeric(results["max_force_eV_A"], errors="coerce").map(np.isfinite)
        & pd.to_numeric(results["rms_force_eV_A"], errors="coerce").map(np.isfinite)
    )
    status = "VERIFIED" if (
        len(results) == expected_pairs
        and duplicate_pairs == 0
        and observed_ids == expected_ids
        and observed_models == expected_models
        and bool(finite_rows.all())
    ) else "PROVENANCE_PARTIAL"

    results["max_force_eV_A"] = pd.to_numeric(results["max_force_eV_A"], errors="coerce")
    results["rms_force_eV_A"] = pd.to_numeric(results["rms_force_eV_A"], errors="coerce")
    results["energy_per_atom_eV"] = pd.to_numeric(results["energy_per_atom_eV"], errors="coerce")
    success = results[results["evaluation_status"].eq("success")].copy()

    summary_rows = []
    for keys, group in success.groupby(["candidate_group", "window", "model_id", "model_family"], dropna=False):
        candidate_group, window, model_id, model_family = keys
        summary_rows.append(
            {
                "candidate_group": candidate_group,
                "window": window,
                "model_id": model_id,
                "model_family": model_family,
                "training_seed": group["training_seed"].iloc[0],
                "n_candidates": len(group),
                "max_force_median_eV_A": quantile(group["max_force_eV_A"], 0.50),
                "max_force_q95_eV_A": quantile(group["max_force_eV_A"], 0.95),
                "max_force_max_eV_A": float(group["max_force_eV_A"].max()),
                "rms_force_median_eV_A": quantile(group["rms_force_eV_A"], 0.50),
                "energy_per_atom_median_eV": quantile(group["energy_per_atom_eV"], 0.50),
            }
        )
    model_stage = pd.DataFrame(summary_rows).sort_values(["candidate_group", "window", "model_family", "model_id"])
    model_stage.to_csv(output / "canonical_model_stage_summary.tsv", sep="\t", index=False)

    dispersion_rows = []
    non_foundation = models[~models["model_family"].eq("foundation_zero_shot")]
    for family, family_models in non_foundation.groupby("model_family"):
        family_ids = set(family_models["model_id"])
        family_results = success[success["model_id"].isin(family_ids)]
        for (candidate_group, window), group in family_results.groupby(["candidate_group", "window"], dropna=False):
            pivot = group.pivot_table(index="candidate_id", columns="training_seed", values="max_force_eV_A", aggfunc="first")
            seed_counts = pivot.notna().sum(axis=1)
            eligible = pivot.loc[seed_counts == len(family_ids)]
            if eligible.empty:
                continue
            values = eligible.to_numpy(float)
            dispersion_rows.append(
                {
                    "model_family": family,
                    "candidate_group": candidate_group,
                    "window": window,
                    "n_candidates_all_seeds": len(eligible),
                    "seed_force_std_median_eV_A": float(np.median(np.std(values, axis=1, ddof=0))),
                    "seed_force_std_q95_eV_A": float(np.quantile(np.std(values, axis=1, ddof=0), 0.95)),
                    "seed_force_range_median_eV_A": float(np.median(np.ptp(values, axis=1))),
                    "seed_force_range_q95_eV_A": float(np.quantile(np.ptp(values, axis=1), 0.95)),
                }
            )
    seed_dispersion = pd.DataFrame(dispersion_rows).sort_values(["model_family", "candidate_group", "window"])
    seed_dispersion.to_csv(output / "canonical_within_family_seed_dispersion.tsv", sep="\t", index=False)

    group_counts = success.groupby(["candidate_group", "window"], dropna=False).agg(
        candidate_count=("candidate_id", "nunique"), model_rows=("candidate_id", "size")
    ).reset_index()
    group_counts.to_csv(output / "canonical_candidate_group_counts.tsv", sep="\t", index=False)

    checks = {
        "schema": "migrationbench_canonical_639_analysis_v1",
        "status": status,
        "analysis_root": str(analysis),
        "analysis_manifest_sha256": sha256(analysis / "analysis_manifest.json"),
        "candidate_manifest": str(manifest_path),
        "candidate_manifest_sha256": sha256(manifest_path),
        "model_registry": str(models_path),
        "model_registry_sha256": sha256(models_path),
        "n_candidates_expected": len(candidates),
        "n_models_expected": len(models),
        "n_rows_expected": expected_pairs,
        "n_rows_observed": len(results),
        "n_success_rows": int(finite_rows.sum()),
        "n_nonfinite_or_failed_rows": int((~finite_rows).sum()),
        "duplicate_candidate_model_pairs": duplicate_pairs,
        "candidate_ids_missing": sorted(expected_ids - observed_ids),
        "unexpected_candidate_ids": sorted(observed_ids - expected_ids),
        "model_ids_missing": sorted(expected_models - observed_models),
        "unexpected_model_ids": sorted(observed_models - expected_models),
        "source_stage_groups_observed": sorted(success["candidate_group"].dropna().unique().tolist()),
        "required_stage_groups": GROUP_ORDER,
        "historical_heterogeneous_models_excluded_from_seed_dispersion": True,
        "interpretation_boundary": "paired static MLIP evidence; not DFT correctness, calibrated uncertainty, causal proof, or transport evidence",
    }
    (output / "CANONICAL_ANALYSIS_MANIFEST.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({key: checks[key] for key in ["status", "n_rows_expected", "n_rows_observed", "n_success_rows", "n_nonfinite_or_failed_rows", "duplicate_candidate_model_pairs"]}, indent=2))
    if status != "VERIFIED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
