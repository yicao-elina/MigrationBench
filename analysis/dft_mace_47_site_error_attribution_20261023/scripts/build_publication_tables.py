#!/usr/bin/env python3
"""Build compact publication tables from the audited endpoint-47 data package."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "processed"
T = ROOT / "tables"


def main() -> None:
    endpoint = pd.read_csv(P / "DFT_MACE47_INTEGRATED_ENDPOINT_TABLE.tsv", sep="\t")
    roles = pd.read_csv(P / "ENDPOINT_PATH_ROLES.tsv", sep="\t")
    four = pd.read_csv(P / "MACE_FOUR_MODEL_ENDPOINT_MODEL_SUMMARY.tsv", sep="\t")
    assoc = pd.read_csv(P / "MACE12_SOURCE_GEOMETRY_ASSOCIATIONS.tsv", sep="\t")

    site_rows = []
    for region, group in endpoint.groupby("vertical_region_geometric", sort=False):
        converged = group.loc[group["relaxation_class"].eq("DFT_LOCAL_MINIMUM_CANDIDATE")]
        site_rows.append(
            {
                "geometric_region": region,
                "n_endpoint_sites": len(group),
                "n_retry21_BFGS": len(converged),
                "retry21_deltaE_median_eV": converged["dft_relative_energy_eV_retry21"].median(),
                "retry21_deltaE_min_eV": converged["dft_relative_energy_eV_retry21"].min(),
                "retry21_deltaE_max_eV": converged["dft_relative_energy_eV_retry21"].max(),
                "median_12model_force_vector_SD_eV_A": group["ensemble_cr_force_vector_std_rms_eV_A"].median(),
                "interpretation": "geometric region; not thermodynamic stability class",
            }
        )
    pd.DataFrame(site_rows).to_csv(
        T / "TABLE1_DFT47_GEOMETRY_ENERGY_AND_ENSEMBLE_SUMMARY.tsv",
        sep="\t",
        index=False,
        float_format="%.6g",
    )

    route = (
        roles.groupby("route_family")["endpoint"]
        .nunique()
        .rename("n_unique_endpoints_with_membership")
        .reset_index()
    )
    route["membership_note"] = "Endpoints may belong to multiple path families; counts are not additive."
    route.to_csv(T / "TABLE_S1_ROUTE_MEMBERSHIP.tsv", sep="\t", index=False)

    four_table = (
        four.groupby("model_label_interpreted")
        .agg(
            expected_starts=("mace_expected_starts", "sum"),
            successful_starts=("mace_successful_starts", "sum"),
            failed_starts=("mace_failure_starts", "sum"),
            median_fmax_eV_A=("mace_median_final_fmax_eV_A", "median"),
            median_three_start_energy_span_eV=("mace_within_model_energy_span_eV", "median"),
            max_three_start_energy_span_eV=("mace_within_model_energy_span_eV", "max"),
            median_final_Cr_span_A=("mace_within_model_final_cr_span_A", "median"),
        )
        .reset_index()
    )
    four_table["scope"] = "model-internal three-start basin sensitivity; not a barrier or DFT error"
    four_table.to_csv(T / "TABLE2_LEGACY_FOUR_MODEL_SENSITIVITY.tsv", sep="\t", index=False, float_format="%.6g")

    keep = [
        ("ensemble_cr_force_vector_std_rms_eV_A", "cr_to_vdw_gap_center_A"),
        ("ensemble_cr_force_vector_std_rms_eV_A", "n_Sb_within_3p2_A"),
        ("ensemble_cr_force_vector_std_rms_eV_A", "n_Te_within_3p2_A"),
        ("ensemble_cr_force_vector_std_rms_eV_A", "nearest_Te_A"),
        ("ensemble_force_direction_disagreement", "n_Sb_within_3p2_A"),
        ("ensemble_relative_energy_rank_sd", "within-gap lateral / same environment"),
    ]
    selected = pd.concat(
        [
            assoc.loc[assoc["outcome"].eq(outcome) & assoc["feature"].eq(feature)]
            for outcome, feature in keep
        ],
        ignore_index=True,
    )
    selected["scope"] = "exploratory model-disagreement associations; not DFT-error calibration"
    selected.to_csv(T / "TABLE3_MACE12_STRUCTURAL_ASSOCIATIONS.tsv", sep="\t", index=False, float_format="%.6g")
    print("Wrote publication tables to", T)


if __name__ == "__main__":
    main()
