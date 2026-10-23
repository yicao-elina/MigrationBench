#!/usr/bin/env python3
"""Build the additive, provenance-aware DFT--MACE endpoint-47 analysis package.

The script intentionally separates (i) local DFT terminal evidence,
(ii) DFT-relative energies that share a confirmed protocol, and
(iii) MACE model-internal relaxation sensitivity.  It does not calculate a
MACE--DFT absolute-energy error from incompatible energy references or
different relaxed geometries.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read
from scipy.stats import mannwhitneyu, spearmanr


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
INPUT_ROOT = RAW / "rockfish_mace_inputs"
RESULT_ROOT = RAW / "rockfish_mace_results"
OUT_PROCESSED = ROOT / "processed"
OUT_TABLES = ROOT / "tables"
AUDIT = ROOT.parents[0] / "discovery_dft_audit_20261016"
RY_TO_EV = 13.605693122994
INTERFACE_MARGIN_FRAC = 0.040


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_roles(value) -> list[dict]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def route_family(path_id: str) -> str:
    if "gap_to_layer_penetration" in path_id:
        return "gap-to-QL penetration"
    if "same_gap_lateral_coordination_switch" in path_id:
        return "within-gap lateral / coordination switch"
    if "same_gap_lateral_same_environment" in path_id:
        return "within-gap lateral / same environment"
    return "unclassified route"


def cr_index(atoms) -> int:
    indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == "Cr"]
    if len(indices) != 1:
        raise ValueError(f"expected exactly one Cr atom, found {len(indices)}")
    return indices[0]


def displacement_a(fracs: list[np.ndarray], cell: np.ndarray, pbc: np.ndarray) -> list[float]:
    distances = []
    for left, right in combinations(fracs, 2):
        delta = np.array(left, dtype=float) - np.array(right, dtype=float)
        for axis, periodic in enumerate(pbc):
            if periodic:
                delta[axis] -= np.rint(delta[axis])
        distances.append(float(np.linalg.norm(delta @ cell)))
    return distances


def base_endpoint_geometry(tasks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_tasks = tasks.loc[tasks["variant"].eq("base")].copy()
    if base_tasks["raw_site_id"].nunique() != 47 or len(base_tasks) != 47:
        raise RuntimeError(f"expected 47 unique base tasks, received {len(base_tasks)}")

    endpoint_rows = []
    role_rows = []
    for task in base_tasks.to_dict("records"):
        endpoint = task["raw_site_id"]
        task_id = task["task_id"]
        input_path = INPUT_ROOT / task["input_extxyz"]
        atoms = read(input_path)
        symbols = np.array(atoms.get_chemical_symbols())
        index = cr_index(atoms)
        scaled = atoms.get_scaled_positions(wrap=True)
        frac = scaled[index]
        cell = np.asarray(atoms.cell)
        te_indices = np.flatnonzero(symbols == "Te")
        sb_indices = np.flatnonzero(symbols == "Sb")
        te_frac_z = scaled[te_indices, 2]
        lower_central = float(np.median(te_frac_z[(te_frac_z > 0.44) & (te_frac_z < 0.50)]))
        upper_central = float(np.median(te_frac_z[(te_frac_z > 0.50) & (te_frac_z < 0.56)]))
        if not np.isfinite(lower_central) or not np.isfinite(upper_central):
            raise RuntimeError(f"central Te surfaces unavailable for {endpoint}")
        gap_center = (lower_central + upper_central) / 2
        if lower_central <= frac[2] <= upper_central:
            vertical_region = "within_vdW_gap"
        elif (lower_central - INTERFACE_MARGIN_FRAC) <= frac[2] < lower_central:
            vertical_region = "gap_QL_interfacial"
        elif upper_central < frac[2] <= (upper_central + INTERFACE_MARGIN_FRAC):
            vertical_region = "gap_QL_interfacial"
        else:
            vertical_region = "within_QL"
        te_distances = atoms.get_distances(index, te_indices, mic=True)
        sb_distances = atoms.get_distances(index, sb_indices, mic=True)
        c_axis_a = float(np.linalg.norm(cell[2]))
        roles = parse_roles(task["roles"])
        families = sorted({route_family(item.get("path_id", "")) for item in roles})
        endpoint_rows.append(
            {
                "endpoint": endpoint,
                "base_task_id": task_id,
                "input_extxyz": task["input_extxyz"],
                "input_extxyz_sha256": task["input_extxyz_sha256"],
                "source_qe_input_sha256": task["source_qe_input_sha256"],
                "n_atoms": len(atoms),
                "cr_frac_x": float(frac[0]),
                "cr_frac_y": float(frac[1]),
                "cr_frac_z": float(frac[2]),
                "cell_c_A": c_axis_a,
                "central_lower_te_surface_frac_z": lower_central,
                "central_upper_te_surface_frac_z": upper_central,
                "vdw_gap_center_frac_z": gap_center,
                "cr_to_vdw_gap_center_A": abs(float(frac[2]) - gap_center) * c_axis_a,
                "vertical_region_geometric": vertical_region,
                "nearest_Te_A": float(np.min(te_distances)),
                "nearest_Sb_A": float(np.min(sb_distances)),
                "n_Te_within_3p2_A": int(np.sum(te_distances <= 3.2)),
                "n_Sb_within_3p2_A": int(np.sum(sb_distances <= 3.2)),
                "n_Te_within_3p6_A": int(np.sum(te_distances <= 3.6)),
                "n_Sb_within_3p6_A": int(np.sum(sb_distances <= 3.6)),
                "route_families_multilabel": " | ".join(families),
                "site_geometry_scope": "MACE base input; not DFT final geometry",
            }
        )
        for item in roles:
            path_id = item.get("path_id", "")
            role_rows.append(
                {
                    "endpoint": endpoint,
                    "base_task_id": task_id,
                    "path_id": path_id,
                    "endpoint_role": item.get("role", ""),
                    "route_family": route_family(path_id),
                }
            )
    return pd.DataFrame(endpoint_rows).sort_values("endpoint"), pd.DataFrame(role_rows).sort_values(
        ["route_family", "path_id", "endpoint"]
    )


def dft_evidence(endpoints: pd.DataFrame) -> pd.DataFrame:
    retry = pd.read_csv(AUDIT / "DISCOVERY_ENDPOINT_RETRY_FINAL_VALUES_20261023.tsv", sep="\t")
    retry = retry.rename(columns={"candidate_id": "endpoint"})
    retry["retry_subset"] = True
    retry["dft_native_energy_Ry"] = pd.to_numeric(retry["energy_ry"])
    retry["dft_final_force_Ry_Bohr"] = pd.to_numeric(retry["total_force_ry_per_au"])
    retry["dft_evidence_source"] = "20261023 retry final-values audit"
    retry["dft_geometry_evidence"] = np.where(
        retry["geometry_termination"].eq("BFGS_FINAL_COORDINATES"),
        "DIRECT_BFGS_FINAL_COORDINATES_REMOTE",
        "TIME_LIMIT_LAST_STEP_REMOTE",
    )
    retry["relaxation_class"] = np.where(
        retry["geometry_termination"].eq("BFGS_FINAL_COORDINATES"),
        "DFT_LOCAL_MINIMUM_CANDIDATE",
        "DFT_TIME_LIMIT_NONCONVERGED_EXCLUDE",
    )
    retry["dft_final_geometry_local"] = False
    retry["scheduler_log_evidence"] = "PRESENT_EMPTY"
    retry["scientific_status"] = np.where(
        retry["geometry_termination"].eq("BFGS_FINAL_COORDINATES"),
        "MERGE_ELIGIBLE_RETRY_PACKAGE_PENDING_FULL_PUBLICATION_GATES",
        "EXCLUDE_NONCONVERGED_RETRY",
    )
    retry["dft_relative_energy_eV_retry21"] = np.nan
    retry_mask = retry["geometry_termination"].eq("BFGS_FINAL_COORDINATES")
    retry.loc[retry_mask, "dft_relative_energy_eV_retry21"] = (
        retry.loc[retry_mask, "dft_native_energy_Ry"] - retry.loc[retry_mask, "dft_native_energy_Ry"].min()
    ) * RY_TO_EV
    retry["dft_energy_scope"] = np.where(
        retry_mask,
        "relative energy valid only within 21 converged 20261017 retry endpoints",
        "not rankable: time-limit nonconverged",
    )

    old = pd.read_csv(
        AUDIT / "mace_dft_endpoint_19_20261017" / "MACE_DFT_ENDPOINT_19_COMPARISON.tsv", sep="\t"
    )
    old = old.rename(
        columns={
            "endpoint": "endpoint",
            "dft_job_id": "job_id",
            "dft_energy_Ry": "dft_native_energy_Ry",
            "dft_force_Ry_Bohr": "dft_final_force_Ry_Bohr",
        }
    )
    old["retry_subset"] = False
    old["dft_evidence_source"] = "20261017 compact DFT/MACE comparison"
    old["dft_geometry_evidence"] = "DIRECT_FINAL_COORDINATES_REMOTE"
    old["relaxation_class"] = "PREEXISTING_TERMINAL_EVIDENCE_PENDING_FULL_GATE"
    old["dft_final_geometry_local"] = False
    old["scheduler_log_evidence"] = "NOT_LOCAL_RECHECKED_IN_20261023_RETRY_AUDIT"
    old["scientific_status"] = "OPERATIONAL_TERMINAL_EVIDENCE_PENDING_FULL_PUBLICATION_GATES"
    old["dft_relative_energy_eV_retry21"] = np.nan
    old["dft_energy_scope"] = "not pooled with retry21 before protocol/cell/pseudopotential reconciliation"

    historical = pd.read_csv(AUDIT / "endpoint_47_live_20261017" / "ENDPOINT_47_STATUS.tsv", sep="\t")
    by_endpoint = {}
    for row in retry.to_dict("records"):
        by_endpoint[row["endpoint"]] = row
    for row in old.to_dict("records"):
        by_endpoint[row["endpoint"]] = row

    reconstructed = {"unique_097", "unique_102", "unique_104", "unique_230"}
    rows = []
    for endpoint in endpoints["endpoint"]:
        if endpoint in by_endpoint:
            row = dict(by_endpoint[endpoint])
        elif endpoint in reconstructed:
            status_row = historical.loc[historical["endpoint"].eq(endpoint)].iloc[0].to_dict()
            row = {
                "endpoint": endpoint,
                "job_id": status_row["job_id"],
                "retry_subset": False,
                "dft_native_energy_Ry": np.nan,
                "dft_final_force_Ry_Bohr": np.nan,
                "dft_evidence_source": "20261019 endpoint identity audit",
                "dft_geometry_evidence": "RECONSTRUCTED_FINAL_COORDINATES_REMOTE_EVIDENCE_ONLY",
                "relaxation_class": "RECONSTRUCTED_FINAL_GEOMETRY_EVIDENCE_ONLY",
                "dft_final_geometry_local": False,
                "scheduler_log_evidence": "NOT_LOCAL_RECHECKED_IN_20261023_RETRY_AUDIT",
                "scientific_status": "RECONSTRUCTED_EVIDENCE_ONLY_PENDING_FULL_PUBLICATION_GATES",
                "dft_relative_energy_eV_retry21": np.nan,
                "dft_energy_scope": "no local numerical value in supplied compact audit",
            }
        else:
            raise RuntimeError(f"endpoint missing from DFT evidence map: {endpoint}")
        row["thermodynamic_label"] = "NOT_CERTIFIED: requires landscape/phonon or Hessian evidence"
        row["matched_dft_mace_error_eligibility"] = (
            "NO: DFT final geometry not locally available for same-geometry MACE evaluation"
        )
        rows.append(row)
    keep = [
        "endpoint",
        "job_id",
        "retry_subset",
        "dft_native_energy_Ry",
        "dft_final_force_Ry_Bohr",
        "dft_relative_energy_eV_retry21",
        "dft_energy_scope",
        "dft_evidence_source",
        "dft_geometry_evidence",
        "dft_final_geometry_local",
        "scheduler_log_evidence",
        "relaxation_class",
        "thermodynamic_label",
        "scientific_status",
        "matched_dft_mace_error_eligibility",
    ]
    return pd.DataFrame(rows)[keep].sort_values("endpoint")


def read_final_cr(summary_path: Path) -> dict:
    final_path = summary_path.parent / "mace_relaxed.extxyz"
    result = {
        "mace_final_extxyz_relpath": rel(final_path),
        "mace_final_extxyz_sha256": sha256(final_path) if final_path.exists() else "",
        "mace_final_cr_frac_x": np.nan,
        "mace_final_cr_frac_y": np.nan,
        "mace_final_cr_frac_z": np.nan,
        "mace_final_cell_json": "",
        "mace_final_pbc_json": "",
    }
    if not final_path.exists():
        return result
    atoms = read(final_path)
    frac = atoms.get_scaled_positions(wrap=True)[cr_index(atoms)]
    result.update(
        {
            "mace_final_cr_frac_x": float(frac[0]),
            "mace_final_cr_frac_y": float(frac[1]),
            "mace_final_cr_frac_z": float(frac[2]),
            "mace_final_cell_json": json.dumps(np.asarray(atoms.cell).round(10).tolist()),
            "mace_final_pbc_json": json.dumps(np.asarray(atoms.pbc, dtype=bool).tolist()),
        }
    )
    return result


def normalize_existing_model(label: str) -> str:
    mapping = {
        "FT-600K": "FT-600K",
        "FT-MultiT": "FT-MultiT",
        "Foundation": "OMAT-derived fine-tuned legacy Foundation",
        "Scratch": "Scratch",
    }
    return mapping.get(label, label)


def existing_mace_records() -> pd.DataFrame:
    rows = []
    collections = [
        ("legacy_ft600k", RESULT_ROOT / "legacy_ft600k", "FT-600K"),
        ("four_model", RESULT_ROOT / "four_model" / "runs", None),
    ]
    for collection, root, assumed_label in collections:
        for summary_path in sorted(root.glob("**/mace_relax_summary.json")):
            data = json.loads(summary_path.read_text())
            task = data["task"]
            original_label = assumed_label or data.get("model_label", "unknown")
            success = bool(data.get("scientific_success", data.get("optimizer_converged", False)))
            row = {
                "mace_collection": collection,
                "mace_summary_relpath": rel(summary_path),
                "mace_summary_sha256": sha256(summary_path),
                "task_id": task["task_id"],
                "endpoint": task["raw_site_id"],
                "variant": task["variant"],
                "model_label_raw": original_label,
                "model_label_interpreted": normalize_existing_model(original_label),
                "model_path": data.get("model_path", ""),
                "model_sha256": data.get("model_sha256", ""),
                "mace_scientific_success": success,
                "mace_optimizer_converged": bool(data.get("optimizer_converged", False)),
                "mace_steps_completed": data.get("optimizer_steps_completed", np.nan),
                "mace_final_energy_eV": data.get("final_energy_eV", np.nan),
                "mace_final_max_force_eV_A": data.get("final_max_force_eV_A", np.nan),
                "mace_input_sha256_verified": data.get("input_sha256_verified", task.get("input_extxyz_sha256", "")),
            }
            row.update(read_final_cr(summary_path))
            rows.append(row)
    records = pd.DataFrame(rows)
    expected = 4 * 47 * 3
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} existing MACE records; found {len(records)}")
    return records.sort_values(["model_label_interpreted", "endpoint", "variant"])


def model_relaxation_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (endpoint, label), group in records.groupby(["endpoint", "model_label_interpreted"], sort=True):
        success = group.loc[group["mace_scientific_success"]].copy()
        frac = [
            np.array([row.mace_final_cr_frac_x, row.mace_final_cr_frac_y, row.mace_final_cr_frac_z])
            for row in success.itertuples()
            if np.isfinite(row.mace_final_cr_frac_x)
        ]
        cell = (
            np.array(json.loads(success.iloc[0]["mace_final_cell_json"]))
            if len(success) and success.iloc[0]["mace_final_cell_json"]
            else np.eye(3)
        )
        pbc = (
            np.array(json.loads(success.iloc[0]["mace_final_pbc_json"]), dtype=bool)
            if len(success) and success.iloc[0]["mace_final_pbc_json"]
            else np.array([True, True, True])
        )
        distance = displacement_a(frac, cell, pbc) if len(frac) > 1 else []
        rows.append(
            {
                "endpoint": endpoint,
                "model_label_interpreted": label,
                "mace_expected_starts": len(group),
                "mace_successful_starts": len(success),
                "mace_failure_starts": int((~group["mace_scientific_success"]).sum()),
                "mace_median_final_fmax_eV_A": success["mace_final_max_force_eV_A"].median(),
                "mace_within_model_energy_span_eV": (
                    success["mace_final_energy_eV"].max() - success["mace_final_energy_eV"].min()
                    if len(success) else np.nan
                ),
                "mace_within_model_final_cr_span_A": max(distance) if distance else (0.0 if len(frac) == 1 else np.nan),
                "mace_sensitivity_interpretation": (
                    "discrete three-start basin sensitivity; not a barrier or DFT error"
                ),
            }
        )
    return pd.DataFrame(rows)


def cross_model_basin_summary(records: pd.DataFrame) -> pd.DataFrame:
    task_rows = []
    for task_id, group in records.groupby("task_id", sort=True):
        success = group.loc[group["mace_scientific_success"]].copy()
        fracs = [
            np.array([row.mace_final_cr_frac_x, row.mace_final_cr_frac_y, row.mace_final_cr_frac_z])
            for row in success.itertuples()
            if np.isfinite(row.mace_final_cr_frac_x)
        ]
        cell = (
            np.array(json.loads(success.iloc[0]["mace_final_cell_json"]))
            if len(success) and success.iloc[0]["mace_final_cell_json"]
            else np.eye(3)
        )
        pbc = (
            np.array(json.loads(success.iloc[0]["mace_final_pbc_json"]), dtype=bool)
            if len(success) and success.iloc[0]["mace_final_pbc_json"]
            else np.array([True, True, True])
        )
        distance = displacement_a(fracs, cell, pbc) if len(fracs) > 1 else []
        task_rows.append(
            {
                "task_id": task_id,
                "endpoint": group.iloc[0]["endpoint"],
                "variant": group.iloc[0]["variant"],
                "models_expected": len(group),
                "models_successful": len(success),
                "models_failed": int((~group["mace_scientific_success"]).sum()),
                "cross_model_cr_displacement_max_A": max(distance) if distance else (0.0 if len(fracs) == 1 else np.nan),
                "cross_model_cr_displacement_median_A": np.median(distance) if distance else (0.0 if len(fracs) == 1 else np.nan),
                "successful_models": " | ".join(sorted(success["model_label_interpreted"].tolist())),
            }
        )
    by_task = pd.DataFrame(task_rows)
    endpoint_rows = []
    for endpoint, group in by_task.groupby("endpoint", sort=True):
        endpoint_rows.append(
            {
                "endpoint": endpoint,
                "mace_four_model_expected_records": int(group["models_expected"].sum()),
                "mace_four_model_successful_records": int(group["models_successful"].sum()),
                "mace_four_model_failed_records": int(group["models_failed"].sum()),
                "mace_cross_model_cr_displacement_max_A": group["cross_model_cr_displacement_max_A"].max(),
                "mace_cross_model_cr_displacement_median_A": group["cross_model_cr_displacement_median_A"].median(),
                "mace_cross_model_scope": (
                    "final Cr basin disagreement across four legacy relaxation models; not DFT error"
                ),
            }
        )
    return by_task, pd.DataFrame(endpoint_rows)


def benjamini_hochberg(values: list[float]) -> list[float]:
    valid = [(index, value) for index, value in enumerate(values) if np.isfinite(value)]
    adjusted = [np.nan] * len(values)
    if not valid:
        return adjusted
    ordered = sorted(valid, key=lambda item: item[1])
    total = len(ordered)
    running = 1.0
    for rank, (index, value) in reversed(list(enumerate(ordered, start=1))):
        candidate = min(running, value * total / rank)
        adjusted[index] = candidate
        running = candidate
    return adjusted


def exploratory_associations(endpoint_table: pd.DataFrame, role_table: pd.DataFrame) -> pd.DataFrame:
    outcome = "mace_cross_model_cr_displacement_max_A"
    numeric_features = [
        "cr_to_vdw_gap_center_A",
        "nearest_Te_A",
        "nearest_Sb_A",
        "n_Te_within_3p2_A",
        "n_Sb_within_3p2_A",
    ]
    rows = []
    eligible = endpoint_table.loc[endpoint_table[outcome].notna()].copy()
    for feature in numeric_features:
        sub = eligible[[feature, outcome]].dropna()
        rho, p_value = spearmanr(sub[feature], sub[outcome])
        rows.append(
            {
                "outcome": outcome,
                "feature": feature,
                "test": "Spearman rank",
                "n": len(sub),
                "effect_size": rho,
                "p_value_raw": p_value,
                "interpretation": "exploratory model-disagreement association; not calibrated DFT error",
            }
        )
    membership = role_table[["endpoint", "route_family"]].drop_duplicates()
    for family in sorted(membership["route_family"].unique()):
        member = set(membership.loc[membership["route_family"].eq(family), "endpoint"])
        yes = eligible.loc[eligible["endpoint"].isin(member), outcome].dropna()
        no = eligible.loc[~eligible["endpoint"].isin(member), outcome].dropna()
        if len(yes) and len(no):
            result = mannwhitneyu(yes, no, alternative="two-sided")
            effect = (2 * result.statistic / (len(yes) * len(no))) - 1
            rows.append(
                {
                    "outcome": outcome,
                    "feature": family,
                    "test": "Mann-Whitney / rank-biserial",
                    "n": len(yes) + len(no),
                    "effect_size": effect,
                    "p_value_raw": result.pvalue,
                    "interpretation": "multi-label route-family membership; exploratory only",
                }
            )
    output = pd.DataFrame(rows)
    output["p_value_BH_FDR"] = benjamini_hochberg(output["p_value_raw"].tolist())
    return output.sort_values(["p_value_BH_FDR", "feature"])


def write_tsv(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(OUT_PROCESSED / filename, sep="\t", index=False, float_format="%.10g")


def main() -> None:
    OUT_PROCESSED.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    tasks = pd.read_csv(INPUT_ROOT / "mace_endpoint_tasks.csv")
    endpoint_geometry, role_table = base_endpoint_geometry(tasks)
    dft = dft_evidence(endpoint_geometry)
    records = existing_mace_records()
    model_summary = model_relaxation_summary(records)
    task_summary, endpoint_mace = cross_model_basin_summary(records)

    four_model_pivot = model_summary.pivot(
        index="endpoint", columns="model_label_interpreted", values="mace_within_model_energy_span_eV"
    ).add_prefix("mace_energy_span_eV__").reset_index()
    endpoint_table = (
        endpoint_geometry.merge(dft, on="endpoint", how="left")
        .merge(endpoint_mace, on="endpoint", how="left")
        .merge(four_model_pivot, on="endpoint", how="left")
        .sort_values("endpoint")
    )
    source_manifest = endpoint_geometry[
        ["endpoint", "base_task_id", "input_extxyz", "input_extxyz_sha256", "source_qe_input_sha256"]
    ].copy()
    source_manifest["static_evaluation_scope"] = (
        "base source geometry only; read-only MACE energy/force evaluation, no relaxation"
    )
    error_eligibility = endpoint_table[
        ["endpoint", "dft_final_geometry_local", "matched_dft_mace_error_eligibility", "dft_energy_scope"]
    ].copy()
    error_eligibility["required_to_enable_true_error"] = (
        "DFT final XYZ plus MACE energy/forces evaluated on that exact geometry, with matching cell/PBC/species"
    )
    associations = exploratory_associations(endpoint_table, role_table)

    write_tsv(endpoint_table, "DFT_ENDPOINT_47_CLASSIFICATION.tsv")
    write_tsv(role_table, "ENDPOINT_PATH_ROLES.tsv")
    write_tsv(source_manifest, "MACE_SOURCE_GEOMETRY_INPUT_MANIFEST.tsv")
    write_tsv(records, "MACE_FOUR_MODEL_RECORDS.tsv")
    write_tsv(model_summary, "MACE_FOUR_MODEL_ENDPOINT_MODEL_SUMMARY.tsv")
    write_tsv(task_summary, "MACE_FOUR_MODEL_TASK_BASIN_SUMMARY.tsv")
    write_tsv(endpoint_mace, "MACE_FOUR_MODEL_ENDPOINT_SUMMARY.tsv")
    write_tsv(error_eligibility, "DFT_MACE_ERROR_ELIGIBILITY.tsv")
    write_tsv(associations, "EXPLORATORY_MODEL_DISAGREEMENT_ASSOCIATIONS.tsv")

    quality = pd.DataFrame(
        [
            {
                "metric": "endpoint base geometries",
                "value": int(len(endpoint_geometry)),
                "status": "VERIFIED",
                "note": "47 source input structures, each with 81 atoms and one Cr",
            },
            {
                "metric": "retry BFGS final coordinates",
                "value": int(dft["relaxation_class"].eq("DFT_LOCAL_MINIMUM_CANDIDATE").sum()),
                "status": "LINEAGE_SUPPORTED",
                "note": "21 of 24 retry subset; scheduler out/err are present but empty",
            },
            {
                "metric": "retry nonconverged time-limit endpoints",
                "value": int(dft["relaxation_class"].eq("DFT_TIME_LIMIT_NONCONVERGED_EXCLUDE").sum()),
                "status": "EXCLUDE_FROM_PUBLICATION",
                "note": "unique_077, unique_078, unique_093",
            },
            {
                "metric": "existing MACE relaxation records",
                "value": int(len(records)),
                "status": "VERIFIED",
                "note": "4 legacy model labels x 47 endpoints x 3 deterministic starts",
            },
            {
                "metric": "existing MACE failed relaxation records",
                "value": int((~records["mace_scientific_success"]).sum()),
                "status": "FAILURE_MODE",
                "note": "three Scratch records at unique_151",
            },
            {
                "metric": "true DFT-MACE error eligible endpoints",
                "value": 0,
                "status": "EXCLUDE_FROM_PUBLICATION",
                "note": "no DFT final XYZ is locally available for matched-geometry MACE evaluation",
            },
        ]
    )
    quality.to_csv(OUT_TABLES / "DFT47_DATA_QUALITY_REPORT.tsv", sep="\t", index=False)
    claim_matrix = pd.DataFrame(
        [
            {
                "claim": "47 candidate source structures have a reproducible lattice/site distribution",
                "status": "VERIFIED",
                "evidence": "MACE source inputs, hashes, and geometry descriptor table",
                "allowed_use": "site-distribution figure",
            },
            {
                "claim": "21 retry endpoints can be ranked by DFT relative energy",
                "status": "LINEAGE_SUPPORTED",
                "evidence": "same retry campaign BFGS final energies",
                "allowed_use": "restricted retry21 relative-energy panel only",
            },
            {
                "claim": "any endpoint is thermodynamically stable, metastable, or unstable",
                "status": "NOT_ESTABLISHED",
                "evidence": "no Hessian/phonon or global-landscape evidence",
                "allowed_use": "do not label stable/metastable/unstable",
            },
            {
                "claim": "MACE has an absolute energy error against DFT",
                "status": "EXCLUDE_FROM_PUBLICATION",
                "evidence": "no matched DFT final geometry locally available",
                "allowed_use": "do not subtract current total energies",
            },
            {
                "claim": "existing four-model runs exhibit local-basin sensitivity",
                "status": "VERIFIED",
                "evidence": "564 final relaxation summaries and coordinates",
                "allowed_use": "model-internal spans and cross-model Cr displacement only",
            },
            {
                "claim": "12-member source-geometry ensemble has been evaluated",
                "status": "PENDING_COMPUTE",
                "evidence": "hash-verified registry and 47 base input manifest",
                "allowed_use": "pending separate additive Slurm run",
            },
        ]
    )
    claim_matrix.to_csv(OUT_TABLES / "CLAIM_EVIDENCE_MATRIX.tsv", sep="\t", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)),
        "input_tasks_sha256": sha256(INPUT_ROOT / "mace_endpoint_tasks.csv"),
        "input_campaign_manifest_sha256": sha256(INPUT_ROOT / "mace_endpoint_campaign_manifest.json"),
        "existing_four_model_manifest_sha256": sha256(RESULT_ROOT / "four_model" / "campaign_manifest.json"),
        "source_geometry_model_registry_sha256": sha256(ROOT / "MODEL_REGISTRY.tsv"),
        "n_endpoints": int(len(endpoint_geometry)),
        "n_existing_mace_records": int(len(records)),
        "n_existing_mace_successes": int(records["mace_scientific_success"].sum()),
        "n_existing_mace_failures": int((~records["mace_scientific_success"]).sum()),
        "boundary": "No cross-method total-energy subtraction or true error claim is produced.",
    }
    (OUT_TABLES / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
