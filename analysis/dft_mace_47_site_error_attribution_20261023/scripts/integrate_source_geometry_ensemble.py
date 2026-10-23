#!/usr/bin/env python3
"""Integrate the completed 12-member static source-geometry MACE ensemble.

All uncertainty-like quantities produced here are model disagreement diagnostics.
They are deliberately not calibrated DFT errors because DFT final geometries are
not locally available for matched-geometry evaluations.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw" / "source_geometry_ensemble"
PROCESSED = ROOT / "processed"
TABLES = ROOT / "tables"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fdr_bh(values: list[float]) -> list[float]:
    valid = [(index, value) for index, value in enumerate(values) if np.isfinite(value)]
    adjusted = [np.nan] * len(values)
    previous = 1.0
    for rank, (index, value) in reversed(list(enumerate(sorted(valid, key=lambda item: item[1]), start=1))):
        candidate = min(previous, value * len(valid) / rank)
        adjusted[index] = candidate
        previous = candidate
    return adjusted


def load_records() -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = pd.read_csv(ROOT / "MODEL_REGISTRY.tsv", sep="\t")
    frames = []
    for model in registry.to_dict("records"):
        destination = RAW / "results" / model["analysis_model_id"]
        prediction = destination / "source_geometry_predictions.tsv"
        provenance = destination / "provenance.json"
        if not prediction.is_file() or not provenance.is_file():
            raise FileNotFoundError(f"missing static output for {model['analysis_model_id']}")
        data = pd.read_csv(prediction, sep="\t")
        metadata = json.loads(provenance.read_text())
        if len(data) != 47:
            raise RuntimeError(f"{model['analysis_model_id']} has {len(data)} rows, expected 47")
        if not data["status"].eq("SUCCESS").all():
            raise RuntimeError(f"{model['analysis_model_id']} reports non-success rows")
        if not data["model_sha256_observed"].eq(model["model_sha256"]).all():
            raise RuntimeError(f"{model['analysis_model_id']} did not preserve its registered hash")
        data["prediction_tsv_sha256"] = sha256(prediction)
        data["provenance_json_sha256"] = sha256(provenance)
        data["provenance_created_utc"] = metadata["created_utc"]
        frames.append(data)
    records = pd.concat(frames, ignore_index=True)
    if len(records) != 12 * 47:
        raise RuntimeError(f"got {len(records)} records, expected 564")
    return registry, records.sort_values(["analysis_model_id", "endpoint"])


def pairwise_cosines(vectors: np.ndarray) -> list[float]:
    norms = np.linalg.norm(vectors, axis=1)
    valid = vectors[norms > 1e-12]
    if len(valid) < 2:
        return []
    normalized = valid / np.linalg.norm(valid, axis=1, keepdims=True)
    return [float(np.dot(left, right)) for left, right in combinations(normalized, 2)]


def endpoint_summary(records: pd.DataFrame) -> pd.DataFrame:
    records = records.copy()
    records["relative_energy_eV_within_model"] = records.groupby("analysis_model_id")["energy_eV"].transform(
        lambda series: series - series.min()
    )
    records["relative_energy_rank_within_model"] = records.groupby("analysis_model_id")["energy_eV"].rank(
        method="average", ascending=True
    )
    summary_rows = []
    for endpoint, group in records.groupby("endpoint", sort=True):
        vectors = group[["cr_force_x_eV_A", "cr_force_y_eV_A", "cr_force_z_eV_A"]].to_numpy(float)
        force_norm = group["cr_force_norm_eV_A"].to_numpy(float)
        cosines = pairwise_cosines(vectors)
        component_std = np.std(vectors, axis=0, ddof=1)
        summary_rows.append(
            {
                "endpoint": endpoint,
                "ensemble_members": len(group),
                "ensemble_cr_force_norm_mean_eV_A": float(np.mean(force_norm)),
                "ensemble_cr_force_norm_sd_eV_A": float(np.std(force_norm, ddof=1)),
                "ensemble_cr_force_vector_std_rms_eV_A": float(np.sqrt(np.mean(component_std**2))),
                "ensemble_force_direction_median_cosine": float(np.median(cosines)),
                "ensemble_force_direction_disagreement": float(1.0 - np.median(cosines)),
                "ensemble_relative_energy_range_eV": float(
                    group["relative_energy_eV_within_model"].max()
                    - group["relative_energy_eV_within_model"].min()
                ),
                "ensemble_relative_energy_rank_sd": float(
                    group["relative_energy_rank_within_model"].std(ddof=1)
                ),
                "ensemble_diagnostic_scope": (
                    "12-member static source-geometry model disagreement; not calibrated DFT error"
                ),
            }
        )
    return records, pd.DataFrame(summary_rows)


def family_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, model_id), group in records.groupby(["analysis_family", "analysis_model_id"], sort=True):
        rows.append(
            {
                "analysis_family": family,
                "analysis_model_id": model_id,
                "n_static_predictions": len(group),
                "mean_cr_force_norm_eV_A": group["cr_force_norm_eV_A"].mean(),
                "median_cr_force_norm_eV_A": group["cr_force_norm_eV_A"].median(),
                "relative_energy_range_eV": group["relative_energy_eV_within_model"].max(),
                "status": "SUCCESS",
            }
        )
    return pd.DataFrame(rows)


def associations(endpoint: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    outcomes = [
        "ensemble_cr_force_vector_std_rms_eV_A",
        "ensemble_force_direction_disagreement",
        "ensemble_relative_energy_rank_sd",
    ]
    features = [
        "cr_to_vdw_gap_center_A",
        "nearest_Te_A",
        "nearest_Sb_A",
        "n_Te_within_3p2_A",
        "n_Sb_within_3p2_A",
    ]
    rows = []
    for outcome in outcomes:
        for feature in features:
            subset = endpoint[[outcome, feature]].dropna()
            rho, pvalue = spearmanr(subset[outcome], subset[feature])
            rows.append(
                {
                    "outcome": outcome,
                    "feature": feature,
                    "test": "Spearman rank",
                    "n": len(subset),
                    "effect_size": rho,
                    "p_value_raw": pvalue,
                    "interpretation": "exploratory ensemble-disagreement association, not DFT-error calibration",
                }
            )
    membership = roles[["endpoint", "route_family"]].drop_duplicates()
    for outcome in outcomes:
        for family in sorted(membership["route_family"].unique()):
            member = set(membership.loc[membership["route_family"].eq(family), "endpoint"])
            yes = endpoint.loc[endpoint["endpoint"].isin(member), outcome].dropna()
            no = endpoint.loc[~endpoint["endpoint"].isin(member), outcome].dropna()
            test = mannwhitneyu(yes, no, alternative="two-sided")
            effect = (2 * test.statistic / (len(yes) * len(no))) - 1
            rows.append(
                {
                    "outcome": outcome,
                    "feature": family,
                    "test": "Mann-Whitney / rank-biserial",
                    "n": len(yes) + len(no),
                    "effect_size": effect,
                    "p_value_raw": test.pvalue,
                    "interpretation": "multi-label route membership; exploratory ensemble diagnostic only",
                }
            )
    output = pd.DataFrame(rows)
    output["p_value_BH_FDR"] = fdr_bh(output["p_value_raw"].tolist())
    return output.sort_values(["outcome", "p_value_BH_FDR", "feature"])


def descriptor_projection(endpoint: pd.DataFrame) -> pd.DataFrame:
    features = [
        "cr_to_vdw_gap_center_A",
        "nearest_Te_A",
        "nearest_Sb_A",
        "n_Te_within_3p2_A",
        "n_Sb_within_3p2_A",
        "cr_frac_x",
        "cr_frac_y",
    ]
    matrix = endpoint[features].to_numpy(float)
    projection = PCA(n_components=2, random_state=20261023).fit_transform(StandardScaler().fit_transform(matrix))
    output = endpoint[["endpoint"]].copy()
    output["descriptor_PC1"] = projection[:, 0]
    output["descriptor_PC2"] = projection[:, 1]
    output["embedding_scope"] = "candidate structural descriptors; not training-data coverage or OOD proof"
    return output


def main() -> None:
    registry, records = load_records()
    records, ensemble = endpoint_summary(records)
    endpoint = pd.read_csv(PROCESSED / "DFT_ENDPOINT_47_CLASSIFICATION.tsv", sep="\t")
    roles = pd.read_csv(PROCESSED / "ENDPOINT_PATH_ROLES.tsv", sep="\t")
    projection = descriptor_projection(endpoint)
    integrated = endpoint.merge(ensemble, on="endpoint", how="left").merge(projection, on="endpoint", how="left")
    stats = associations(integrated, roles)
    models = family_summary(records)

    records.to_csv(PROCESSED / "MACE12_SOURCE_GEOMETRY_RECORDS.tsv", sep="\t", index=False, float_format="%.12g")
    ensemble.to_csv(PROCESSED / "MACE12_SOURCE_GEOMETRY_ENDPOINT_SUMMARY.tsv", sep="\t", index=False, float_format="%.12g")
    models.to_csv(PROCESSED / "MACE12_SOURCE_GEOMETRY_MODEL_SUMMARY.tsv", sep="\t", index=False, float_format="%.12g")
    projection.to_csv(PROCESSED / "MACE12_SOURCE_GEOMETRY_DESCRIPTOR_PROJECTION.tsv", sep="\t", index=False, float_format="%.12g")
    integrated.to_csv(PROCESSED / "DFT_MACE47_INTEGRATED_ENDPOINT_TABLE.tsv", sep="\t", index=False, float_format="%.12g")
    stats.to_csv(PROCESSED / "MACE12_SOURCE_GEOMETRY_ASSOCIATIONS.tsv", sep="\t", index=False, float_format="%.12g")

    quality = pd.DataFrame(
        [
            {
                "metric": "hash-verified model members",
                "value": len(registry),
                "status": "VERIFIED",
                "note": "3 Foundation-family members plus three seeds in each of FT-600K, FT-MultiT, and Scratch",
            },
            {
                "metric": "static source-geometry predictions",
                "value": len(records),
                "status": "VERIFIED",
                "note": "12 models x 47 immutable base geometries; all successful",
            },
            {
                "metric": "DFT-error-calibrated endpoints",
                "value": 0,
                "status": "EXCLUDE_FROM_PUBLICATION",
                "note": "source geometries are not locally available DFT final geometries",
            },
        ]
    )
    quality.to_csv(TABLES / "MACE12_SOURCE_ENSEMBLE_QUALITY_REPORT.tsv", sep="\t", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_registry_sha256": sha256(ROOT / "MODEL_REGISTRY.tsv"),
        "n_models": int(len(registry)),
        "n_endpoints": 47,
        "n_predictions": int(len(records)),
        "boundary": "Model disagreement is not a calibrated DFT error and descriptor PCA is not training-data coverage.",
    }
    (TABLES / "MACE12_SOURCE_ENSEMBLE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
