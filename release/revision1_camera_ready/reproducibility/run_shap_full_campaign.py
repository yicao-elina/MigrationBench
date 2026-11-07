#!/usr/bin/env python3
"""Run the four-family, three-training-seed SHAP error analysis.

The analysis uses the common 267-frame test scope shared by the 600K and
Multi-T campaigns.  All four model families therefore explain the same
structures, while the three training seeds are retained as independent
replicates and aggregated only after TreeSHAP is computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler

from shap_feature_labels import all_feature_names


FAMILIES = {
    "FT-600K": "ft_600k",
    "FT-MultiT": "ft_multit",
    "Scratch-600K": "scratch_600k",
    "Scratch-MultiT": "scratch_multit",
}
SEEDS = (123, 234, 345)
N_COMMON_FRAMES = 267
EXPECTED_N_ATOMS = 86


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_features(xyz: Path) -> tuple[np.ndarray, list[str], list[dict[str, object]]]:
    frames = read(xyz, index=":")
    if len(frames) != N_COMMON_FRAMES:
        raise ValueError(f"expected {N_COMMON_FRAMES} common frames, got {len(frames)}")
    atom_counts = [len(frame) for frame in frames]
    if set(atom_counts) != {EXPECTED_N_ATOMS}:
        raise ValueError(f"common test scope must contain {EXPECTED_N_ATOMS}-atom frames, got {sorted(set(atom_counts))}")

    soap = SOAP(
        species=["Cr", "Sb", "Te"],
        periodic=True,
        r_cut=5.0,
        n_max=4,
        l_max=4,
        average="inner",
    )
    raw = np.asarray(soap.create(frames), dtype=float)
    if raw.shape != (N_COMMON_FRAMES, 390):
        raise ValueError(f"expected SOAP shape {(N_COMMON_FRAMES, 390)}, got {raw.shape}")
    scaled = StandardScaler().fit_transform(raw)

    structural: list[list[float]] = []
    frame_metadata: list[dict[str, object]] = []
    for frame_index, atoms in enumerate(frames):
        i, _, distances = neighbor_list("ijd", atoms, cutoff=5.0)
        coordination = np.bincount(i, minlength=len(atoms))
        symbols = atoms.get_chemical_symbols()
        structural.append([
            len(atoms), atoms.get_volume(), atoms.get_volume() / len(atoms),
            *atoms.cell.lengths(), *atoms.cell.angles(),
            coordination.mean(), coordination.std(), distances.mean(), distances.std(), distances.min(),
            symbols.count("Cr") / len(atoms), symbols.count("Sb") / len(atoms), symbols.count("Te") / len(atoms),
        ])
        frame_metadata.append({
            "frame": frame_index,
            "trajectory": str(atoms.info.get("source_trajectory", "NA")),
            "source_frame_after_equilibration": atoms.info.get("source_frame_after_equilibration", "NA"),
            "n_atoms": len(atoms),
        })

    names = all_feature_names()
    features = np.hstack([scaled, np.asarray(structural, dtype=float)])
    if features.shape != (N_COMMON_FRAMES, len(names)):
        raise ValueError(f"feature matrix/name mismatch: {features.shape} vs {len(names)} names")
    return features, names, frame_metadata


def load_targets(source_tables: Path, frame_metadata: list[dict[str, object]]) -> tuple[pd.DataFrame, dict[str, object]]:
    aligned: list[pd.DataFrame] = []
    diagnostics: dict[str, object] = {"families": {}, "common_frame_count": N_COMMON_FRAMES}
    for family, stem in FAMILIES.items():
        family_diag: dict[str, object] = {}
        for seed in SEEDS:
            path = source_tables / f"{stem}_seed{seed}_test__per_frame.csv"
            df = pd.read_csv(path)
            required = {
                "frame", "trajectory", "n_atoms", "energy_ref_eV_per_atom",
                "energy_pred_eV_per_atom", "energy_abs_error_meV_per_atom", "force_rmse_meV_A",
            }
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
            common = df[df["trajectory"] == "2L_o3_t3_600K_aimd_1"].sort_values("frame").reset_index(drop=True)
            if len(common) != N_COMMON_FRAMES or common["frame"].tolist() != list(range(N_COMMON_FRAMES)):
                raise ValueError(f"{path.name} does not contain the expected ordered common {N_COMMON_FRAMES}-frame trajectory")
            if not np.isfinite(common[["energy_ref_eV_per_atom", "energy_pred_eV_per_atom", "energy_abs_error_meV_per_atom", "force_rmse_meV_A"]].to_numpy(float)).all():
                raise ValueError(f"{path.name} contains non-finite evaluation values")
            if set(common["n_atoms"].unique()) != {EXPECTED_N_ATOMS}:
                raise ValueError(f"{path.name} common scope contains unexpected atom counts: {sorted(common['n_atoms'].unique())}")
            row = common[["frame", "trajectory", "n_atoms", "energy_ref_eV_per_atom", "energy_pred_eV_per_atom", "energy_abs_error_meV_per_atom", "force_rmse_meV_A"]].copy()
            row.insert(0, "seed", seed)
            row.insert(0, "model_family", family)
            aligned.append(row)
            family_diag[str(seed)] = {
                "path": str(path),
                "rows_total": int(len(df)),
                "rows_in_common_scope": int(len(common)),
                "rows_excluded_from_common_scope": int(len(df) - len(common)),
                "sha256": sha256_file(path),
            }
        diagnostics["families"][family] = family_diag

    joined = pd.concat(aligned, ignore_index=True)
    ref = joined[joined.model_family == "FT-600K"].sort_values(["seed", "frame"])["energy_ref_eV_per_atom"].to_numpy()
    for family in FAMILIES:
        values = joined[joined.model_family == family].sort_values(["seed", "frame"])["energy_ref_eV_per_atom"].to_numpy()
        if not np.allclose(values, ref, rtol=0.0, atol=1e-10):
            raise ValueError(f"reference energies differ across families for {family}")
    return joined, diagnostics


def run_analysis(features: np.ndarray, names: list[str], targets: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    import shap

    output.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    value_blocks: list[np.ndarray] = []
    for family, _stem in FAMILIES.items():
        for seed in SEEDS:
            group = targets[(targets.model_family == family) & (targets.seed == seed)].sort_values("frame")
            y = group["energy_abs_error_meV_per_atom"].to_numpy(float)
            estimator = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                random_state=42,
            )
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_estimator = GradientBoostingRegressor(
                n_estimators=50,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=42,
            )
            cv_result = cross_validate(
                cv_estimator,
                features,
                y,
                cv=cv,
                scoring={"r2": "r2", "mae": "neg_mean_absolute_error"},
                n_jobs=1,
            )
            estimator.fit(features, y)
            values = np.asarray(shap.TreeExplainer(estimator)(features).values, dtype=float)
            if values.shape != features.shape or not np.isfinite(values).all():
                raise ValueError(f"non-finite or malformed TreeSHAP array for {family} seed {seed}: {values.shape}")
            mean_abs = np.abs(values).mean(axis=0)
            ranks = pd.Series(mean_abs, index=names).rank(method="min", ascending=False).astype(int)
            for index, feature in enumerate(names):
                importance_rows.append({
                    "model_family": family,
                    "seed": seed,
                    "feature": feature,
                    "feature_index": index,
                    "mean_abs_shap_meV_per_atom": float(mean_abs[index]),
                    "rank": int(ranks.loc[feature]),
                })
            metrics.append({
                "model_family": family,
                "seed": seed,
                "n_samples": int(len(y)),
                "target_unit": "meV_per_atom",
                "target_mean": float(y.mean()),
                "target_median": float(np.median(y)),
                "target_max": float(y.max()),
                "r2_in_sample": float(estimator.score(features, y)),
                "mae_in_sample_meV_per_atom": float(np.abs(estimator.predict(features) - y).mean()),
                "r2_5fold_cv": float(cv_result["test_r2"].mean()),
                "mae_5fold_cv_meV_per_atom": float(-cv_result["test_mae"].mean()),
            })
            value_blocks.append(values)

    return pd.DataFrame(metrics), pd.DataFrame(importance_rows), np.stack(value_blocks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--source-tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    features, names, frame_metadata = extract_features(args.xyz)
    targets, target_diag = load_targets(args.source_tables, frame_metadata)
    metrics, importance, shap_values = run_analysis(features, names, targets, args.output)

    frame_table = targets.copy()
    frame_table["source_frame_after_equilibration"] = np.tile(
        [x["source_frame_after_equilibration"] for x in frame_metadata], len(FAMILIES) * len(SEEDS)
    )
    frame_table.to_csv(args.output / "shap_frame_aligned_full_campaign.csv", index=False)
    metrics.to_csv(args.output / "shap_seed_metrics.csv", index=False)
    importance.to_csv(args.output / "shap_seed_feature_importance.csv", index=False)

    aggregate = (
        importance.groupby(["model_family", "feature", "feature_index"], as_index=False)
        .agg(
            mean_abs_shap_meV_per_atom=("mean_abs_shap_meV_per_atom", "mean"),
            sd_abs_shap_meV_per_atom=("mean_abs_shap_meV_per_atom", "std"),
            mean_seed_rank=("rank", "mean"),
            sd_seed_rank=("rank", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    aggregate["sd_abs_shap_meV_per_atom"] = aggregate["sd_abs_shap_meV_per_atom"].fillna(0.0)
    aggregate["sd_seed_rank"] = aggregate["sd_seed_rank"].fillna(0.0)
    global_order = (
        aggregate.groupby("feature", as_index=False)["mean_abs_shap_meV_per_atom"].mean()
        .sort_values("mean_abs_shap_meV_per_atom", ascending=False)
        .reset_index(drop=True)
    )
    global_order["global_rank"] = np.arange(1, len(global_order) + 1)
    aggregate = aggregate.merge(global_order[["feature", "global_rank"]], on="feature", how="left")
    aggregate.sort_values(["global_rank", "model_family"], inplace=True)
    aggregate.to_csv(args.output / "shap_family_aggregate.csv", index=False)

    np.savez_compressed(
        args.output / "shap_values_full_campaign.npz",
        values=shap_values.astype(np.float32),
        feature_names=np.asarray(names),
        model_family=np.asarray([family for family in FAMILIES for _seed in SEEDS]),
        seed=np.asarray([seed for _family in FAMILIES for seed in SEEDS], dtype=np.int32),
        frame=np.arange(N_COMMON_FRAMES, dtype=np.int32),
    )

    manifest = {
        "analysis_id": "shap_full_campaign_20261101",
        "status": "COMPLETE_WITH_COMMON_TEST_SCOPE",
        "scope": {
            "families": list(FAMILIES),
            "seeds": list(SEEDS),
            "n_model_seed_runs": len(FAMILIES) * len(SEEDS),
            "split": "test",
            "n_common_frames": N_COMMON_FRAMES,
            "n_atoms": EXPECTED_N_ATOMS,
            "target": "absolute total-energy error per atom",
            "target_unit": "meV_per_atom",
            "excluded_scope": "500 additional 2L_tetra_Cr4 Multi-T test frames because no 600K cross-family predictions exist for them",
        },
        "method": {
            "descriptor": "DScribe SOAP periodic inner average, species Cr/Sb/Te, r_cut=5.0, n_max=4, l_max=4",
            "feature_count": len(names),
            "structural_feature_count": 17,
            "surrogate": "GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, subsample=0.8, random_state=42) for final TreeSHAP",
            "cv_surrogate": "GradientBoostingRegressor(n_estimators=50, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42) for 5-fold quality smoke",
            "shap": "shap.TreeExplainer",
            "cv": "5-fold shuffled KFold(random_state=42)",
            "seed_uncertainty": "mean +/- sample SD across training seeds 123/234/345",
        },
        "inputs": {
            "xyz": {"path": str(args.xyz), "sha256": sha256_file(args.xyz), "frames": N_COMMON_FRAMES},
            "source_tables": target_diag,
            "python": platform.python_version(),
        },
        "outputs": [
            "shap_frame_aligned_full_campaign.csv",
            "shap_seed_metrics.csv",
            "shap_seed_feature_importance.csv",
            "shap_family_aggregate.csv",
            "shap_values_full_campaign.npz",
        ],
        "quality_checks": {
            "all_12_seed_runs": bool(len(metrics) == 12),
            "all_samples_finite": bool(np.isfinite(shap_values).all()),
            "all_seed_counts_three": bool((aggregate.n_seeds == 3).all()),
            "reference_energy_consistent_across_families": True,
        },
    }
    (args.output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
