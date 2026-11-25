#!/usr/bin/env python3
"""Reproduce the MigrationBench Sec. 3.5 three-model SHAP surrogate.

This is the X-FORCE Fig. 6 workflow: 150 aligned frames from the compact XYZ,
the three-model 450-row prediction CSV, 390 SOAP plus 17 structural features,
GradientBoostingRegressor, 5-fold CV, and TreeSHAP. It intentionally does not
consume the four-model CSV, 2-concentration data, or Dual-X ID/OOD/NEB data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from shap_feature_labels import all_feature_names

MODELS = ("FT-600K", "FT-MultiT", "Scratch")


def extract_features(xyz: Path) -> tuple[np.ndarray, list[str]]:
    frames = read(xyz, index=":")
    if len(frames) != 150 or any(len(frame) != 82 for frame in frames):
        raise ValueError("Fig. 6 input must contain exactly 150 frames of 82 atoms")
    soap = SOAP(species=["Cr", "Sb", "Te"], periodic=True, r_cut=5.0,
                n_max=4, l_max=4, average="inner")
    raw = np.asarray(soap.create(frames), dtype=float)
    if raw.shape != (150, 390):
        raise ValueError(f"expected SOAP shape (150, 390), got {raw.shape}")
    scaled = StandardScaler().fit_transform(raw)
    structural = []
    for atoms in frames:
        i, _, distances = neighbor_list("ijd", atoms, cutoff=5.0)
        coordination = np.bincount(i, minlength=len(atoms))
        symbols = atoms.get_chemical_symbols()
        structural.append([
            len(atoms), atoms.get_volume(), atoms.get_volume() / len(atoms),
            *atoms.cell.lengths(), *atoms.cell.angles(),
            coordination.mean(), coordination.std(), distances.mean(), distances.std(), distances.min(),
            symbols.count("Cr") / len(atoms), symbols.count("Sb") / len(atoms),
            symbols.count("Te") / len(atoms),
        ])
    features = np.hstack([scaled, np.asarray(structural, dtype=float)])
    return features, all_feature_names()


def load_aligned(xyz: Path, predictions: Path) -> tuple[pd.DataFrame, list[str]]:
    features, names = extract_features(xyz)
    pred = pd.read_csv(predictions)
    required = {"model_name", "image_index", "total_energy_error_eV"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"prediction CSV missing columns: {sorted(missing)}")
    if set(pred.model_name.unique()) != set(MODELS):
        raise ValueError(f"expected exactly models {MODELS}, got {sorted(pred.model_name.unique())}")
    if len(pred) != 450 or pred.image_index.min() != 0 or pred.image_index.max() != 149:
        raise ValueError("expected 450 rows with image_index 0..149 for each model")
    pred = pred.sort_values(["model_name", "image_index"]).reset_index(drop=True)
    feature_rows = np.vstack([features[int(i)] for i in pred.image_index])
    frame = pd.DataFrame(feature_rows, columns=names)
    frame.insert(0, "image_index", pred.image_index.to_numpy())
    frame.insert(0, "model_name", pred.model_name.to_numpy())
    frame["target"] = pred.total_energy_error_eV.abs().to_numpy()
    return frame, names


def analyse(frame: pd.DataFrame, feature_names: list[str], output: Path, seed: int = 42) -> None:
    output.mkdir(parents=True, exist_ok=True)
    metrics, top = [], []
    for model in MODELS:
        group = frame[frame.model_name == model].sort_values("image_index")
        X, y = group[feature_names], group.target.to_numpy(float)
        estimator = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                              max_depth=8, subsample=0.8, random_state=seed)
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        r2 = cross_val_score(estimator, X, y, cv=cv, scoring="r2")
        mae = -cross_val_score(estimator, X, y, cv=cv, scoring="neg_mean_absolute_error")
        estimator.fit(X, y)
        import shap
        values = np.asarray(shap.TreeExplainer(estimator)(X).values)
        importance = np.abs(values).mean(axis=0)
        for rank, idx in enumerate(np.argsort(importance)[::-1][:50], 1):
            top.append({"model": model, "rank": rank, "feature": feature_names[idx],
                        "mean_abs_shap_eV": importance[idx], "feature_index": idx})
        metrics.append({"model": model, "n_samples": len(group),
                        "r2_in_sample": estimator.score(X, y),
                        "mae_in_sample": np.abs(estimator.predict(X) - y).mean(),
                        "r2_5fold_cv": r2.mean(), "mae_5fold_cv": mae.mean()})
    pd.DataFrame(metrics).to_csv(output / "shap_surrogate_r2_revised.csv", index=False)
    pd.DataFrame(top).to_csv(output / "shap_top_features_revised.csv", index=False)
    frame.to_csv(output / "shap_frame_aligned.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame, names = load_aligned(args.xyz, args.predictions)
    analyse(frame, names, args.output)


if __name__ == "__main__":
    main()
