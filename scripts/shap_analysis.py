#!/usr/bin/env python3
"""Reproducible SHAP surrogate analysis for MigrationBench.

The published analysis explains a surrogate for absolute total-energy error;
it does not attribute MACE internals.  Inputs are deliberately explicit:
either a single CSV containing ``model``, ``target`` and feature columns, or
one feature CSV plus a prediction CSV with ``model`` and ``target`` columns.
The ``--demo`` mode is a deterministic smoke test for a clean installation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _demo_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(150, 12))
    y = 0.8 * x[:, 0] - 0.4 * x[:, 3] ** 2 + 0.2 * x[:, 7] + rng.normal(0, 0.03, 150)
    return pd.DataFrame(x, columns=[f"feature_{i:03d}" for i in range(x.shape[1])]).assign(
        model="demo", target=np.abs(y)
    )


def _load(args: argparse.Namespace) -> pd.DataFrame:
    if args.demo:
        return _demo_frame()
    if args.input:
        return pd.read_csv(args.input)
    if not args.features or not args.predictions:
        raise SystemExit("provide --input, or both --features and --predictions")
    features = pd.read_csv(args.features)
    predictions = pd.read_csv(args.predictions)
    keys = [args.key] if args.key in features.columns and args.key in predictions.columns else []
    return predictions.merge(features, on=keys, how="inner", validate="one_to_one")


def analyse(frame: pd.DataFrame, output: Path, target: str, model_col: str, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    ignored = {target, model_col}
    feature_cols = [c for c in frame.columns if c not in ignored and pd.api.types.is_numeric_dtype(frame[c])]
    if len(feature_cols) < 2:
        raise ValueError("at least two numeric feature columns are required")
    results, top_rows = [], []
    for model_name, group in frame.groupby(model_col, sort=True):
        group = group.dropna(subset=feature_cols + [target])
        X, y = group[feature_cols], group[target].to_numpy(float)
        if len(group) < 10:
            raise ValueError(f"model {model_name!r} has only {len(group)} usable rows")
        estimator = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=8, subsample=0.8, random_state=seed
        )
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        scores = cross_val_score(estimator, X, y, cv=cv, scoring="r2")
        maes = -cross_val_score(estimator, X, y, cv=cv, scoring="neg_mean_absolute_error")
        estimator.fit(X, y)
        try:
            import shap
        except ImportError as exc:
            raise SystemExit("TreeSHAP requires requirements-shap.txt (or install shap==0.46.0)") from exc
        importance = np.abs(np.asarray(shap.TreeExplainer(estimator)(X).values)).mean(axis=0)
        for rank, idx in enumerate(np.argsort(np.abs(importance))[::-1][:50], 1):
            top_rows.append({"model": model_name, "rank": rank, "feature": feature_cols[idx],
                             "mean_abs_shap_eV": float(abs(importance[idx])), "feature_index": int(idx)})
        results.append({"model": model_name, "n_samples": len(group),
                       "r2_in_sample": float(estimator.score(X, y)),
                       "mae_in_sample": float(np.mean(np.abs(estimator.predict(X) - y))),
                       "r2_5fold_cv": float(scores.mean()), "mae_5fold_cv": float(maes.mean())})
    pd.DataFrame(results).to_csv(output / "shap_surrogate_r2.csv", index=False)
    pd.DataFrame(top_rows).to_csv(output / "shap_top_features.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, help="CSV with model, target, and numeric feature columns")
    p.add_argument("--features", type=Path)
    p.add_argument("--predictions", type=Path)
    p.add_argument("--key", default="frame", help="join key for split inputs")
    p.add_argument("--target", default="target")
    p.add_argument("--model-column", default="model")
    p.add_argument("--output", type=Path, default=Path("results/shap"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--demo", action="store_true", help="run deterministic installation smoke test")
    args = p.parse_args()
    analyse(_load(args), args.output, args.target, args.model_column, args.seed)


if __name__ == "__main__":
    main()
