from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def q(s: pd.Series, value: float) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return float(x.quantile(value)) if len(x) else float("nan")


def js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    a = a / a.sum() if a.sum() else a
    b = b / b.sum() if b.sum() else b
    m = 0.5 * (a + b)
    terms = []
    for x, y in ((a, m), (b, m)):
        mask = x > 0
        terms.append(np.sum(x[mask] * np.log2(x[mask] / y[mask])))
    return float(0.5 * sum(terms))


def add_common_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["candidate_id"] = out["candidate_id"].astype(str)
    out["model_id"] = out["model_id"].astype(str)
    if "model_training_seed" not in out:
        out["model_training_seed"] = out.get("training_seed", np.nan)
    if "evaluation_status" not in out:
        out["evaluation_status"] = "success"
    return out


def model_stage_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in results.groupby(["candidate_group", "window", "model_id"], dropna=False):
        cg, window, model_id = keys
        ok = group[group.evaluation_status == "success"]
        row = {
            "candidate_group": cg,
            "window": window,
            "model_id": model_id,
            "model_label": group.model_label.iloc[0] if "model_label" in group else model_id,
            "model_family": group.model_family.iloc[0] if "model_family" in group else "",
            "model_training_seed": group.model_training_seed.iloc[0],
            "n_candidates": int(len(group)),
            "n_success": int(len(ok)),
            "success_fraction": float(len(ok) / len(group)) if len(group) else np.nan,
        }
        for col in ["max_force_eV_A", "rms_force_eV_A", "p95_force_eV_A", "p99_force_eV_A", "energy_per_atom_eV"]:
            for suffix, quantile in [("median", 0.50), ("q25", 0.25), ("q75", 0.75), ("q95", 0.95)]:
                row[f"{col}_{suffix}"] = q(ok[col], quantile) if col in ok else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["candidate_group", "window", "model_id"])


def candidate_coverage(manifest: pd.DataFrame) -> pd.DataFrame:
    cols = ["candidate_id", "source", "run_key", "training_seed", "window", "candidate_group", "time_ps", "step"]
    out = manifest[cols].copy()
    out["candidate_id"] = out.candidate_id.astype(str)
    return out.sort_values("candidate_id")


def distribution_audit(manifest: pd.DataFrame, frame_features: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    m = manifest.copy()
    m["candidate_id"] = m.candidate_id.astype(str)
    f = frame_features.copy()
    if "snapshot_path" in f and "source_snapshot" not in f:
        f = f.rename(columns={"snapshot_path": "source_snapshot"})
    if "source_snapshot" in f and "source_snapshot" in m:
        join_cols = ["source_snapshot"]
        keep = [c for c in ["source_snapshot", "pca1", "pca2", "distance_to_train", "distance_to_neb", "outside_train_q95", "outside_train_q99"] if c in f]
        f = f[keep].drop_duplicates("source_snapshot")
        merged = m.merge(f, on="source_snapshot", how="left", suffixes=("", "_frame"))
    else:
        merged = m.copy()
    for col in ["distance_to_train", "distance_to_neb", "outside_train_q95", "outside_train_q99", "pca1", "pca2"]:
        if f"{col}_frame" in merged:
            merged[col] = merged[col].fillna(merged[f"{col}_frame"])
    metrics = ["distance_to_train", "distance_to_neb", "pca1", "pca2"]
    rows = []
    for keys, g in merged.groupby(["candidate_group", "window"], dropna=False):
        row = {"candidate_group": keys[0], "window": keys[1], "n_candidates": int(len(g))}
        for col in metrics:
            row[f"{col}_median"] = q(g[col], 0.50) if col in g else np.nan
            row[f"{col}_q95"] = q(g[col], 0.95) if col in g else np.nan
        for col in ["outside_train_q95", "outside_train_q99"]:
            vals = g[col].astype(str).str.lower().isin(["true", "1", "yes"])
            row[f"{col}_fraction"] = float(vals.mean()) if len(vals) else np.nan
        rows.append(row)
    stage = pd.DataFrame(rows).sort_values(["candidate_group", "window"])

    ref_rows = []
    for name, g in reference.groupby("reference", dropna=False):
        ref_rows.append({
            "reference": name,
            "n": int(len(g)),
            "pca1_median": q(g.pca1, .5), "pca1_q05": q(g.pca1, .05), "pca1_q95": q(g.pca1, .95),
            "pca2_median": q(g.pca2, .5), "pca2_q05": q(g.pca2, .05), "pca2_q95": q(g.pca2, .95),
        })
    ref_summary = pd.DataFrame(ref_rows).sort_values("reference")

    dist = {"candidate_reference_join_rows": int(merged["pca1"].notna().sum()) if "pca1" in merged else 0,
            "candidate_rows": int(len(merged)), "reference_rows": int(len(reference))}
    if merged[[c for c in ["pca1", "pca2"] if c in merged]].dropna().shape[0] and {"pca1", "pca2"}.issubset(reference.columns):
        cxy = merged[["pca1", "pca2"]].dropna().to_numpy(float)
        rxy = reference[["pca1", "pca2"]].dropna().to_numpy(float)
        lo = np.minimum(cxy.min(axis=0), rxy.min(axis=0))
        hi = np.maximum(cxy.max(axis=0), rxy.max(axis=0))
        bins = [np.linspace(lo[i], hi[i], 21) for i in range(2)]
        ch, _, _ = np.histogram2d(cxy[:, 0], cxy[:, 1], bins=bins)
        rh, _, _ = np.histogram2d(rxy[:, 0], rxy[:, 1], bins=bins)
        dist["pca_2d_js_divergence_candidate_vs_all_reference"] = js_divergence(ch, rh)
        dist["candidate_pca1_range"] = [float(x) for x in [cxy[:, 0].min(), cxy[:, 0].max()]]
        dist["candidate_pca2_range"] = [float(x) for x in [cxy[:, 1].min(), cxy[:, 1].max()]]
        dist["reference_pca1_range"] = [float(x) for x in [rxy[:, 0].min(), rxy[:, 0].max()]]
        dist["reference_pca2_range"] = [float(x) for x in [rxy[:, 1].min(), rxy[:, 1].max()]]
    return stage, ref_summary, dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--new-results", required=True)
    ap.add_argument("--from-scratch-results", required=True)
    ap.add_argument("--frame-features", required=True)
    ap.add_argument("--reference-embedding", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    new = add_common_columns(pd.read_csv(args.new_results))
    old = add_common_columns(pd.read_csv(args.from_scratch_results))
    # Keep the candidate/source fields from the canonical manifest as the sole source of stage labels.
    source_cols = ["candidate_id", "source", "run_key", "onset_ps", "trigger", "candidate_group", "window", "training_seed"]
    source = manifest[source_cols].drop_duplicates("candidate_id").rename(columns={"training_seed": "source_training_seed"})
    results = pd.concat([old, new], ignore_index=True, sort=False)
    results = results.drop(columns=[c for c in results.columns if c.endswith("_x") or c.endswith("_y")], errors="ignore")
    results = results.drop(columns=[c for c in ["source", "run_key", "onset_ps", "trigger", "candidate_group", "window", "training_seed"] if c in results.columns], errors="ignore")
    results = results.merge(source, on="candidate_id", how="left", validate="many_to_one")
    results = results.sort_values(["candidate_id", "model_id"]).reset_index(drop=True)
    results.to_csv(out / "combined_results_12models.csv", index=False)
    force_stats = (results[results.evaluation_status == "success"]
                   .groupby("candidate_id")["max_force_eV_A"]
                   .agg(max_force_min_eV_A="min", max_force_max_eV_A="max", max_force_median_eV_A="median", n_models="count")
                   .reset_index())
    force_stats["max_force_range_eV_A"] = force_stats.max_force_max_eV_A - force_stats.max_force_min_eV_A
    force_stats["max_force_ratio_max_min"] = force_stats.max_force_max_eV_A / force_stats.max_force_min_eV_A.replace(0, np.nan)
    force_stats = force_stats.merge(source, on="candidate_id", how="left", validate="one_to_one")
    force_stats.to_csv(out / "candidate_force_spread_12models.csv", index=False)
    model_summary = model_stage_summary(results)
    model_summary.to_csv(out / "model_stage_summary_12models.csv", index=False)
    coverage = candidate_coverage(manifest)
    coverage.to_csv(out / "candidate_source_coverage.csv", index=False)
    frame = pd.read_csv(args.frame_features)
    reference = pd.read_csv(args.reference_embedding)
    stage_dist, reference_summary, dist_meta = distribution_audit(manifest, frame, reference)
    stage_dist.to_csv(out / "candidate_distribution_by_stage.csv", index=False)
    reference_summary.to_csv(out / "reference_distribution_summary.csv", index=False)

    model_counts = results.groupby("model_id")["candidate_id"].nunique().to_dict()
    pair_counts = results.groupby(["candidate_id", "model_id"]).size()
    checks = {
        "status": "VERIFIED" if len(manifest) == 639 and results["candidate_id"].nunique() == 639 and pair_counts.max() == 1 and all(v == 639 for v in model_counts.values()) else "PROVENANCE_PARTIAL",
        "n_candidates": int(manifest.candidate_id.nunique()), "n_models": int(results.model_id.nunique()),
        "n_result_rows": int(len(results)), "model_candidate_counts": {str(k): int(v) for k, v in model_counts.items()},
        "duplicate_candidate_model_pairs": int((pair_counts > 1).sum()),
        "failed_rows": int((results.evaluation_status != "success").sum()),
        "finite_success_rows": int(((results.evaluation_status == "success") & results.forces_finite.astype(bool)).sum()),
        "candidate_manifest_sha256": sha256(Path(args.manifest)),
        "new_results_sha256": sha256(Path(args.new_results)),
        "from_scratch_results_sha256": sha256(Path(args.from_scratch_results)),
        "frame_features_sha256": sha256(Path(args.frame_features)),
        "reference_embedding_sha256": sha256(Path(args.reference_embedding)),
        "distribution": dist_meta,
    }
    (out / "combined_analysis_manifest.json").write_text(json.dumps(checks, indent=2, default=str) + "\n")
    print(json.dumps({k: checks[k] for k in ["status", "n_candidates", "n_models", "n_result_rows", "failed_rows", "duplicate_candidate_model_pairs"]}, indent=2))


if __name__ == "__main__":
    main()
