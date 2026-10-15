from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def q(s, p):
    x = pd.to_numeric(s, errors="coerce").dropna()
    return float(x.quantile(p)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--frame-features", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(args.manifest)
    f = pd.read_csv(args.frame_features).rename(columns={"snapshot_path": "source_snapshot"})
    r = pd.read_csv(args.reference)
    merged = m.merge(f[[c for c in ["source_snapshot", "pca1", "pca2", "distance_to_train", "distance_to_neb", "outside_train_q95", "outside_train_q99"] if c in f]], on="source_snapshot", how="left", suffixes=("", "_frame"))
    for col in ["pca1", "pca2", "distance_to_train", "distance_to_neb", "outside_train_q95", "outside_train_q99"]:
        if f"{col}_frame" in merged:
            merged[col] = merged[col].fillna(merged[f"{col}_frame"])
    by_run = merged.groupby(["source", "run_key", "training_seed"], dropna=False).agg(n_candidates=("candidate_id", "size"), n_groups=("candidate_group", "nunique"), first_time_ps=("time_ps", "min"), last_time_ps=("time_ps", "max")).reset_index()
    by_group = merged.groupby(["candidate_group", "window"], dropna=False).agg(n_candidates=("candidate_id", "size"), n_runs=("run_key", "nunique"), n_sources=("source", "nunique"), time_ps_median=("time_ps", "median"), time_ps_q25=("time_ps", lambda x: q(x, .25)), time_ps_q75=("time_ps", lambda x: q(x, .75)), distance_to_train_median=("distance_to_train", "median"), distance_to_train_q95=("distance_to_train", lambda x: q(x, .95)), distance_to_neb_median=("distance_to_neb", "median"), pca1_median=("pca1", "median"), pca2_median=("pca2", "median")).reset_index()
    novelty = merged.groupby("candidate_group", dropna=False).agg(n=("candidate_id", "size"), train_q95_outside_fraction=("outside_train_q95", lambda x: x.astype(str).str.lower().isin(["true", "1", "yes"]).mean()), train_q99_outside_fraction=("outside_train_q99", lambda x: x.astype(str).str.lower().isin(["true", "1", "yes"]).mean()), distance_to_train_median=("distance_to_train", "median"), distance_to_train_q95=("distance_to_train", lambda x: q(x, .95)), distance_to_neb_median=("distance_to_neb", "median")).reset_index()
    ref = r.groupby("reference", dropna=False).agg(n=("reference", "size"), pca1_median=("pca1", "median"), pca1_q05=("pca1", lambda x: q(x,.05)), pca1_q95=("pca1", lambda x: q(x,.95)), pca2_median=("pca2", "median"), pca2_q05=("pca2", lambda x: q(x,.05)), pca2_q95=("pca2", lambda x: q(x,.95))).reset_index()
    by_run.to_csv(out / "candidate_coverage_by_run.csv", index=False)
    by_group.to_csv(out / "candidate_coverage_by_stage.csv", index=False)
    novelty.to_csv(out / "candidate_novelty_by_stage.csv", index=False)
    ref.to_csv(out / "reference_embedding_summary.csv", index=False)
    meta = {"status": "PREPRODUCTION_AUDIT", "n_candidates": int(len(m)), "n_runs": int(m.run_key.nunique()), "n_sources": int(m.source.nunique()), "n_reference_rows": int(len(r)), "reference_counts": r.reference.value_counts().to_dict(), "candidate_counts_by_source": m.source.value_counts().to_dict(), "candidate_counts_by_group": m.candidate_group.value_counts().to_dict(), "interpretation": "Stage-balanced failure-attribution sample, not time-weighted MD distribution; descriptor novelty is descriptive rather than causal."}
    (out / "source_distribution_preflight.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    main()
