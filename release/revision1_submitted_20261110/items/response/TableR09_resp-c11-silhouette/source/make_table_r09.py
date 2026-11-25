#!/usr/bin/env python3
"""Response Table R9 (tab:resp-c11-silhouette): original-space and t-SNE silhouettes.

Original-space rows from fig6_current_original_space_metrics.csv; t-SNE rows from
fig6_current_tsne_sensitivity.csv collapsed over the embedding seeds (the check below
confirms the seeds give identical values). Both CSVs are written by
release/revision1_camera_ready/reproducibility/recompute_feature_metrics.py (init='pca').
Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import Evidence, fmt_int, fmt_num, fmt_sci  # noqa: E402

RELEASE = "paper/revision1/data_processed/current_release/"
ORIGINAL = RELEASE + "fig6_current_original_space_metrics.csv"
TSNE = RELEASE + "fig6_current_tsne_sensitivity.csv"
SPACE_LABEL = {  # data `space` value -> submitted row label
    "Original output features (6191)": "Original output features, Euclidean",
    "Original output features, cosine": "Original output features, cosine",
    "Z-scored output features": "Z-scored output features",
    "Common neighbour histogram (30)": "Common neighbour histogram",
    "Z-scored neighbour histogram": "Z-scored neighbour histogram",
}
TSNE_LABEL = "$t$-SNE, PCA init."
NO_VALUE = "---"
SILHOUETTE_DIGITS = 2
TRUST_DECIMALS = 2


def main() -> int:
    ev = Evidence(__file__, "tab:resp-c11-silhouette")
    original = pd.read_csv(ev.input(ORIGINAL))
    tsne = pd.read_csv(ev.input(TSNE))
    ev.check("label map covers every original-space row", set(original["space"]) == set(SPACE_LABEL),
             SPACE_LABEL)
    spread = tsne.groupby("perplexity")[["silhouette", "trustworthiness"]].agg(lambda s: s.max() - s.min())
    ev.check("t-SNE values are identical across embedding seeds", bool((spread == 0).all().all()),
             {"seeds": sorted(tsne["seed"].unique().tolist()), "max_range": float(spread.to_numpy().max())})

    rows = [[SPACE_LABEL[r.space], fmt_int(r.dimensions), fmt_sci(r.silhouette, SILHOUETTE_DIGITS), NO_VALUE]
            for r in original.itertuples()]
    collapsed = tsne.groupby("perplexity", sort=True)[["silhouette", "trustworthiness"]].first()
    rows += [[TSNE_LABEL, f"perplexity {perplexity}", fmt_sci(r.silhouette, SILHOUETTE_DIGITS),
              fmt_num(r.trustworthiness, TRUST_DECIMALS)] for perplexity, r in collapsed.iterrows()]
    ev.compare_table(rows, "response_to_reviewers.tex")
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
