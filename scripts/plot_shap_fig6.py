#!/usr/bin/env python3
"""Build a deterministic Fig. 6-style summary from revised SHAP rankings."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ranking", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = pd.read_csv(args.ranking)
    top = data[data["rank"] <= 20].copy()
    pivot = top.pivot_table(index="feature", columns="model", values="mean_abs_shap_eV", fill_value=0)
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1, 1.25]})
    best = top[top.model == "FT-MultiT"].sort_values("mean_abs_shap_eV").tail(20)
    axes[0].barh(best.feature, best.mean_abs_shap_eV, color="#68ACE5")
    axes[0].set_title("FT-MultiT surrogate")
    axes[0].set_xlabel("mean |TreeSHAP| (eV)")
    image = axes[1].imshow(pivot.to_numpy(), aspect="auto", cmap="Blues")
    axes[1].set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    axes[1].set_yticks(range(len(pivot.index)), pivot.index)
    fig.colorbar(image, ax=axes[1], label="mean |TreeSHAP| (eV)")
    axes[1].set_title("Top features across models")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()
