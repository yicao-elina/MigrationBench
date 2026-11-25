#!/usr/bin/env python3
"""Reconstruct the two-panel revised MigrationBench Fig. 6 from a ranking table.

The historical panel-rendering source is not present in the surviving release.
The feature orders below are transcribed from the locked Fig. 6 PDF, while all
bar/heatmap values are read from the supplied ranking CSV.  This is therefore
an explicit reconstruction, not a claim of byte-identical historical rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PANEL_A_ORDER = [
    "Cr-Cr_n11_l4", "Sb-Te_n44_l2", "Cr-Sb_n12_l3", "Cr-Sb_n24_l1",
    "Cr-Cr_n13_l0", "Cr-Te_n24_l3", "Cr-Cr_n12_l1", "Sb-Sb_n44_l2",
    "Cr-Cr_n23_l4", "Sb-Sb_n34_l0", "Cr-Sb_n23_l1", "Cr-Cr_n22_l3",
    "Sb-Te_n23_l1", "Te-Te_n33_l0", "Cr-Cr_n34_l0", "Sb-Te_n41_l0",
    "Cr-Sb_n33_l0", "Te-Te_n11_l0", "Cr-Sb_n33_l3", "Cr-Te_n33_l1",
    "Cr-Te_n34_l3", "Cr-Sb_n44_l0", "Te-Te_n11_l4", "Sb-Sb_n44_l1",
    "Cr-Te_n23_l2", "Te-Te_n22_l0", "Te-Te_n44_l2", "Cr-Cr_n12_l4",
    "Te-Te_n12_l4", "Te-Te_n44_l1", "Te-Te_n23_l4", "Sb-Te_n43_l2",
    "Cr-Te_n11_l2", "Cr-Cr_n11_l1", "Te-Te_n34_l0", "Te-Te_n13_l2",
    "Te-Te_n33_l4", "Sb-Te_n44_l1", "Cr-Te_n22_l2", "Cr-Te_n21_l2",
    "Cr-Te_n13_l2", "Cr-Sb_n43_l1", "Cr-Cr_n44_l0", "Cr-Cr_n34_l2",
    "Cr-Cr_n14_l2", "Cr-Te_n32_l2", "Cr-Cr_n24_l2", "Cr-Te_n31_l2",
    "Cr-Te_n33_l2", "Cr-Sb_n43_l0",
]

PANEL_B_ORDER = [
    "Cr-Sb_n43_l0", "Cr-Cr_n24_l2", "Cr-Te_n31_l2", "Cr-Te_n33_l2",
    "Cr-Te_n32_l2", "Cr-Cr_n14_l2", "Te-Te_n44_l2", "Cr-Sb_n13_l0",
    "Sb-Te_n23_l1", "Cr-Cr_n34_l2", "Cr-Te_n21_l2", "Sb-Te_n12_l0",
    "Te-Te_n12_l0", "Cr-Sb_n43_l1", "Cr-Cr_n44_l0", "Cr-Cr_n23_l3",
    "Sb-Sb_n12_l0", "Cr-Te_n13_l2", "Sb-Te_n44_l4", "Sb-Sb_n33_l4",
    "Sb-Sb_n11_l0", "Cr-Cr_n22_l0", "Te-Te_n13_l2", "Te-Te_n11_l0",
    "Cr-Te_n22_l2", "Te-Te_n33_l4", "Sb-Te_n44_l1", "Cr-Cr_n11_l3",
    "Cr-Cr_n11_l1", "Sb-Te_n21_l0", "Sb-Sb_n11_l4", "Te-Te_n44_l0",
    "Sb-Te_n43_l2", "Sb-Sb_n14_l4", "Cr-Te_n24_l2", "Cr-Te_n42_l0",
    "Te-Te_n34_l0", "Sb-Sb_n14_l0", "Cr-Te_n11_l2", "Sb-Sb_n12_l4",
    "Cr-Sb_n44_l0", "Te-Te_n23_l4", "Cr-Cr_n34_l3", "Te-Te_n44_l1",
    "Sb-Te_n24_l4", "Te-Te_n11_l4", "Sb-Te_n21_l1", "Te-Te_n12_l4",
    "Cr-Cr_n12_l4", "Sb-Te_n32_l0",
]

COLORS = {"FT-MultiT": "#2A7A73", "Scratch": "#D86343"}
MODELS = ("FT-600K", "FT-MultiT", "Scratch")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_values(path: Path) -> dict[tuple[str, str], float]:
    data = pd.read_csv(path)
    required = {"model", "rank", "feature", "mean_abs_shap_eV"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"ranking CSV missing columns: {sorted(missing)}")
    values = {}
    for row in data.itertuples(index=False):
        if int(row.rank) <= 50:
            values[(row.model, row.feature)] = float(row.mean_abs_shap_eV)
    return values


def validate_orders(values: dict[tuple[str, str], float]) -> None:
    missing_a = [f for f in PANEL_A_ORDER if not any((m, f) in values for m in MODELS)]
    missing_b = [f for f in PANEL_B_ORDER if not any((m, f) in values for m in MODELS)]
    if missing_a or missing_b:
        raise ValueError(f"reference labels absent from ranking CSV: a={missing_a}, b={missing_b}")


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", labelsize=9, width=1.0, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def draw_panel_a(ax: plt.Axes, values: dict[tuple[str, str], float]) -> None:
    x = np.arange(len(PANEL_A_ORDER))
    best = np.array([values.get(("FT-MultiT", f), 0.0) for f in PANEL_A_ORDER])
    worst = np.array([values.get(("Scratch", f), 0.0) for f in PANEL_A_ORDER])
    width = 0.36
    ax.bar(x - width / 2, best, width=width, color=COLORS["FT-MultiT"],
           label="FT - Multi-T (best)")
    ax.bar(x + width / 2, worst, width=width, color=COLORS["Scratch"],
           label="Scratch (worst)")
    ax.set_title("Top 50 features: best vs worst model", fontsize=16, pad=11)
    ax.set_ylabel("Mean |SHAP value| (eV)", fontsize=14)
    ax.set_xlabel("SOAP feature (physical label)", fontsize=13, labelpad=12)
    ax.set_xticks(x, PANEL_A_ORDER, rotation=90, ha="center", fontsize=7.2)
    ax.set_ylim(0, 0.08)
    ax.set_yticks(np.arange(0, 0.08, 0.01))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=11, frameon=True, borderpad=0.55)
    style_axes(ax)


def draw_panel_b(ax: plt.Axes, values: dict[tuple[str, str], float]) -> plt.cm.ScalarMappable:
    matrix = np.array([
        [values.get((model, feature), 0.0) for feature in PANEL_B_ORDER]
        for model in ("FT-600K", "FT-MultiT", "Scratch")
    ])
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=0.076)
    ax.set_title("Top 50 SOAP feature importance across models (error prediction)",
                 fontsize=16, pad=11)
    ax.set_xticks(np.arange(len(PANEL_B_ORDER)), PANEL_B_ORDER, rotation=90,
                  ha="center", fontsize=7.2)
    ax.set_yticks(np.arange(3), ["FT - 600K", "FT - Multi-T", "Scratch"], fontsize=12)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=4, width=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    return image


def render(ranking: Path, output_dir: Path) -> dict[str, str]:
    values = load_values(ranking)
    validate_orders(values)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_a, ax_a = plt.subplots(figsize=(9.0, 7.1))
    draw_panel_a(ax_a, values)
    fig_a.tight_layout()
    fig_a.savefig(output_dir / "fig6_panel_a_reconstructed.pdf", bbox_inches="tight")
    fig_a.savefig(output_dir / "fig6_panel_a_reconstructed.png", dpi=300, bbox_inches="tight")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(11.2, 7.1))
    image = draw_panel_b(ax_b, values)
    fig_b.colorbar(image, ax=ax_b, pad=0.025, label="Mean |SHAP value| (eV)")
    fig_b.tight_layout()
    fig_b.savefig(output_dir / "fig6_panel_b_reconstructed.pdf", bbox_inches="tight")
    fig_b.savefig(output_dir / "fig6_panel_b_reconstructed.png", dpi=300, bbox_inches="tight")
    plt.close(fig_b)

    fig, axes = plt.subplots(1, 2, figsize=(16.11, 7.12),
                             gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.18})
    draw_panel_a(axes[0], values)
    image = draw_panel_b(axes[1], values)
    fig.colorbar(image, ax=axes[1], pad=0.025, label="Mean |SHAP value| (eV)")
    fig.suptitle("SHAP-based dissection of MACE prediction errors (revised pipeline)",
                 fontsize=21, fontweight="bold", y=0.995)
    axes[0].text(-0.08, 1.08, "(a)", transform=axes[0].transAxes,
                 fontsize=18, fontweight="bold", va="bottom")
    axes[1].text(-0.08, 1.08, "(b)", transform=axes[1].transAxes,
                 fontsize=18, fontweight="bold", va="bottom")
    fig.subplots_adjust(left=0.045, right=0.975, bottom=0.18, top=0.84,
                        wspace=0.18)
    combined = output_dir / "fig6_shap_reproduced_from_locked_table.pdf"
    fig.savefig(combined, bbox_inches="tight")
    fig.savefig(output_dir / "fig6_shap_reproduced_from_locked_table.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "status": "RECONSTRUCTION",
        "ranking_csv": str(ranking.resolve()),
        "ranking_sha256": sha256(ranking),
        "panel_a_features": PANEL_A_ORDER,
        "panel_b_features": PANEL_B_ORDER,
        "models": list(MODELS),
        "notes": [
            "Historical two-panel plotting source was not present in the surviving release.",
            "Feature orders were transcribed from the locked Fig. 6 PDF; values come from the ranking CSV.",
            "This output is not asserted byte-identical to the historical PDF.",
        ],
    }
    (output_dir / "reproduction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {"combined": str(combined), "manifest": str(output_dir / "reproduction_manifest.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.ranking, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
