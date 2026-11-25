#!/usr/bin/env python3
"""Re-render legacy Figure 6 with the current MigrationBench visual grammar.

The historical feature matrix, t-SNE parameters, PHATE parameters, and model
block order are preserved.  This script changes only the figure presentation:
Foundation / Fine-tuning / Scratch colors, open-versus-filled temperature
markers, Arial-first typography, heavier axes, and a readable shared legend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
import numpy as np
import phate
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score


ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURES = Path(
    "/private/tmp/migrationbench_legacy_fig6.dokELc/results/t-SNE/phate-tsne/"
    "analysis_Cr_Doped_600K_v2/Cr_Doped_Sb2Te3_600K_probe_features.csv"
)
OUT = ROOT / "updated_palette_20261102"

COLORS = {
    "Foundation": "#8FC7AB",
    "Fine-tuning": "#29A7B6",
    "Scratch": "#327FB1",
    "DFT": "#303840",
}
ENERGY_CMAP = LinearSegmentedColormap.from_list(
    "migrationbench_energy",
    ["#E8F1F5", "#BDD7E3", "#80B5CB", "#4382A7", "#174A70"],
)
MODEL_BLOCK_ORDER = ["Foundation", "Scratch", "FT - 600K", "FT - Multi-T"]
MODEL_PLOT_ORDER = ["FT - 600K", "FT - Multi-T", "Foundation", "Scratch"]
BLOCK_SIZE = 101


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strategy(label: str) -> str:
    if label.startswith("FT"):
        return "Fine-tuning"
    return label


def marker(label: str) -> str:
    return "D" if label == "Foundation" else "o"


def temperature(label: str) -> str | None:
    if label == "FT - 600K":
        return "600 K"
    if label == "FT - Multi-T":
        return "Multi-T"
    return None


def marker_face(label: str, color: str) -> str:
    return "white" if temperature(label) == "600 K" else color


def configure_axes(ax, letter: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    # The letter is placed above the tick-label band so it cannot collide with
    # the highest y tick after final-size rasterization.
    ax.text(
        -0.14,
        1.14,
        letter,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )
    ax.tick_params(which="major", labelsize=11.5, width=1.2, length=5.0, direction="out")
    ax.tick_params(which="minor", width=0.9, length=3.0, direction="out")
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
        spine.set_color(COLORS["DFT"])
    ax.grid(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12.8, fontweight="bold", labelpad=5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12.8, fontweight="bold", labelpad=5)


def strategy_handles() -> list[Line2D]:
    return [
        Line2D([], [], marker="D", linestyle="None", markersize=7.0,
               markerfacecolor=COLORS["Foundation"], markeredgecolor=COLORS["Foundation"],
               label="Foundation"),
        Line2D([], [], marker="o", linestyle="None", markersize=7.0,
               markerfacecolor=COLORS["Fine-tuning"], markeredgecolor=COLORS["Fine-tuning"],
               label="Fine-tuning"),
        Line2D([], [], marker="o", linestyle="None", markersize=7.0,
               markerfacecolor=COLORS["Scratch"], markeredgecolor=COLORS["Scratch"],
               label="Scratch"),
    ]


def temperature_handles() -> list[Line2D]:
    return [
        Line2D([], [], marker="o", linestyle="None", markersize=7.0,
               markerfacecolor="white", markeredgecolor=COLORS["DFT"],
               markeredgewidth=1.1, label="600 K"),
        Line2D([], [], marker="o", linestyle="None", markersize=7.0,
               markerfacecolor=COLORS["DFT"], markeredgecolor=COLORS["DFT"],
               label="Multi-T"),
    ]


def scatter_by_model(ax, embedding: np.ndarray, labels: np.ndarray, *, size: float = 25.0, alpha: float = 0.84) -> None:
    for label in MODEL_PLOT_ORDER:
        mask = labels == label
        color = COLORS[strategy(label)]
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=size,
            marker=marker(label),
            facecolors=marker_face(label, color),
            edgecolors=color,
            linewidths=0.65,
            alpha=alpha,
            label=label,
            zorder=3,
        )


def energy_scatter(ax, embedding: np.ndarray, labels: np.ndarray, energies: np.ndarray, norm: Normalize) -> None:
    point_colors = ENERGY_CMAP(norm(energies))
    for label in MODEL_PLOT_ORDER:
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=25,
            marker=marker(label),
            facecolors="none" if temperature(label) == "600 K" else point_colors[mask],
            edgecolors=point_colors[mask],
            linewidths=0.65,
            alpha=0.88,
            zorder=3,
        )


def add_model_legends(
    fig,
    *,
    y: float = 0.035,
    strategy_x: float = 0.24,
    temperature_x: float = 0.67,
    strategy_y: float | None = None,
    temperature_y: float | None = None,
    fontsize: float = 11.6,
) -> None:
    strategy_y = y if strategy_y is None else strategy_y
    temperature_y = y if temperature_y is None else temperature_y
    fig.legend(
        handles=strategy_handles(),
        loc="lower center",
        bbox_to_anchor=(strategy_x, strategy_y),
        ncol=3,
        frameon=False,
        fontsize=fontsize,
        title="Training strategy",
        title_fontsize=fontsize,
        handletextpad=0.45,
        columnspacing=1.0,
    )
    fig.legend(
        handles=temperature_handles(),
        loc="lower center",
        bbox_to_anchor=(temperature_x, temperature_y),
        ncol=2,
        frameon=False,
        fontsize=fontsize,
        title="Temperature",
        title_fontsize=fontsize,
        handletextpad=0.45,
        columnspacing=1.0,
    )


def panel_a(ax, tsne: np.ndarray, labels: np.ndarray) -> None:
    scatter_by_model(ax, tsne, labels)
    ax.set_xlim(-36, 54)
    ax.set_ylim(-9.2, 9.2)
    configure_axes(ax, "a", "t-SNE Dimension 1", "t-SNE Dimension 2")


def panel_b(ax, phate_embedding: np.ndarray, labels: np.ndarray) -> None:
    scatter_by_model(ax, phate_embedding, labels)
    ax.set_xlim(-0.085, 0.035)
    ax.set_ylim(-0.0175, 0.0165)
    configure_axes(ax, "b", "PHATE Dimension 1", "PHATE Dimension 2")


def panel_c(ax, cbar_ax, phate_embedding: np.ndarray, labels: np.ndarray, energies: np.ndarray) -> None:
    norm = Normalize(vmin=float(np.min(energies)), vmax=float(np.max(energies)))
    energy_scatter(ax, phate_embedding, labels, energies, norm)
    ax.set_xlim(-0.085, 0.035)
    ax.set_ylim(-0.0175, 0.0165)
    configure_axes(ax, "c", "PHATE Dimension 1", "PHATE Dimension 2")
    sm = ScalarMappable(norm=norm, cmap=ENERGY_CMAP)
    sm.set_array(energies)
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Potential energy (10^6 eV)", fontsize=12.0, fontweight="bold", labelpad=7)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e6:g}"))
    cbar.ax.tick_params(labelsize=10.5, width=1.0, length=4.0)
    for text in cbar.ax.get_yticklabels():
        text.set_fontweight("bold")
    cbar.outline.set_linewidth(1.0)


def panel_d(ax, silhouette_by_model: dict[str, float], overall: float) -> None:
    values = [silhouette_by_model[label] for label in MODEL_PLOT_ORDER]
    bar_colors = [
        "white",
        COLORS["Fine-tuning"],
        COLORS["Foundation"],
        COLORS["Scratch"],
    ]
    edge_colors = [
        COLORS["Fine-tuning"],
        COLORS["Fine-tuning"],
        COLORS["Foundation"],
        COLORS["Scratch"],
    ]
    ax.bar(
        np.arange(4),
        values,
        color=bar_colors,
        edgecolor=edge_colors,
        linewidth=1.1,
        width=0.72,
        zorder=3,
    )
    ax.axhline(overall, color=COLORS["DFT"], linestyle=(0, (3, 2)), linewidth=1.25,
               label=f"Overall average: {overall:.3f}", zorder=2)
    ax.set_xticks(np.arange(4), ["FT-600 K", "FT-Multi-T", "Foundation", "Scratch"], rotation=25, ha="right")
    ax.set_ylim(-0.10, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    configure_axes(ax, "d", None, "Avg. silhouette score")
    ax.tick_params(axis="x", labelsize=10.8, pad=4)
    for text in ax.get_xticklabels():
        text.set_fontweight("bold")
    ax.legend(loc="upper right", frameon=False, fontsize=11.6, handlelength=1.7, handletextpad=.45)


def save(fig, path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=320 if path.suffix == ".png" else None, bbox_inches="tight", facecolor="white")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def render(features_path: Path) -> dict[str, object]:
    features = np.loadtxt(features_path, delimiter=",")
    if features.shape != (404, 6192):
        raise ValueError(f"Expected (404, 6192) features, got {features.shape}")
    labels = np.repeat(MODEL_BLOCK_ORDER, BLOCK_SIZE)
    energies = features[:, 0]

    tsne_embedding = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        max_iter=1000,
    ).fit_transform(features)
    phate_embedding = phate.PHATE(random_state=42, n_jobs=-1).fit_transform(features)
    overall = float(silhouette_score(tsne_embedding, labels))
    sample_scores = silhouette_samples(tsne_embedding, labels)
    silhouette_by_model = {
        label: float(np.mean(sample_scores[labels == label])) for label in MODEL_PLOT_ORDER
    }

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "fig6_palette_updated_embeddings.npz",
        tsne=tsne_embedding,
        phate=phate_embedding,
        labels=labels,
        energies=energies,
    )

    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "font.weight": "normal",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 1.25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })

    outputs: dict[str, object] = {}
    fig = plt.figure(figsize=(1521 / 72, 393 / 72), facecolor="white")
    gs = fig.add_gridspec(1, 5, width_ratios=[1.35, 1.25, 1.28, .11, .98], wspace=.60)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    cbar_ax = fig.add_subplot(gs[0, 3])
    ax_d = fig.add_subplot(gs[0, 4])
    panel_a(ax_a, tsne_embedding, labels)
    panel_b(ax_b, phate_embedding, labels)
    panel_c(ax_c, cbar_ax, phate_embedding, labels, energies)
    panel_d(ax_d, silhouette_by_model, overall)
    fig.subplots_adjust(left=.065, right=.985, bottom=.27, top=.86)
    add_model_legends(fig)
    for ext in ("png", "pdf", "svg"):
        outputs[f"main_{ext}"] = save(fig, OUT / f"fig6_phate_palette_updated_1x4.{ext}")
    plt.close(fig)

    single_specs = [
        ("a_tsne", (5.4, 5.2), lambda fig: panel_a(fig.add_subplot(111), tsne_embedding, labels)),
        ("b_phate", (5.4, 5.2), lambda fig: panel_b(fig.add_subplot(111), phate_embedding, labels)),
        ("c_phate_energy", (5.7, 5.2), None),
        ("d_silhouette", (5.0, 5.2), lambda fig: panel_d(fig.add_subplot(111), silhouette_by_model, overall)),
    ]
    for name, size, renderer in single_specs:
        fig = plt.figure(figsize=size, facecolor="white")
        if name == "c_phate_energy":
            gs_single = fig.add_gridspec(1, 2, width_ratios=[1, .08], wspace=.35)
            ax = fig.add_subplot(gs_single[0, 0])
            cax = fig.add_subplot(gs_single[0, 1])
            panel_c(ax, cax, phate_embedding, labels, energies)
            fig.subplots_adjust(left=.18, right=.93, bottom=.30, top=.86)
            fig.legend(temperature_handles(), ["600 K", "Multi-T"], loc="lower center", bbox_to_anchor=(.53, .055), ncol=2, frameon=False, fontsize=11.6)
        else:
            renderer(fig)
            fig.subplots_adjust(left=.18, right=.95, bottom=.34, top=.86)
            if name in {"a_tsne", "b_phate"}:
                add_model_legends(
                    fig,
                    strategy_x=.50,
                    temperature_x=.50,
                    strategy_y=.115,
                    temperature_y=.035,
                )
        for ext in ("png", "pdf", "svg"):
            outputs[f"{name}_{ext}"] = save(fig, OUT / f"fig6_panel_{name}_palette_updated.{ext}")
        plt.close(fig)

    manifest = {
        "status": "RENDERED_WITH_CURRENT_MIGRATIONBENCH_VISUAL_STANDARD",
        "historical_reference_manifest": str(ROOT / "reproduced/reproduction_manifest.json"),
        "features_path": str(features_path),
        "features_sha256": sha256(features_path),
        "features_shape": list(features.shape),
        "model_block_order_in_input": MODEL_BLOCK_ORDER,
        "model_plot_order": MODEL_PLOT_ORDER,
        "tsne": {"perplexity": 30, "random_state": 42, "max_iter": 1000},
        "phate": {"random_state": 42, "n_jobs": -1},
        "silhouette_space": "2D t-SNE embedding",
        "scores": {"overall": overall, **silhouette_by_model},
        "visual_encoding": {
            "strategy_colors": COLORS,
            "temperature_fill": {"600 K": "open", "Multi-T": "filled"},
            "foundation_marker": "diamond",
            "energy_palette": ["#E8F1F5", "#BDD7E3", "#80B5CB", "#4382A7", "#174A70"],
            "axes_font": "Arial",
            "axis_label_pt": 12.8,
            "tick_label_pt": 11.5,
            "legend_pt": 11.6,
        },
        "outputs": outputs,
        "boundary": "The historical feature matrix and dimensionality-reduction parameters are unchanged; this is an additive visual-standard update.",
    }
    (OUT / "fig6_phate_palette_updated_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    args = parser.parse_args()
    manifest = render(args.features.resolve())
    print(json.dumps({"output": str(OUT), "overall_silhouette": manifest["scores"]["overall"], "files": len(manifest["outputs"])}, indent=2))


if __name__ == "__main__":
    main()
