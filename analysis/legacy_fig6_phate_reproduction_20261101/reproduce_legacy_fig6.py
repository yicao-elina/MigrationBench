#!/usr/bin/env python3
"""Reproduce the legacy NeurIPS workshop Fig. 6 (t-SNE/PHATE, four panels).

This is a data-preserving wrapper around the historical analysis in
MigrationBench commit 61c29dd.  The historical script rebuilt the 6192-column
feature matrix from four 600 K XYZ trajectories.  This entry point consumes the
committed feature-matrix artifact directly, so the projection and plotting
stages can be rerun without access to the original cluster filesystem.

The output is explicitly labelled RECONSTRUCTION because the original cluster
environment (Python/scikit-learn/PHATE versions) is not recreated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import sklearn
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score

import phate


MODEL_ORDER = ("FT - 600K", "FT - Multi-T", "Foundation", "Scratch")
INPUT_MODEL_ORDER = ("Foundation", "Scratch", "FT - 600K", "FT - Multi-T")
MODEL_COLORS = {
    "FT - 600K": "#002D72",
    "FT - Multi-T": "#68ACE5",
    "Foundation": "#CF4520",
    "Scratch": "#FF9E1B",
}
JHU_BLUE = LinearSegmentedColormap.from_list(
    "jhu_blue",
    [MODEL_COLORS["FT - 600K"], MODEL_COLORS["FT - Multi-T"]],
)
ENERGY_TICKS = (-4e6, -3e6, -2e6, -1e6)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    features = np.loadtxt(path, delimiter=",")
    if features.shape != (404, 6192):
        raise ValueError(f"expected historical feature matrix (404, 6192), got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("feature matrix contains non-finite values")
    labels = np.repeat(INPUT_MODEL_ORDER, 101)
    return features, labels


def project(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # sklearn renamed n_iter to max_iter after the historical run.  The value
    # and all other historical parameters are kept unchanged.
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(features) - 1),
        random_state=42,
        max_iter=1000,
    )
    tsne_embedding = tsne.fit_transform(features)
    phate_operator = phate.PHATE(random_state=42, n_jobs=-1)
    phate_embedding = phate_operator.fit_transform(features)
    return tsne_embedding, phate_embedding


def make_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def scatter_by_model(ax, embedding: np.ndarray, labels: np.ndarray, *, legend_loc: str) -> None:
    for model in MODEL_ORDER:
        mask = labels == model
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            label=model,
            color=MODEL_COLORS[model],
            alpha=0.7,
            s=20,
            edgecolors="none",
        )
    ax.legend(title="Model Type", fontsize=10, loc=legend_loc)


def style_panel(ax, letter: str) -> None:
    ax.text(-0.16, 1.08, f"{letter})", transform=ax.transAxes, fontsize=18,
            fontweight="bold", va="bottom")


def save_panel(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_panels(
    features: np.ndarray,
    labels: np.ndarray,
    tsne_embedding: np.ndarray,
    phate_embedding: np.ndarray,
    output_dir: Path,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    make_style()
    scores = silhouette_samples(tsne_embedding, labels)
    overall = float(silhouette_score(tsne_embedding, labels))
    per_model = {model: float(scores[labels == model].mean()) for model in MODEL_ORDER}

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    scatter_by_model(ax, tsne_embedding, labels, legend_loc="lower right")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    style_panel(ax, "a")
    fig.tight_layout()
    save_panel(fig, output_dir / "fig6_panel_a_tsne")

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    scatter_by_model(ax, phate_embedding, labels, legend_loc="upper right")
    ax.set_xlabel("PHATE Dimension 1")
    ax.set_ylabel("PHATE Dimension 2")
    style_panel(ax, "b")
    fig.tight_layout()
    save_panel(fig, output_dir / "fig6_panel_b_phate")

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    image = ax.scatter(
        phate_embedding[:, 0],
        phate_embedding[:, 1],
        c=features[:, 0],
        cmap=JHU_BLUE,
        s=10,
        alpha=0.8,
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_ticks(ENERGY_TICKS)
    colorbar.set_label("Potential Energy (eV)")
    ax.set_xlabel("PHATE Dimension 1")
    ax.set_ylabel("PHATE Dimension 2")
    style_panel(ax, "c")
    fig.tight_layout()
    save_panel(fig, output_dir / "fig6_panel_c_phate_energy")

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.bar(MODEL_ORDER, [per_model[model] for model in MODEL_ORDER],
           color=[MODEL_COLORS[model] for model in MODEL_ORDER])
    ax.axhline(overall, color="#D43E1F", linestyle="--",
               label=f"Overall Avg: {overall:.3f}")
    ax.legend(loc="upper right")
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.set_ylabel("Avg. Silhouette Score")
    ax.set_ylim(min(list(per_model.values()) + [-0.1]) * 1.1, 1.0)
    style_panel(ax, "d")
    fig.tight_layout()
    save_panel(fig, output_dir / "fig6_panel_d_silhouette")

    # Attachment-style horizontal composition.  Each panel remains available
    # independently above; this figure owns layout only.
    fig, axes = plt.subplots(1, 4, figsize=(20.0, 5.1),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.05]})
    scatter_by_model(axes[0], tsne_embedding, labels, legend_loc="lower right")
    axes[0].set_xlabel("t-SNE Dimension 1")
    axes[0].set_ylabel("t-SNE Dimension 2")
    scatter_by_model(axes[1], phate_embedding, labels, legend_loc="upper right")
    axes[1].set_xlabel("PHATE Dimension 1")
    axes[1].set_ylabel("PHATE Dimension 2")
    image = axes[2].scatter(phate_embedding[:, 0], phate_embedding[:, 1],
                            c=features[:, 0], cmap=JHU_BLUE, s=10, alpha=0.8)
    colorbar = fig.colorbar(image, ax=axes[2], pad=0.04)
    colorbar.set_ticks(ENERGY_TICKS)
    colorbar.set_label("Potential Energy (eV)")
    axes[2].set_xlabel("PHATE Dimension 1")
    axes[2].set_ylabel("PHATE Dimension 2")
    axes[3].bar(MODEL_ORDER, [per_model[model] for model in MODEL_ORDER],
                color=[MODEL_COLORS[model] for model in MODEL_ORDER])
    axes[3].axhline(overall, color="#D43E1F", linestyle="--",
                    label=f"Overall Avg: {overall:.3f}")
    axes[3].legend(loc="upper right")
    axes[3].tick_params(axis="x", rotation=30, labelsize=10)
    axes[3].set_ylabel("Avg. Silhouette Score")
    axes[3].set_ylim(min(list(per_model.values()) + [-0.1]) * 1.1, 1.0)
    for axis, letter in zip(axes, "abcd"):
        style_panel(axis, letter)
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.18, top=0.97,
                        wspace=0.42)
    fig.savefig(output_dir / "fig6_phate_reproduced_1x4.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig6_phate_reproduced_1x4.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"overall": overall, **per_model}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    features, labels = load_features(args.features)
    tsne_embedding, phate_embedding = project(features)
    scores = render_panels(features, labels, tsne_embedding, phate_embedding, args.output_dir)
    manifest = {
        "status": "RECONSTRUCTION",
        "historical_commit": "61c29dd012bbf794213aa92238943afad9177ad7",
        "historical_script": "results/t-SNE/phate-tsne/tsne.py",
        "historical_script_blob": "0717cbe90dc0be62cc73eeb6bc573ae857e32492",
        "feature_matrix_git_source": {
            "repository": "/Users/alina/Project/migrationbench-paper-repro",
            "commit": "61c29dd012bbf794213aa92238943afad9177ad7",
            "path": "results/t-SNE/phate-tsne/analysis_Cr_Doped_600K_v2/Cr_Doped_Sb2Te3_600K_probe_features.csv",
            "blob": "cae0107d7afbd3511cefaf9a9cef09e43527a982",
        },
        "features_path": str(args.features.resolve()),
        "features_sha256": sha256(args.features),
        "features_shape": list(features.shape),
        "model_block_order_in_input": ["Foundation", "Scratch", "FT - 600K", "FT - Multi-T"],
        "model_plot_order": list(MODEL_ORDER),
        "tsne": {"perplexity": 30, "random_state": 42, "max_iter": 1000},
        "phate": {"random_state": 42, "n_jobs": -1, "package": "phate"},
        "environment": {
            "python": __import__("sys").version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "phate": phate.__version__,
            "matplotlib": plt.matplotlib.__version__,
        },
        "silhouette_space": "2D t-SNE embedding",
        "scores": scores,
        "diagnostics": [
            "PHATE reported 2 disconnected graph components with the historical default knn=5.",
            "PHATE metric MDS reported a small final-iteration stress change warning.",
        ],
        "outputs": [
            "fig6_panel_a_tsne.pdf/png",
            "fig6_panel_b_phate.pdf/png",
            "fig6_panel_c_phate_energy.pdf/png",
            "fig6_panel_d_silhouette.pdf/png",
            "fig6_phate_reproduced_1x4.pdf/png",
        ],
        "caveat": "Numerical/visual parity is reconstruction-level because the original cluster environment is not recreated.",
    }
    (args.output_dir / "reproduction_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
