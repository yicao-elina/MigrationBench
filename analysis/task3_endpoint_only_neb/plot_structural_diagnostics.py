"""Plot stage-aware geometry and model-disagreement diagnostics with Plot Atlas."""
from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/Users/alina/Documents/Codex/plot_library/src")
from plot_atlas import apply_publication_style, panel_label, save_figure, timecourse

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("STRUCT_PLOT_DATA", str(ROOT / "exploratory_energy_audit/structural_diagnostics")))
OUT = Path(os.environ.get("STRUCT_PLOT_OUT", str(DATA)))
STAGE_COLORS = {
    "endpoint": "#202020", "near_endpoint": "#327BB4",
    "transition_state_window": "#C12F55", "interior": "#178F78",
}
SYSTEM_COLORS = {"Cu": "#C12F55", "Li": "#327BB4", "Na": "#178F78"}


def short_path(value: str) -> str:
    return (value.replace("2L-", "").replace("__neb-", " ").replace("__neb", " ")
            .replace("gap-diffuse", "gap-diff.").replace("gap2Ql", "gap2QL"))


def clean(ax):
    ax.grid(False)
    for spine in ax.spines.values(): spine.set_linewidth(.8)
    ax.tick_params(width=.8, length=3, direction="out", pad=2)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()): label.set_fontweight("bold")


def main() -> None:
    apply_publication_style(8.2)
    d = pd.read_csv(DATA / "qe_path_structure_stage_metrics.csv")
    emb = pd.read_csv(DATA / "neb_descriptor_embedding.csv")
    d = d.merge(emb[["path_id", "image_index", "pca1", "pca2", "tsne1", "tsne2"]],
                on=["path_id", "image_index"], how="left", validate="one_to_one")
    d_with_models = d.dropna(subset=["foundation_range_eV"])

    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.9))
    # (a) all retained and rejected path-table traces; rejected paths are kept
    # for provenance but are not silently promoted to a reference set.
    for system, color in SYSTEM_COLORS.items():
        for path, q in d[d.system.eq(f"2L-{system}")].groupby("path_id"):
            q = q.sort_values("image_index")
            axes[0, 0].plot(q.image_index, q.qe_relative_energy_eV, color=color, alpha=.55,
                            lw=.9, marker="o", ms=2.2)
    axes[0, 0].set_xlabel("Image index"); axes[0, 0].set_ylabel("QE path-table ΔE (eV)")
    axes[0, 0].set_title("All 10 retained-snapshot paths", fontsize=8.7)
    axes[0, 0].text(.03, .95, "color = species", transform=axes[0, 0].transAxes, fontsize=6.5, va="top")
    # (b) local geometric coordinate versus QE energy, with stage explicit.
    for stage, q in d.groupby("stage"):
        axes[0, 1].scatter(q.dopant_displacement_from_image1_A, q.qe_relative_energy_eV,
                           s=17, color=STAGE_COLORS[stage], edgecolor="white", linewidth=.25,
                           label=stage.replace("_", " "))
    axes[0, 1].set_xlabel("Dopant displacement from image 1 (Å)")
    axes[0, 1].set_ylabel("QE path-table ΔE (eV)")
    axes[0, 1].set_title("Path energy versus local displacement", fontsize=8.7)
    axes[0, 1].legend(frameon=False, fontsize=5.6, loc="upper left")
    # (c) model spread is evaluated only where the 9-model same-geometry
    # matrix exists (8 eligible paths).
    for stage, q in d_with_models.groupby("stage"):
        axes[0, 2].scatter(q.dopant_nearest_host_A, q.foundation_range_eV,
                           s=18, color=STAGE_COLORS[stage], edgecolor="white", linewidth=.25,
                           label=stage.replace("_", " "))
    axes[0, 2].set_xlabel("Nearest host distance (Å)")
    axes[0, 2].set_ylabel("9-model Foundation range (eV)")
    axes[0, 2].set_title("Model disagreement by local environment", fontsize=8.7)
    # (d) PCA descriptor embedding.
    for stage, q in emb.groupby("stage"):
        axes[1, 0].scatter(q.pca1, q.pca2, s=18, color=STAGE_COLORS[stage],
                           edgecolor="white", linewidth=.25, label=stage.replace("_", " "))
    axes[1, 0].set_xlabel("PCA 1"); axes[1, 0].set_ylabel("PCA 2")
    axes[1, 0].set_title("Geometry/energy descriptor PCA", fontsize=8.7)
    # (e) t-SNE descriptor embedding, with identical stage coding.
    for stage, q in emb.groupby("stage"):
        axes[1, 1].scatter(q.tsne1, q.tsne2, s=18, color=STAGE_COLORS[stage],
                           edgecolor="white", linewidth=.25, label=stage.replace("_", " "))
    axes[1, 1].set_xlabel("t-SNE 1"); axes[1, 1].set_ylabel("t-SNE 2")
    axes[1, 1].set_title("Geometry/energy descriptor t-SNE", fontsize=8.7)
    # (f) stage-level distribution of model disagreement.
    stages = ["endpoint", "near_endpoint", "interior", "transition_state_window"]
    values = [d_with_models[d_with_models.stage.eq(s)].foundation_range_eV.dropna().to_numpy() for s in stages]
    axes[1, 2].boxplot(values, positions=np.arange(len(stages)), widths=.55, patch_artist=True,
                       boxprops={"facecolor": "white", "edgecolor": "#333333", "linewidth": .8},
                       medianprops={"color": "#C12F55", "linewidth": 1.0},
                       whiskerprops={"color": "#333333", "linewidth": .8},
                       capprops={"color": "#333333", "linewidth": .8},
                       flierprops={"marker": "o", "markersize": 2.3, "markerfacecolor": "#999999", "markeredgecolor": "none"})
    axes[1, 2].set_xticks(np.arange(len(stages)), ["endpoint", "near\nendpoint", "interior", "TS\nwindow"])
    axes[1, 2].set_ylabel("9-model Foundation range (eV)")
    axes[1, 2].set_title("Stage-level disagreement", fontsize=8.7)

    for ax in axes.flat: clean(ax)
    for i, ax in enumerate(axes.flat): panel_label(ax, chr(ord("a") + i), x=-.16, y=1.12)
    fig.subplots_adjust(left=.08, right=.98, bottom=.11, top=.93, wspace=.46, hspace=.57)
    stem = OUT / "figure_structural_stage_diagnostics"
    save_figure(fig, stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    (OUT / "structural_diagnostics_paper_text.md").write_text(
        """# Structural-stage diagnostics

**Figure X | Stage-aware structural and model-disagreement diagnostics along the retained QE path snapshots.** (a) QE path-table profiles for all ten available paths, retaining the two paths that fail the exploratory final-table stability screen for provenance. (b) Relative path energy versus dopant displacement from image 1, with endpoint, near-endpoint, transition-state-window, and interior frames distinguished. (c) Range across the nine same-geometry Foundation-model profiles versus the nearest host distance. (d,e) PCA and t-SNE embeddings of standardized geometry/energy descriptors; these are NEB-frame-only embeddings and do not claim a training/test distribution. (f) Stage-level distribution of the Foundation-model range.

The stage analysis is geometric and diagnostic rather than a claim of equilibrium or a converged transition state. The transition-state window is defined operationally as the maximum QE path-table image plus or minus one image, and the QE values remain screened path-table references because the NEB logs retain SCF warnings. The same-geometry nine-model range is available for eight paths; the two excluded paths are retained in panel (a) but are not imputed in the model-disagreement panels.
"""
    )
    print({"frames": len(d), "model_frames": len(d_with_models), "output": str(OUT / "figure_structural_stage_diagnostics.png")})


if __name__ == "__main__":
    main()
