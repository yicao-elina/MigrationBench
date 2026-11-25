"""Summarize the ten-path Foundation/QE experiment by migration-path class."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/Users/alina/Documents/Codex/plot_library/src")
from plot_atlas import apply_publication_style, panel_label, save_figure

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "exploratory_energy_audit"
IMG = AUDIT / "all_dft_foundation_models/corrected_dft_foundation_image_profiles.csv"
STRUCT = AUDIT / "all_path_structural_diagnostics/qe_path_structure_stage_metrics.csv"
QUAL = AUDIT / "all_path_diagnostic_framework/dft_reference_qualification.csv"
OUT = AUDIT / "all_path_diagnostic_framework"
OUT.mkdir(parents=True, exist_ok=True)

TYPE_ORDER = ["in_gap_diffusion", "gap_to_QL", "within_QL", "unclassified_control"]
TYPE_LABEL = {"in_gap_diffusion": "in-gap diffusion", "gap_to_QL": "gap → QL", "within_QL": "within QL", "unclassified_control": "control / unspecified"}
TYPE_COLOR = {"in_gap_diffusion": "#327BB4", "gap_to_QL": "#178F78", "within_QL": "#C12F55", "unclassified_control": "#6B6B6B"}
SYSTEM_COLOR = {"2L-Cu": "#C12F55", "2L-Li": "#327BB4", "2L-Na": "#178F78"}


def path_type(path_id: str) -> str:
    if "gap-diffuse" in path_id: return "in_gap_diffusion"
    if "gap2Q" in path_id or "gap2q" in path_id: return "gap_to_QL"
    if "QLs" in path_id: return "within_QL"
    return "unclassified_control"


def short(value: str) -> str:
    return value.replace("2L-", "").replace("__neb-", " ").replace("__neb", " ").replace("gap-diffuse", "gap-diff.").replace("gap2Ql", "gap2QL")


def clean(ax):
    ax.grid(False)
    for s in ax.spines.values(): s.set_linewidth(.8)
    ax.tick_params(width=.8, length=3, direction="out", pad=2)
    for lab in (*ax.get_xticklabels(), *ax.get_yticklabels()): lab.set_fontweight("bold")


def main():
    apply_publication_style(8.2)
    image = pd.read_csv(IMG)
    image["path_type"] = image.path_id.map(path_type)
    structural = pd.read_csv(STRUCT)
    structural["path_type"] = structural.path_id.map(path_type)
    quality = pd.read_csv(QUAL)
    quality["path_type"] = quality.path_id.map(path_type)

    barriers = image.groupby(["model_id", "path_id", "path_type"], as_index=False).agg(
        foundation_barrier_like_eV=("relative_energy_eV", "max"), qe_barrier_like_eV=("qe_relative_energy_eV", "max"))
    barriers["barrier_error_eV"] = barriers.foundation_barrier_like_eV - barriers.qe_barrier_like_eV
    barriers["abs_barrier_error_eV"] = barriers.barrier_error_eV.abs()
    qbar = barriers.drop_duplicates(["path_id", "path_type"])
    qbar = qbar.merge(quality[["path_id", "reference_tier", "max_abs_image_energy_delta_eV"]], on="path_id", how="left")
    barriers.to_csv(OUT / "path_type_model_barriers.csv", index=False)

    stage = image.merge(structural[["path_id", "image_index", "stage"]], on=["path_id", "image_index"], how="left", validate="many_to_one")
    stage["path_type"] = stage.path_id.map(path_type)
    stage["abs_error_eV"] = (stage.relative_energy_eV - stage.qe_relative_energy_eV).abs()
    stage_summary = stage.groupby(["path_type", "stage"], as_index=False).agg(n=("abs_error_eV", "size"), mae_eV=("abs_error_eV", "mean"), median_abs_error_eV=("abs_error_eV", "median"))
    stage_summary.to_csv(OUT / "path_type_stage_error_summary.csv", index=False)

    type_summary = barriers.groupby("path_type", as_index=False).agg(
        n_paths=("path_id", "nunique"), median_qe_barrier_eV=("qe_barrier_like_eV", "median"),
        median_abs_barrier_error_eV=("abs_barrier_error_eV", "median"), max_abs_barrier_error_eV=("abs_barrier_error_eV", "max"),
        median_model_range_eV=("foundation_barrier_like_eV", lambda x: float(x.max() - x.min())))
    type_summary.to_csv(OUT / "path_type_summary.csv", index=False)
    type_summary.to_latex(OUT / "path_type_summary.tex", index=False, escape=False, float_format=lambda x: f"{x:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.0))
    xpos = np.arange(len(TYPE_ORDER))
    # (a) QE path-table difficulty with unstable paths retained as open markers.
    qbar_unique = qbar.drop_duplicates("path_id")
    for path_class in TYPE_ORDER:
        subset = qbar_unique[qbar_unique.path_type.eq(path_class)].reset_index(drop=True)
        offsets = np.linspace(-.10, .10, len(subset)) if len(subset) > 1 else np.array([0.0])
        for offset, (_, row) in zip(offsets, subset.iterrows()):
            x = TYPE_ORDER.index(row.path_type)
            system = next((key for key in SYSTEM_COLOR if row.path_id.startswith(key)), "")
            color = SYSTEM_COLOR.get(system, "#555555")
            filled = row.reference_tier == "exploratory_fixed_geometry"
            axes[0, 0].scatter(x + offset, row.qe_barrier_like_eV,
                               s=35, color=color if filled else "white", edgecolor=color, linewidth=.9, zorder=3)
            axes[0, 0].text(x + offset, row.qe_barrier_like_eV, short(row.path_id), fontsize=4.7, ha="center", va="bottom")
    axes[0, 0].set_xticks(xpos, [TYPE_LABEL[x] for x in TYPE_ORDER], rotation=25, ha="right")
    axes[0, 0].set_ylabel("QE path-table barrier-like ΔE (eV)")
    axes[0, 0].set_title("Three canonical migration classes plus control", fontsize=8.8)
    axes[0, 0].set_yscale("log")

    # (b) Foundation barrier errors, all nine models and all ten paths.
    vals = [barriers[barriers.path_type.eq(t)].abs_barrier_error_eV.to_numpy() for t in TYPE_ORDER]
    axes[0, 1].boxplot(vals, positions=xpos, widths=.55, patch_artist=True,
                       boxprops={"facecolor": "white", "edgecolor": "#333333"}, medianprops={"color": "#C12F55"},
                       whiskerprops={"color": "#333333"}, capprops={"color": "#333333"},
                       flierprops={"marker": "o", "markersize": 2.2, "markerfacecolor": "#999999", "markeredgecolor": "none"})
    for i, t in enumerate(TYPE_ORDER):
        axes[0, 1].scatter(np.full(len(vals[i]), i), vals[i], s=11, color=TYPE_COLOR[t], alpha=.55, edgecolor="white", linewidth=.2)
    axes[0, 1].set_xticks(xpos, [TYPE_LABEL[x] for x in TYPE_ORDER], rotation=25, ha="right")
    axes[0, 1].set_ylabel("Foundation |barrier error| (eV)")
    axes[0, 1].set_title("Model error by path class", fontsize=8.8)
    axes[0, 1].set_yscale("log")

    # (c) Stage decomposition by path type.
    heat = stage_summary.pivot(index="path_type", columns="stage", values="mae_eV").reindex(index=TYPE_ORDER, columns=["endpoint", "near_endpoint", "interior", "transition_state_window"])
    im = axes[1, 0].imshow(heat.to_numpy(), aspect="auto", cmap="magma", vmin=0, vmax=np.nanpercentile(heat.to_numpy(), 95))
    axes[1, 0].set_xticks(np.arange(4), ["endpoint", "near-endpoint", "interior", "TS window"], rotation=30, ha="right")
    axes[1, 0].set_yticks(np.arange(4), [TYPE_LABEL[x] for x in TYPE_ORDER])
    axes[1, 0].set_ylabel("Path class"); axes[1, 0].set_title("Stage-specific error", fontsize=8.8)
    cbar = fig.colorbar(im, ax=axes[1, 0], fraction=.046, pad=.04); cbar.set_label("MAE (eV)")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            if np.isfinite(heat.iloc[i, j]): axes[1, 0].text(j, i, f"{heat.iloc[i,j]:.2f}", ha="center", va="center", fontsize=5, color="white")

    # (d) final-table stability and model spread, keeping failed paths visible.
    spread = barriers.groupby(["path_id", "path_type"], as_index=False).foundation_barrier_like_eV.agg(lambda x: float(x.max() - x.min())).rename(columns={"foundation_barrier_like_eV": "model_range_eV"})
    q = qbar.drop_duplicates("path_id").merge(spread, on=["path_id", "path_type"])
    for _, row in q.iterrows():
        x = TYPE_ORDER.index(row.path_type)
        color = TYPE_COLOR[row.path_type]
        axes[1, 1].scatter(row.max_abs_image_energy_delta_eV, row.model_range_eV, s=36, color=color if row.reference_tier == "exploratory_fixed_geometry" else "white", edgecolor=color, linewidth=.9)
        axes[1, 1].text(row.max_abs_image_energy_delta_eV, row.model_range_eV, short(row.path_id), fontsize=4.7, ha="center", va="bottom")
    axes[1, 1].axvline(.1, color="#98489F", lw=.9, ls=(0, (3, 2)))
    axes[1, 1].set_xscale("log"); axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("max penultimate → final |ΔE| (eV)"); axes[1, 1].set_ylabel("9-model profile range (eV)")
    axes[1, 1].set_title("Reference stability versus model spread", fontsize=8.8)

    for ax in axes.flat: clean(ax)
    for i, ax in enumerate(axes.flat): panel_label(ax, chr(ord("a") + i), x=-.16, y=1.12)
    fig.subplots_adjust(left=.13, right=.98, bottom=.17, top=.93, wspace=.50, hspace=.58)
    stem = OUT / "figure_path_type_summary"
    save_figure(fig, stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight"); fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    (OUT / "path_type_summary_paper_text.md").write_text("""# Path-type summary\n\n**Figure X | A clean three-class view of the ten saved NEB path snapshots.** The canonical classes are in-gap diffusion, gap-to-QL migration, and within-QL migration. The Cu path with an unspecified `neb` label is retained as a separate control rather than forced into one of the three classes. Open markers denote paths that fail the exploratory final-table stability screen. All nine Foundation models are included in the error distributions.\n\nThe three-class organization separates the physical exam questions while preserving the two unstable paths as explicit evidence. It should be used for stratified diagnosis, not as a claim that the current QE values are converged reference barriers.\n""")
    print({"paths": int(image.path_id.nunique()), "models": int(image.model_id.nunique()), "output": str(stem.with_suffix('.png'))})


if __name__ == "__main__":
    main()
