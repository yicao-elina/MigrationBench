"""Rebuild the complete Foundation-model comparison on the retained QE path frames.

This is deliberately separate from the endpoint-only linear diagnostic.  Every
Foundation model and the QE reference in this script uses the same eight
retained, ten-image path geometries.  The QE path-table energies are converted
from Hartree to eV upstream and are reported as a screened diagnostic trace,
not as a converged DFT barrier.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "/Users/alina/Documents/Codex/plot_library/src")
from plot_atlas import apply_publication_style, get_palette, panel_label, save_figure, timecourse
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neb_plot_style import QE_COLOR, extract_qe_neb_uncertainty, model_colors

ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("FOUNDATION_RESULTS", "/tmp/migrationbench_foundation_on_corrected_dft_paths_20261003"))
QE_IMAGE = Path(os.environ.get("FOUNDATION_QE_IMAGE", str(ROOT / "exploratory_energy_audit/processed/mace_dft_relative_image_metrics.csv")))
OUT = Path(os.environ.get("FOUNDATION_OUT", str(ROOT / "exploratory_energy_audit/corrected_dft_foundation_models")))
EXPECTED_PATHS = int(os.environ.get("FOUNDATION_EXPECTED_PATHS", "8"))
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "mace-large-density-agnesi-stress", "mace-medium-density-agnesi-stress",
    "mace-small-density-agnesi-stress", "mace_agnesi_medium", "mace-mh-0",
    "mace-mh-1", "mace-mp-0b3-medium", "mace-omat-0-medium", "mace-omat-0-small",
]
LABELS = {
    "mace-large-density-agnesi-stress": "Agnesi-L-dens",
    "mace-medium-density-agnesi-stress": "Agnesi-M-dens",
    "mace-small-density-agnesi-stress": "Agnesi-S-dens",
    "mace_agnesi_medium": "Agnesi-M", "mace-mh-0": "MH-0", "mace-mh-1": "MH-1",
    "mace-mp-0b3-medium": "MP-0B3", "mace-omat-0-medium": "OMAT-0-M",
    "mace-omat-0-small": "OMAT-0-S",
}
COLORS = model_colors(MODEL_ORDER)


def short_path(value: str) -> str:
    return (value.replace("2L-", "").replace("__neb-", " ").replace("__neb", " ")
            .replace("gap-diffuse", "gap-diff.").replace("gap2Ql", "gap2QL"))


def load_data() -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        source = RESULTS / model / "mace_single_point_results.csv"
        d = pd.read_csv(source)
        d["model_id"] = model
        d["path_id"] = d["sample_file"].str.replace("__base.extxyz", "", regex=False)
        d["image_index"] = d["frame_index"] + 1
        d["relative_energy_eV"] = d["energy_eV"] - d.groupby("path_id")["energy_eV"].transform("first")
        rows.append(d)
    models = pd.concat(rows, ignore_index=True)
    qe = pd.read_csv(QE_IMAGE)
    if "variant" in qe.columns:
        qe = qe[qe["variant"].eq("base")][["path_id", "image_index", "dft_relative_energy_eV"]].drop_duplicates()
        qe = qe.rename(columns={"dft_relative_energy_eV": "qe_relative_energy_eV"})
    else:
        qe = qe[["path_id", "image_index", "qe_relative_energy_eV"]].drop_duplicates()
    d = models.merge(qe, on=["path_id", "image_index"], how="left", validate="many_to_one")
    uncertainty = extract_qe_neb_uncertainty(
        d.path_id.unique(),
        remote_root=ROOT / "remote_snapshot",
        n_images=10,
        n_tail_tables=5,
    )
    d = d.merge(
        uncertainty[["path_id", "image_index", "qe_energy_std_eV", "qe_energy_range_eV", "qe_force_error_eV_A",
                     "qe_last_iteration", "qe_n_complete_tables", "qe_uncertainty_status"]],
        on=["path_id", "image_index"], how="left", validate="many_to_one",
    )
    if len(d) != 9 * EXPECTED_PATHS * 10 or d.qe_relative_energy_eV.isna().any():
        raise ValueError(f"Expected 9 x {EXPECTED_PATHS} x 10 aligned Foundation/QE rows")
    if d.scientific_success.ne(True).any() or d.status.ne("success").any():
        raise ValueError("A Foundation single-point evaluation failed")
    return d


def make_tables(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d.to_csv(OUT / "corrected_dft_foundation_image_profiles.csv", index=False)
    barriers = d.groupby(["model_id", "path_id"], as_index=False)["relative_energy_eV"].max()
    barriers = barriers.rename(columns={"relative_energy_eV": "foundation_barrier_like_eV"})
    qe_barriers = d.groupby("path_id", as_index=False)["qe_relative_energy_eV"].max()
    qe_barriers = qe_barriers.rename(columns={"qe_relative_energy_eV": "qe_path_table_barrier_like_eV"})
    barriers = barriers.merge(qe_barriers, on="path_id", validate="many_to_one")
    barriers["signed_error_eV"] = barriers.foundation_barrier_like_eV - barriers.qe_path_table_barrier_like_eV
    barriers["absolute_error_eV"] = barriers.signed_error_eV.abs()
    barriers.to_csv(OUT / "corrected_dft_foundation_path_barriers.csv", index=False)

    summary = barriers.groupby("model_id").agg(
        median_foundation_barrier_like_eV=("foundation_barrier_like_eV", "median"),
        mean_foundation_barrier_like_eV=("foundation_barrier_like_eV", "mean"),
        median_signed_error_eV=("signed_error_eV", "median"),
        median_absolute_error_eV=("absolute_error_eV", "median"),
        max_absolute_error_eV=("absolute_error_eV", "max"),
    ).reset_index()
    pivot = d.pivot_table(index=["path_id", "image_index"], columns="model_id", values="relative_energy_eV")
    median_profile = pivot.median(axis=1)
    ensemble = pd.DataFrame({
        "model_id": MODEL_ORDER,
        "rmse_to_model_median_eV": [float(np.sqrt(np.mean((pivot[m] - median_profile) ** 2))) for m in MODEL_ORDER],
        "mae_to_model_median_eV": [float(np.mean(np.abs(pivot[m] - median_profile))) for m in MODEL_ORDER],
    })
    summary = summary.merge(ensemble, on="model_id")
    summary["model_label"] = summary.model_id.map(LABELS)
    summary = summary[["model_id", "model_label", "median_foundation_barrier_like_eV",
                       "mean_foundation_barrier_like_eV", "median_signed_error_eV",
                       "median_absolute_error_eV", "max_absolute_error_eV",
                       "rmse_to_model_median_eV", "mae_to_model_median_eV"]]
    summary = summary.sort_values("median_absolute_error_eV")
    summary.to_csv(OUT / "corrected_dft_foundation_summary.csv", index=False)
    summary.to_latex(OUT / "corrected_dft_foundation_summary.tex", index=False, index_names=False,
                     escape=False, float_format=lambda x: f"{x:.4f}")
    return barriers, summary


def render_profiles(d: pd.DataFrame) -> None:
    apply_publication_style(8.2)
    paths = sorted(d.path_id.unique())
    ncols = 4
    nrows = int(np.ceil(len(paths) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.55, 2.52 * nrows), sharey=False)
    axes = np.asarray(axes).reshape(-1)
    for ax, path in zip(axes, paths):
        q = d[d.path_id.eq(path)]
        xs = np.arange(1, 11)
        q_qe = q.drop_duplicates("image_index").sort_values("image_index")
        qe = q_qe.qe_relative_energy_eV.to_numpy()
        ys = [qe] + [q[q.model_id.eq(m)].sort_values("image_index").relative_energy_eV.to_numpy() for m in MODEL_ORDER]
        colors = [QE_COLOR] + COLORS
        labels = ["QE-NEB path table"] + [LABELS[m] for m in MODEL_ORDER]
        timecourse(ax, xs, ys, labels, colors, markers=True, legend=False, line_width=1.0,
                   marker_size=2.65, xlabel="Image index", ylabel="ΔE (eV)")
        ax.set_title(short_path(path), fontsize=8.2, pad=3)
        for spine in ax.spines.values(): spine.set_linewidth(.8)
    handles = [plt.Line2D([], [], color=QE_COLOR, lw=1.0, marker="o", ms=3.0,
                          mec=QE_COLOR, mfc=QE_COLOR, label="QE-NEB path table")]
    handles += [plt.Line2D([], [], color=COLORS[i], lw=1.0, marker="o", ms=2.7,
                           mec=COLORS[i], mfc=COLORS[i], label=LABELS[m]) for i, m in enumerate(MODEL_ORDER)]
    for ax in axes[len(paths):]:
        ax.set_visible(False)
    axes[len(paths) - 1].legend(handles=handles, frameon=False, fontsize=5.65, loc="upper left",
                         bbox_to_anchor=(1.02, 1.02), ncol=1)
    for i, ax in enumerate(axes[:len(paths)]): panel_label(ax, chr(ord("a") + i), x=-.16, y=1.12)
    fig.subplots_adjust(left=.08, right=.84, bottom=.11, top=.94, wspace=.40, hspace=.52)
    stem = OUT / "figure_corrected_dft_foundation_path_profiles"
    save_figure(fig, stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def render_barrier_comparison(barriers: pd.DataFrame, summary: pd.DataFrame) -> None:
    apply_publication_style(8.2)
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.35), gridspec_kw={"width_ratios": [1.42, 1]})
    order = sorted(barriers.path_id.unique())
    heat = barriers.pivot(index="model_id", columns="path_id", values="signed_error_eV").reindex(index=summary.model_id, columns=order)
    im = axes[0].imshow(heat.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-max(abs(heat.min().min()), abs(heat.max().max())), vmax=max(abs(heat.min().min()), abs(heat.max().max())))
    axes[0].set_xticks(np.arange(len(order)), [short_path(x) for x in order], rotation=55, ha="right")
    axes[0].set_yticks(np.arange(len(summary)), summary.model_label)
    axes[0].set_xlabel("Retained QE path")
    axes[0].set_ylabel("Foundation model")
    axes[0].set_title("Barrier-like error relative to QE path table", fontsize=8.8)
    cbar = fig.colorbar(im, ax=axes[0], fraction=.046, pad=.04)
    cbar.set_label("Foundation − QE (eV)")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            axes[0].text(j, i, f"{heat.iloc[i,j]:.2g}", ha="center", va="center", fontsize=4.8,
                         color="white" if abs(heat.iloc[i,j]) > .5 * np.nanmax(abs(heat.to_numpy())) else "black")
    # One bar per model: median absolute path error.  Use the bar color to
    # retain model identity; signed error is already explicit in panel (a).
    y = np.arange(len(summary))
    axes[1].barh(y, summary.median_absolute_error_eV.to_numpy(),
                 color=[COLORS[MODEL_ORDER.index(m)] for m in summary.model_id],
                 edgecolor="none", height=.62)
    axes[1].set_yticks(y, summary.model_label)
    axes[1].set_xlabel("Median |barrier error| (eV)")
    axes[1].set_title("Model-level error summary", fontsize=8.8)
    for spine in (*axes[0].spines.values(), *axes[1].spines.values()): spine.set_linewidth(.8)
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    fig.subplots_adjust(left=.20, right=.98, bottom=.30, top=.88, wspace=.48)
    stem = OUT / "figure_corrected_dft_foundation_barrier_comparison"
    save_figure(fig, stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    d = load_data()
    barriers, summary = make_tables(d)
    render_profiles(d)
    render_barrier_comparison(barriers, summary)
    model_files = []
    for model in MODEL_ORDER:
        path = Path("/Users/alina/Downloads/mace-foundation-models") / f"{model}.model"
        model_files.append({"model_id": model, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "head": "omat_pbe" if model in {"mace-mh-0", "mace-mh-1"} else None})
    manifest = {
        "analysis": "all_local_foundation_models_on_retained_corrected_qe_neb_path_frames",
        "n_models": len(MODEL_ORDER), "n_paths": int(d.path_id.nunique()), "n_images_per_path": 10,
        "n_evaluations": int(len(d)), "model_files": model_files,
        "geometry_alignment": "Every QE and Foundation value uses the same saved path frame, joined by path_id and image_index; unstable paths are retained with their qualification flags.",
        "energy_units": "sb2te3.path Hartree values converted to eV with 27.211386245988; Foundation energies are eV and normalized within path.",
        "qe_scope": "Screened QE path-table reference only; all ten NEB logs contain at least one SCF warning and no formal converged DFT barrier is claimed.",
        "endpoint_only_matrix": "The prior fresh endpoint-only linear matrix remains a separate diagnostic and is not mixed into this same-geometry result.",
    }
    (OUT / "corrected_dft_foundation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "corrected_dft_foundation_paper_text.md").write_text(
        f"""# Corrected same-geometry Foundation-model comparison

## Figure caption

**Figure X | Same-geometry comparison of the complete local MACE Foundation-model set with the retained QE path table.** Each panel shows ten images from one retained Cu/Li/Na path. The black trace is the QE path-table energy, shown with explicit points and line segments; each colored trace is a single-point energy profile from one of the nine local Foundation models evaluated on exactly the same geometries. Models are colored with related Morandi shades within each model family. QE `.path` energies were converted from Hartree to eV; Foundation energies were normalized within each path to image 1. The NEB image-table ``error`` column is a force/path residual in eV Å⁻¹, not an energy standard error, and is reported separately in the convergence-audit companion figure.

## Results and discussion

The corrected comparison removes two confounders in the earlier exploratory figure: the QE `.path` energy unit conversion and the mismatch between the retained QE geometries and fresh endpoint-only linear interpolation geometries. The resulting model spread is strongly path dependent. Some paths show close model/QE agreement, whereas the Cu QLs path retains a large Foundation-model elevation despite exact geometry alignment. This indicates a model-response or extrapolation sensitivity along that path rather than a plotting-scale artifact. The barrier-like error heatmap reports these differences quantitatively without interpreting them as converged activation barriers.

The QE traces remain screened path-table references, not converged DFT barriers: the NEB audit found at least one electronic SCF warning in every path, and the formal DFT gate is 0/10. Therefore the figure supports a controlled model-disagreement and alignment analysis, while physical barrier ranking requires converged QE/NEB reruns.

## Reproducibility

Nine models were evaluated successfully on {int(d.path_id.nunique())} paths × 10 images ({len(d)}/{9 * int(d.path_id.nunique()) * 10} evaluations). MH-0 and MH-1 used the explicit `omat_pbe` head. Input path identity is preserved through the processed path hashes and the path/image join. The previous endpoint-only linear matrix is retained separately and is not used as the QE comparison in this figure.
"""
    )
    print(json.dumps({"models": len(MODEL_ORDER), "paths": int(d.path_id.nunique()), "evaluations": len(d), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
