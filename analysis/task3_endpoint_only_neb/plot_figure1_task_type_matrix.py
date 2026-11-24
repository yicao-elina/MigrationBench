"""Re-render the Cr/Cu/Li/Na task-type matrix with the shared Plot Atlas style."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, "/Users/alina/Documents/Codex/plot_library/src")
from plot_atlas import apply_publication_style, panel_label, save_figure, timecourse
sys.path.insert(0, str(ROOT))
from neb_plot_style import QE_COLOR, model_colors, model_label


OUT = ROOT / "exploratory_energy_audit" / "figure1_task_type_matrix"
CR = OUT / "cr_foundation_path_image_profiles.csv"
CURRENT = ROOT / "exploratory_energy_audit" / "corrected_dft_foundation_models" / "corrected_dft_foundation_image_profiles.csv"

CR_MODELS = [
    "agnesi_stress_large", "agnesi_stress_medium", "agnesi_stress_small", "agnesi_medium",
    "mh0", "mh1", "mp_0b3_medium", "mpa0_medium", "omat0_medium", "omat0_small",
]
CURRENT_MODELS = [
    "mace-large-density-agnesi-stress", "mace-medium-density-agnesi-stress",
    "mace-small-density-agnesi-stress", "mace_agnesi_medium", "mace-mh-0", "mace-mh-1",
    "mace-mp-0b3-medium", "mace-omat-0-medium", "mace-omat-0-small",
]
CURRENT_LABELS = {
    "mace-large-density-agnesi-stress": "Agnesi-L-dens", "mace-medium-density-agnesi-stress": "Agnesi-M-dens",
    "mace-small-density-agnesi-stress": "Agnesi-S-dens", "mace_agnesi_medium": "Agnesi-M",
    "mace-mh-0": "MH-0", "mace-mh-1": "MH-1", "mace-mp-0b3-medium": "MP-0B3",
    "mace-omat-0-medium": "OMAT-0-M", "mace-omat-0-small": "OMAT-0-S",
}


def clean(ax):
    ax.grid(False)
    for spine in ax.spines.values(): spine.set_linewidth(.75)
    ax.tick_params(width=.75, length=3, direction="out", pad=2)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()): label.set_fontweight("bold")


def draw_series(ax, x, qe, model_arrays, labels, colors, title):
    ys = [qe] + model_arrays
    cols = [QE_COLOR] + colors
    names = ["QE-NEB path table (screened)"] + labels
    timecourse(ax, x, ys, names, cols, markers=True, legend=False, line_width=.95,
               marker_size=2.6, xlabel="Image index", ylabel="ΔE (eV)")
    ax.set_title(title, fontsize=7.5, pad=3)
    clean(ax)


def draw_class_schematic(ax, kind: str, label: str) -> None:
    """Draw a deliberately generic, geometry-agnostic path-class schematic."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    slab = "#C8C7C2"
    gap = "#F3F0EA"
    ion = {"within": "#B2768E", "gap": "#7697A8", "gap2ql": "#8B9270"}[kind]
    ax.add_patch(plt.Rectangle((.08, .15), .84, .18, facecolor=slab, edgecolor="#777777", lw=.7))
    ax.add_patch(plt.Rectangle((.08, .67), .84, .18, facecolor=slab, edgecolor="#777777", lw=.7))
    ax.add_patch(plt.Rectangle((.08, .33), .84, .34, facecolor=gap, edgecolor="none", zorder=-1))
    ax.text(.11, .50, "gap", fontsize=5.5, color="#777777", va="center")
    if kind == "within":
        start, end = (.24, .76), (.76, .76)
        curve = "arc3,rad=0"
    elif kind == "gap":
        start, end = (.24, .50), (.76, .50)
        curve = "arc3,rad=0"
    else:
        start, end = (.24, .50), (.76, .76)
        curve = "arc3,rad=-.12"
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
                                 connectionstyle=curve, lw=1.35, color=ion))
    ax.add_patch(Circle(start, .035, facecolor=ion, edgecolor="white", lw=.7, zorder=3))
    ax.add_patch(Circle(end, .035, facecolor=ion, edgecolor="white", lw=.7, zorder=3))
    ax.text(.5, .95, "schematic", ha="center", va="top", fontsize=5.7, color="#666666")
    ax.text(.5, .04, label, ha="center", va="bottom", fontsize=7.0, color="#333333")


def main():
    apply_publication_style(7.8)
    cr = pd.read_csv(CR)
    cur = pd.read_csv(CURRENT)
    fig = plt.figure(figsize=(11.8, 12.7))
    gs = fig.add_gridspec(4, 4, left=.055, right=.985, bottom=.12, top=.91, wspace=.38, hspace=.52)
    def cell(index: int):
        return gs[index // 4, index % 4]

    schematics = [("within", "within-QL"), ("gap", "in-gap diffusion"), ("gap2ql", "gap → QL")]
    for i, (kind, label) in enumerate(schematics):
        ax = fig.add_subplot(gs[0, i])
        draw_class_schematic(ax, kind, label)
        panel_label(ax, chr(ord("a") + i), x=-.16, y=1.12)

    panel_specs = [
        ("Cu QLs", "2L-Cu__neb-QLs"), ("Li QLs", "2L-Li__neb-QLs"),
        ("Cu gap-diff.", "2L-Cu__neb-gap-diffuse"), ("Li gap-diff.", "2L-Li__neb-gap-diffuse"),
        ("Na gap-diff.", "2L-Na__neb-gap-diffuse"), ("Cu gap2QL", "2L-Cu__neb-gap2QL"),
        ("Li gap2QL", "2L-Li__neb-gap2Ql"), ("Na gap2QL", "2L-Na__neb-gap2QL"),
    ]
    current_colors = model_colors(CURRENT_MODELS)
    for j, (title, path) in enumerate(panel_specs, start=3):
        ax = fig.add_subplot(cell(j))
        q = cur[cur.path_id.eq(path)].sort_values(["image_index", "model_id"])
        qe = q.drop_duplicates("image_index").sort_values("image_index").qe_relative_energy_eV.to_numpy()
        arrays = [q[q.model_id.eq(m)].sort_values("image_index").relative_energy_eV.to_numpy() for m in CURRENT_MODELS]
        draw_series(ax, np.arange(1, len(qe) + 1), qe, arrays, [CURRENT_LABELS[m] for m in CURRENT_MODELS], current_colors, title)
        panel_label(ax, chr(ord("a") + j), x=-.16, y=1.12)

    cr_paths = [("Cr gap2QL (path 1-2)", "1-2"), ("Cr gap2QL (path 1-3)", "1-3"),
                ("Cr gap2QL (path 1-4)", "1-4"), ("Cr gap2QL (path 1-5)", "1-5"),
                ("Cr gap2QL (path 1-7)", "1-7")]
    cr_colors = model_colors(CR_MODELS)
    for j, (title, path) in enumerate(cr_paths, start=11):
        ax = fig.add_subplot(cell(j))
        q = cr[cr.path_id.eq(path)].sort_values(["image_index", "model_id"])
        qe = q[q.model_id.eq("QE")].sort_values("image_index").relative_energy_eV.to_numpy()
        arrays = [q[q.model_id.eq(m)].sort_values("image_index").relative_energy_eV.to_numpy() for m in CR_MODELS]
        draw_series(ax, np.arange(1, len(qe) + 1), qe, arrays, [model_label(m) for m in CR_MODELS], cr_colors, title)
        panel_label(ax, chr(ord("a") + j), x=-.16, y=1.12)

    # Hide the unused final cell while keeping the 4x4 story compact.
    ax_unused = fig.add_subplot(cell(15)); ax_unused.axis("off")
    handles = [plt.Line2D([], [], color=QE_COLOR, lw=1.0, marker="o", ms=3, label="QE-NEB path table (screened)")]
    handles += [plt.Line2D([], [], color=c, lw=1.0, marker="o", ms=2.8, label=model_label(m)) for m, c in zip(CR_MODELS, cr_colors)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .035), ncol=5,
               frameon=False, fontsize=6.2, handletextpad=.35, columnspacing=1.1)
    fig.suptitle("Foundation-model energy profiles by dopant and migration-path class", fontsize=11, y=.975)
    stem = OUT / "figure1_task_type_matrix"
    save_figure(fig, stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight"); fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    (OUT / "figure1_task_type_matrix_caption.md").write_text(
        """# Figure 1 task-type matrix

**Figure 1 | Foundation-model energy profiles stratified by dopant and migration-path class.** The three upper-left cells define the generic within-QL, in-gap-diffusion, and gap-to-QL path classes schematically; they are not atom-resolved structural renderings. Remaining panels show screened QE path-table points and line segments together with Foundation-model single-point profiles. The ten Cr Foundation models are colored with related Morandi shades within the Agnesi, MH, MP, and OMAT families. The QE `error (eV/A)` residual is not converted into an energy band; it is reported in the separate NEB residual-audit figure.
"""
    )
    print({"output": str(stem.with_suffix('.png')), "cr_models": len(CR_MODELS), "current_models": len(CURRENT_MODELS)})


if __name__ == "__main__": main()
