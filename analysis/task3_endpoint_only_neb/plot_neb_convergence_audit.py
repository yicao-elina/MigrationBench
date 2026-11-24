"""Plot the QE-NEB image-wise residual audit as a separate QA figure.

The energy-profile figures deliberately do not show the QE ``error`` column as
an energy uncertainty.  This companion figure displays that quantity in its
native unit (eV/A), together with the ``path_thr`` requested by each NEB
input.  It is therefore suitable for a convergence/quality-control supplement
and prevents a unit-mismatched uncertainty band in the main energy figure.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/Users/alina/Documents/Codex/plot_library/src")
from plot_atlas import apply_publication_style, panel_label, save_figure
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neb_plot_style import QE_COLOR, extract_qe_neb_uncertainty, read_qe_path_threshold


ROOT = Path(__file__).resolve().parent
REMOTE = ROOT / "remote_snapshot"
OUT = ROOT / "exploratory_energy_audit" / "neb_convergence_audit"
OUT.mkdir(parents=True, exist_ok=True)

PATHS = [
    "2L-Cu__neb", "2L-Cu__neb-QLs", "2L-Cu__neb-gap-diffuse", "2L-Cu__neb-gap2QL",
    "2L-Li__neb-QLs", "2L-Li__neb-gap-diffuse", "2L-Li__neb-gap2Ql",
    "2L-Na__neb-QLs", "2L-Na__neb-gap-diffuse", "2L-Na__neb-gap2QL",
]

SYSTEM_COLORS = {"2L-Cu": "#9B7772", "2L-Li": "#708A9A", "2L-Na": "#75988E"}


def short_path(path_id: str) -> str:
    return (path_id.replace("2L-", "").replace("__neb-", " ").replace("__neb", " ")
            .replace("gap-diffuse", "gap-diff.").replace("gap2Ql", "gap2QL"))


def main() -> None:
    apply_publication_style(8.2)
    d = extract_qe_neb_uncertainty(PATHS, remote_root=REMOTE, n_images=10, n_tail_tables=5)
    d["path_thr_eV_A"] = d.path_id.map(lambda p: read_qe_path_threshold(REMOTE, p))
    d["final_residual_over_threshold"] = d.qe_force_error_eV_A / d.path_thr_eV_A
    d.to_csv(OUT / "qe_neb_image_residual_audit.csv", index=False)

    summary = d.groupby("path_id", as_index=False).agg(
        path_thr_eV_A=("path_thr_eV_A", "first"),
        max_final_residual_eV_A=("qe_force_error_eV_A", "max"),
        median_final_residual_eV_A=("qe_force_error_eV_A", "median"),
        last_iteration=("qe_last_iteration", "first"),
        n_complete_tables=("qe_n_complete_tables", "first"),
    )
    summary["max_residual_over_threshold"] = summary.max_final_residual_eV_A / summary.path_thr_eV_A
    summary["passes_imagewise_gate"] = summary.max_final_residual_eV_A <= summary.path_thr_eV_A
    summary.to_csv(OUT / "qe_neb_path_residual_summary.csv", index=False)

    fig, axes = plt.subplots(2, 5, figsize=(9.2, 4.75), sharey=False)
    for ax, path_id in zip(axes.flat, PATHS):
        q = d[d.path_id.eq(path_id)].sort_values("image_index")
        color = SYSTEM_COLORS.get(path_id.split("__", 1)[0], QE_COLOR)
        ax.plot(q.image_index, q.qe_force_error_eV_A, color=color, lw=1.15, marker="o", ms=3.0,
                mfc=color, mec="white", mew=.35, zorder=3)
        threshold = float(q.path_thr_eV_A.iloc[0])
        ax.axhline(threshold, color="#595959", lw=.85, ls=(0, (3, 2)), zorder=1)
        ax.set_yscale("log")
        ax.set_xticks([1, 5, 10])
        ax.set_title(short_path(path_id), fontsize=7.8, pad=3)
        ax.set_xlabel("Image index")
        ax.set_ylabel(r"NEB residual (eV Å$^{-1}$)")
        ax.text(.97, .94, fr"thr={threshold:g}", transform=ax.transAxes, ha="right", va="top", fontsize=5.5,
                color="#595959")
        ax.grid(False)
        for spine in ax.spines.values(): spine.set_linewidth(.8)
        ax.tick_params(width=.8, length=3, direction="out", pad=2)
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()): label.set_fontweight("bold")

    for i, ax in enumerate(axes.flat):
        panel_label(ax, chr(ord("a") + i), x=-.16, y=1.12)
    fig.subplots_adjust(left=.08, right=.99, bottom=.14, top=.92, wspace=.48, hspace=.58)
    stem = OUT / "figure_neb_residual_audit"
    save_figure(fig, stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    (OUT / "figure_neb_residual_audit_caption.md").write_text(
        """# NEB residual audit

**Figure Sx | Image-wise residuals of the screened QE-NEB paths.** Each line and point shows the `error (eV/A)` reported by QE for the final complete image table. The dashed horizontal line is the `path_thr` requested in the corresponding `neb.dat` input. The residual is a force/path convergence diagnostic, not an energy uncertainty or standard error. A path passes this image-wise gate only when all movable images lie below the dashed threshold. The energy-profile figures therefore do not convert this quantity into a shaded eV interval.

The accompanying CSV preserves the final residual, threshold, last parsed outer iteration, and number of complete image tables for every path. Because electronic SCF warnings remain present in the current outputs, this audit is a quality-control record and does not promote any path to a formally converged DFT barrier.
"""
    )
    print({"paths": len(PATHS), "output": str(stem.with_suffix(".png")), "passing_paths": int(summary.passes_imagewise_gate.sum())})


if __name__ == "__main__":
    main()

