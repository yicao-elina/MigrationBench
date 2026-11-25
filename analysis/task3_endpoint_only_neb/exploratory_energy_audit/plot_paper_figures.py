"""Publication figures/tables for the exploratory NEB-path dataset.

Uses the reusable Plot Atlas primitives for ordered trajectories, joint
relationship plots, grouped comparisons, and publication styling. The data
remain explicitly exploratory: the DFT paths are not labelled as converged NEB
barriers.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "processed"
OUT = PROC / "figures"

import sys
sys.path.insert(0, "/Users/alina/Documents/Codex/plot_library/src")
from plot_atlas import (apply_publication_style, bar_with_points, get_palette,
                        joint_scatter, panel_label, save_figure, timecourse)
sys.path.insert(0, str(ROOT.parent))
from neb_plot_style import QE_COLOR, extract_qe_neb_uncertainty

PALETTE = get_palette("nature_feature_analysis")
JOINT_PALETTE = get_palette("agro_joint")
INK = PALETTE["ink"]
DFT = QE_COLOR
FOUNDATION = PALETTE["roles"]["global"]
MPA0 = PALETTE["roles"]["retrained"]
FAIL = "#999999"


def _clean(ax):
    ax.grid(False)
    for side in ax.spines.values():
        side.set_linewidth(0.9)
        side.set_color(INK)
    ax.tick_params(width=0.9, length=4, direction="out", pad=2)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight("bold")


def _short_path(s: str) -> str:
    return s.replace("2L-", "").replace("__neb-", " ").replace("__neb", " ").replace("gap-diffuse", "gap-diff.").replace("gap2QL", "gap2QL").replace("gap2Ql", "gap2QL").replace("QLs", "QLs")


def render_stability(ax, audit: pd.DataFrame) -> None:
    """Standalone stability panel: one point per path and an explicit gate."""
    a = audit.sort_values("max_abs_image_energy_delta_eV", ascending=True).copy()
    y = np.arange(len(a))
    passed = a["exploratory_energy_stable"].astype(bool).to_numpy()
    ax.axvline(0.1, color="#98489F", lw=1.15, ls=(0, (4, 2)), zorder=1)
    ax.scatter(a.loc[passed, "max_abs_image_energy_delta_eV"], y[passed], s=35,
               color=FOUNDATION, edgecolor="white", linewidth=0.55, zorder=3, label="eligible")
    ax.scatter(a.loc[~passed, "max_abs_image_energy_delta_eV"], y[~passed], s=38,
               color=FAIL, marker="X", edgecolor="white", linewidth=0.45, zorder=3, label="excluded")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yticks(y, [_short_path(x) for x in a.path_id])
    ax.set_xlabel(r"max $|E_{n}-E_{n-1}|$ across images (eV)")
    ax.set_xlim(0.0002, 30)
    ax.text(0.105, 0.98, "0.1 eV screen", transform=ax.get_xaxis_transform(), color="#98489F", fontsize=7, va="top")
    ax.legend(frameon=False, loc="lower right", fontsize=7, handletextpad=.35)
    _clean(ax)


def render_parity(fig, spec, image: pd.DataFrame) -> dict:
    """Standalone relationship panel using the Atlas joint-scatter grammar."""
    d = image[image.status.eq("success") & image.variant.eq("base")].copy()
    # Keep the two successful models visible as distinct distributions.
    groups = d.model_id.map({"legacy_foundation": "Foundation", "legacy_mpa0": "MPA0"}).to_numpy()
    pal = {"categorical": [FOUNDATION, MPA0], "sequential": JOINT_PALETTE["sequential"], "ink": INK}
    result = joint_scatter(
        fig, spec, d.dft_relative_energy_eV.to_numpy(), d.mace_relative_energy_eV.to_numpy(), pal,
        groups=groups, marginal="hist", regression=False,
        x_ref=0, y_ref=0, xlabel=r"DFT relative energy (eV)", ylabel=r"MACE relative energy (eV)",
        point_size=18, alpha=.67, marginal_axes=False, axis_linewidth=1.0, tick_width=1.0,
        legend_frame=False, point_edgecolor="white", point_linewidth=.25,
    )
    ax = result["main"]
    for extra in (result["top"], result["right"], result["corner"]):
        extra.set_visible(False)
    lim = max(float(np.nanmax(d.dft_relative_energy_eV)), float(np.nanmax(d.mace_relative_energy_eV)))
    ax.plot([0, lim], [0, lim], color="#555555", lw=1.0, ls=(0, (3, 2)), zorder=1)
    ax.text(.05, .94, "identity", transform=ax.transAxes, color="#555555", fontsize=7)
    return result


def render_perturbation(ax, response: pd.DataFrame) -> None:
    """Standalone paired grouped comparison of absolute perturbation responses."""
    d = response[response.status.eq("success") & response.variant.ne("base")].copy()
    d["model"] = d.model_id.map({"legacy_foundation": "Foundation", "legacy_mpa0": "MPA0"})
    d["perturbation"] = d.variant.map({"perturb_a_dx020": "A: +0.20 Å x", "perturb_b_tilt020": "B: tilt"})
    cats = ["A: +0.20 Å x\nFoundation", "A: +0.20 Å x\nMPA0", "B: tilt\nFoundation", "B: tilt\nMPA0"]
    vals = []
    points = []
    for perturb in ["perturb_a_dx020", "perturb_b_tilt020"]:
        for model in ["Foundation", "MPA0"]:
            x = d[(d.variant == perturb) & (d.model == model)].groupby("path_id").mace_perturbation_delta_eV.median().abs().to_numpy()
            vals.append(float(np.median(x))); points.append(x)
    errors = [float(np.percentile(x, 75) - np.percentile(x, 25)) / 2 for x in points]
    bar_with_points(ax, cats, vals, errors, [FOUNDATION, MPA0, FOUNDATION, MPA0], points=points,
                    ylabel=r"median $|\Delta E_{\mathrm{pert}}|$ (eV)", rotate=0)
    ax.axhline(0, color=INK, lw=.7)
    _clean(ax)


def render_profiles(fig, axes, image: pd.DataFrame, paths: list[str], uncertainty: pd.DataFrame) -> None:
    """Standalone repeated ordered-trajectory panels using Plot Atlas timecourse."""
    colors = [DFT, FOUNDATION, MPA0]
    labels = ["screened DFT", "Foundation", "MPA0"]
    for ax, path in zip(axes.flat, paths):
        d = image[(image.path_id == path) & image.variant.eq("base") & image.status.eq("success")]
        x = np.arange(1, 11)
        ys = []
        for model in ["dft", "legacy_foundation", "legacy_mpa0"]:
            if model == "dft":
                q = d.drop_duplicates("image_index").sort_values("image_index").dft_relative_energy_eV
            else:
                q = d[d.model_id.eq(model)].sort_values("image_index").mace_relative_energy_eV
            ys.append(q.to_numpy())
        uq = uncertainty[uncertainty.path_id.eq(path)].sort_values("image_index")
        timecourse(ax, x, ys, labels, colors, markers=True, legend=False, line_width=1.25, marker_size=2.4,
                   xlabel="Image index", ylabel="ΔE (eV)")
        ax.set_title(_short_path(path), fontsize=8, pad=3)
        _clean(ax)
    handles = [plt.Line2D([], [], color=c, marker="o", lw=1.3, ms=3, label=l) for c, l in zip(colors, labels)]
    axes.flat[-1].legend(handles=handles, frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.02))


def make_tables(audit: pd.DataFrame, summary: pd.DataFrame, response: pd.DataFrame) -> None:
    tab = audit[["path_id", "neb_outer_iterations", "self_consistency_image_calls", "max_abs_image_energy_delta_eV", "barrier_delta_eV", "exploratory_energy_stable"]].copy()
    tab.columns = ["Path", "NEB outer iterations", "Image self-consistency calls", "Max final-table ΔE (eV)", "Barrier ΔE (eV)", "Eligible"]
    tab.to_csv(OUT / "table_s1_dft_stability.csv", index=False)
    (OUT / "table_s1_dft_stability.tex").write_text(tab.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.6f}"))
    s = summary[(summary.status == "success") & summary.variant.eq("base")].copy()
    s["Model"] = s.model_id.map({"legacy_foundation": "Foundation", "legacy_mpa0": "MPA0"})
    t = s.groupby("Model").agg(
        Paths=("path_id", "nunique"),
        Median_profile_MAE_eV=("profile_mae_eV", "median"),
        Median_abs_barrier_error_eV=("barrier_difference_eV", lambda x: x.abs().median()),
        Max_abs_barrier_error_eV=("barrier_difference_eV", lambda x: x.abs().max()),
    ).reset_index()
    t.to_csv(OUT / "table_1_model_summary.csv", index=False)
    (OUT / "table_1_model_summary.tex").write_text(t.to_latex(index=False, index_names=False, escape=False, float_format=lambda x: f"{x:.4f}"))
    r = response[(response.status == "success") & response.variant.ne("base")].copy()
    r["Model"] = r.model_id.map({"legacy_foundation": "Foundation", "legacy_mpa0": "MPA0"})
    r["Perturbation"] = r.variant.map({"perturb_a_dx020": "A: +0.20 Å x", "perturb_b_tilt020": "B: tilt"})
    rr = r.groupby(["Model", "Perturbation", "path_id"]).mace_perturbation_delta_eV.median().abs().reset_index()
    rt = rr.groupby(["Model", "Perturbation"]).agg(Median_abs_delta_eV=("mace_perturbation_delta_eV", "median"), Max_abs_delta_eV=("mace_perturbation_delta_eV", "max")).reset_index()
    rt.to_csv(OUT / "table_s2_perturbation_response.csv", index=False)
    (OUT / "table_s2_perturbation_response.tex").write_text(rt.to_latex(index=False, index_names=False, escape=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    apply_publication_style(8.4)
    OUT.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(ROOT / "scf_neb_iteration_energy_audit.csv")
    summary = pd.read_csv(PROC / "mace_dft_relative_path_summary.csv")
    image = pd.read_csv(PROC / "mace_dft_relative_image_metrics.csv")
    response = pd.read_csv(PROC / "mace_perturbation_response.csv")

    fig = plt.figure(figsize=(7.2, 5.4), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.03], width_ratios=[1.03, 1.0], wspace=.50, hspace=.58)
    ax = fig.add_subplot(gs[0, 0]); render_stability(ax, audit); panel_label(ax, "a")
    parity = render_parity(fig, gs[0, 1], image); panel_label(parity["main"], "b")
    ax = fig.add_subplot(gs[1, :]); render_perturbation(ax, response); panel_label(ax, "c")
    fig.subplots_adjust(left=.20, right=.96, bottom=.13, top=.96)
    save_figure(fig, OUT / "figure_1_stability_parity_perturbation.pdf")
    fig.savefig(OUT / "figure_1_stability_parity_perturbation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    paths = [p for p in summary[(summary.status == "success") & summary.variant.eq("base")].path_id.unique()]
    paths = sorted(paths)
    uncertainty = extract_qe_neb_uncertainty(
        paths, remote_root=ROOT.parent / "remote_snapshot", n_images=10, n_tail_tables=5
    )
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.7), sharey=False)
    render_profiles(fig, axes, image, paths, uncertainty)
    fig.subplots_adjust(left=.09, right=.86, bottom=.12, top=.93, wspace=.42, hspace=.55)
    for i, ax in enumerate(axes.flat): panel_label(ax, chr(ord("a") + i), x=-.16, y=1.12)
    save_figure(fig, OUT / "figure_2_relative_energy_profiles.pdf")
    fig.savefig(OUT / "figure_2_relative_energy_profiles.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    make_tables(audit, summary, response)

    (OUT / "captions_and_discussion.md").write_text("""# Publication text\n\n## Figure 1 caption\n**Figure 1 | Stability-gated exploratory path sampling and model response.** (a) Maximum absolute energy change between the penultimate and final complete NEB energy tables, evaluated over all images. The dashed line marks the exploratory 0.1 eV screening threshold; two paths are excluded. The logarithmic/symlog scale preserves both sub-meV stability and the large instability of the rejected paths. (b) Image-level relative-energy comparison for the eight eligible paths, with image 1 of each path and variant used as the zero. Points are shown for the two models that completed all evaluations; the dashed diagonal is identity. This comparison tests agreement with the screened path-energy trace, not DFT accuracy, because the DFT electronic residual history and force convergence are unavailable. (c) Median per-path absolute energy response to the two prescribed perturbations. Points are individual paths and bars show the median with half-IQR.\n\n## Figure 2 caption\n**Figure 2 | Relative energy profiles along the eight eligible non-equilibrium paths.** Each panel shows the ten-image path trace from the screened QE path table, Foundation, and MPA0. The QE `.path` energies were converted from Hartree to eV before image-1 referencing; all model energies are independently normalized within each path. The curves are intended to expose profile shape, curvature, and model sensitivity; they are not converged NEB barriers or equilibrium reaction coordinates.\n\n## Results and discussion\nThe last-table stability screen retained eight of ten candidate paths. The retained paths span a maximum penultimate-to-final image energy change of 0.000491–0.050366 eV, whereas the two rejected paths exhibit 0.154644 and 19.062111 eV. This separation motivates using the retained paths as exploratory fixed-geometry energy samples while keeping the rejected paths outside the model-comparison reference set.\n\nAfter the Hartree-to-eV correction, the retained base paths give median relative-profile MAEs of 0.0606 eV for Foundation and 0.1154 eV for MPA0, with median absolute barrier-like discrepancies of 0.0541 and 0.1418 eV, respectively. Agreement remains strongly path dependent: the Cu neb-QLs profile is an outlier for both models, while the gap2QL traces are comparatively consistent. Foundation barrier-like errors on the three gap2QL paths are 0.0018–0.0549 eV in absolute value; the corresponding MPA0 range is 0.0741–0.1768 eV. The pattern is therefore more informative as a stress test of model response and path-local extrapolation than as a ranking of DFT fidelity.\n\nThe imposed perturbations reveal an additional sensitivity axis. Median absolute responses remain small on most paths but become large on the same high-discrepancy trajectories, indicating that local perturbation response can identify unstable or out-of-distribution regions. Because the underlying images are not demonstrated equilibrium structures and the perturbation amplitudes are finite, these results should not be interpreted as harmonic force constants or equilibrium phonon response.\n\nFinally, FT-600K was not silently treated as a scientific failure: all 240 evaluations failed at model input validation because the deployed model lacks Cu, Li, and Na species support. This is a package/model-coverage blocker and should be reported separately from energetic disagreement.\n""")
    print(f"Wrote figures, tables, captions to {OUT}")


if __name__ == "__main__":
    main()
