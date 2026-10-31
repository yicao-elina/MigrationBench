#!/usr/bin/env python3
"""Build the Atomic Migration blueprint figures from audited local sources.

This is an additive, fail-closed builder.  It consumes the existing MigrationBench
processed tables and manifests by absolute path, writes only to this PR worktree,
and labels provisional/proxy/SCF-warning evidence in the figure package.
No synthetic observations are generated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.lines import Line2D

WORKTREE = Path(__file__).resolve().parents[4]
WILEY = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench")
ATLAS = Path("/Users/alina/Documents/Codex/plot_library_recovered")
sys.path.insert(0, str(ATLAS / "src"))
from plot_atlas import aggregate_profile_rows, get_palette, plot_profile_ensemble  # noqa: E402


OUT_DEFAULT = WORKTREE / "docs/reproducibility/canva_figure_provenance"
FONT = "Arial"
COLORS = {
    "Foundation": "#E68F1D",
    "FT-600K": "#C4547A",
    "FT-MultiT": "#A83C63",
    "Scratch-600K": "#4B75A8",
    "Scratch-MultiT": "#6B8FBA",
    "QE reference": "#222222",
    "MACE-MP": "#4B75A8",
    "MACE-OMAT-0": "#C4547A",
    "MACE-MH": "#E68F1D",
}


def read_rows(path: Path, delimiter: str | None = None) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if delimiter is None:
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open(newline="", errors="replace") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def f(row: dict[str, str], key: str, default=np.nan) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_house_style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT,
            "font.sans-serif": [FONT, "Helvetica"],
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12.8,
            "axes.labelweight": "bold",
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "axes.linewidth": 1.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def style_ax(ax, *, bold_ticks: bool = True) -> None:
    ax.grid(False)
    ax.tick_params(axis="both", direction="out", length=4.5, width=1.0, pad=3)
    ax.tick_params(axis="both", which="minor", direction="out", length=2.5, width=0.8)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.05)
        spine.set_color("#111111")
    if bold_ticks:
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontweight("bold")


def save_figure(fig: plt.Figure, out: Path, stem: str, dpi: int = 400) -> dict[str, str]:
    out.mkdir(parents=True, exist_ok=True)
    path = out / stem
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {ext: str(path.with_suffix(f".{ext}")) for ext in ("png", "pdf", "svg")}


def status_box(ax, title: str, body: str, color: str = "#B23A48") -> None:
    ax.set_axis_off()
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.04, 0.08), 0.92, 0.84, boxstyle="round,pad=0.02", fc="#FFF7F7", ec=color, lw=1.6
        )
    )
    ax.text(0.5, 0.72, title, ha="center", va="center", color=color, fontsize=13, fontweight="bold")
    ax.text(0.5, 0.47, body, ha="center", va="center", color="#333333", fontsize=10.5, linespacing=1.35)


def family_from_stem(stem: str) -> str:
    if stem.startswith("ft_600k"):
        return "FT-600K"
    if stem.startswith("ft_multit"):
        return "FT-MultiT"
    if stem.startswith("scratch_600k"):
        return "Scratch-600K"
    return "Scratch-MultiT"


def downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Reduce display vertices while retaining the complete source tables in the manifest."""
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
    return x[idx], y[idx]


def fig1(out: Path) -> tuple[dict[str, str], list[Path]]:
    src = WILEY / "analysis/migrationbench_universe_publication_20261026/derived/comprehensive_task_map_points.csv"
    rows = read_rows(src)
    fig = plt.figure(figsize=(10.8, 5.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.9], wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    groups = ["L0 DFT Reference", "L1 Thermal Equilibrium", "L2 NEB Migration", "L3 Layer Sliding"]
    group_colors = {groups[0]: "#8DA9BF", groups[1]: "#6E9E75", groups[2]: "#C4547A", groups[3]: "#E68F1D"}
    markers = {"train": ".", "valid": "^", "test": "s", "md_sample": "o", "dft_neb": "D", "dft_interlayer": "P"}
    for group in groups:
        for split, marker in markers.items():
            sub = [r for r in rows if r.get("task_label") == group and r.get("split") == split]
            x = np.array([f(r, "x_pca") for r in sub]); y = np.array([f(r, "y_pca") for r in sub])
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.any():
                ax.scatter(x[mask], y[mask], s=8 if marker == "." else 18, marker=marker,
                           color=group_colors[group], alpha=0.35 if marker == "." else 0.8,
                           linewidths=0.2, edgecolors="white", label=group if split == "train" else None)
    ax.set_xlabel("SOAP PCA 1")
    ax.set_ylabel("SOAP PCA 2")
    ax.set_title("A  Configuration universe", loc="left")
    style_ax(ax)
    handles = [Line2D([], [], marker="o", linestyle="", color=group_colors[g], label=g, markersize=6) for g in groups]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="best")

    ax = fig.add_subplot(gs[0, 1])
    ax.set_axis_off()
    ax.set_title("B  Matched benchmark workflow", loc="left")
    boxes = [
        (0.08, 0.72, "L0\nreference"),
        (0.58, 0.72, "L1\nthermal"),
        (0.08, 0.42, "L2\nfixed images"),
        (0.58, 0.42, "L3\nself-relaxed"),
        (0.33, 0.10, "MD / sliding\nvalidation"),
    ]
    for x, y, label in boxes:
        ax.add_patch(patches.FancyBboxPatch((x, y), 0.32, 0.15, boxstyle="round,pad=0.02", fc="#EDF2F6", ec="#8DA9BF", lw=1.0))
        ax.text(x + 0.16, y + 0.075, label, ha="center", va="center", fontsize=10, fontweight="bold")
    for x1, y1, x2, y2 in [(0.40, .72, .58, .80), (.24, .72, .24, .57), (.74, .72, .74, .57), (.40, .42, .49, .25), (.74, .42, .65, .25)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#555555"})

    ax = fig.add_subplot(gs[0, 2])
    counts = defaultdict(int)
    for r in rows:
        counts[r.get("task_label", "unlabelled")] += 1
    labels = ["L0 reference", "L1 thermal", "L2 NEB", "L3 sliding"]
    keys = groups
    vals = [counts[k] for k in keys]
    ax.barh(labels[::-1], vals[::-1], color=[group_colors[k] for k in keys[::-1]], edgecolor="#111111", lw=0.5)
    for i, v in enumerate(vals[::-1]): ax.text(v + max(vals) * .02, i, f"n={v}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("structure rows")
    ax.set_title("C  Task coverage", loc="left")
    style_ax(ax)
    fig.suptitle("Figure 1 | Task-conditioned benchmark and configuration universe", x=0.02, ha="left", y=1.02, fontsize=17, fontweight="bold")
    fig.text(0.02, -0.035, "PCA is a projection of the audited structure universe; it is not, by itself, an OOD-risk metric.", fontsize=9, color="#555555")
    assets = save_figure(fig, out, "figure01_task_conditioned_universe")
    return assets, [src]


def fig2(out: Path) -> tuple[dict[str, str], list[Path]]:
    root = WILEY / "analysis/rmse_train_test_20261026"
    summary_path = root / "parity_summary.csv"
    summary = read_rows(summary_path)
    per_files = sorted((root / "raw/rockfish_parity").glob("*_test__per_frame.csv"))
    fig = plt.figure(figsize=(11.5, 7.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, .9], hspace=.42, wspace=.34)
    axa, axb, axc = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axd, axe, axf = [fig.add_subplot(gs[1, i]) for i in range(3)]
    force_x, force_y, force_family = [], [], []
    all_frame = []
    for path in per_files:
        fam = family_from_stem(path.stem)
        rows = read_rows(path)
        for r in rows:
            x = f(r, "energy_ref_eV_per_atom"); y = (f(r, "energy_pred_eV_per_atom") - x) * 1000
            if np.isfinite([x, y]).all():
                axa.scatter(x, y, s=7, alpha=.32, color=COLORS[fam], edgecolors="none")
            all_frame.append((fam, r.get("trajectory", "unknown"), f(r, "force_rmse_meV_A")))
        npz = path.with_name(path.name.replace("__per_frame.csv", "__parity.npz"))
        if npz.exists():
            dat = np.load(npz)
            ref = np.asarray(dat["force_ref_eV_A"], dtype=float)
            frame_idx = np.asarray(dat["force_frame_index"], dtype=int)
            rmse = np.asarray(dat["frame_force_rmse_meV_A"], dtype=float)
            if len(ref) % 3 == 0:
                norms = np.linalg.norm(ref.reshape(-1, 3), axis=1)
                n_atoms = len(norms) // len(rmse)
                if n_atoms > 0 and n_atoms * len(rmse) == len(norms):
                    means = norms.reshape(len(rmse), n_atoms).mean(axis=1)
                    axb.scatter(means, rmse, s=7, alpha=.35, color=COLORS[fam], edgecolors="none")
    for ax in [axa, axb]:
        ax.axhline(0, color="#333333", lw=1.0, ls=(0, (4, 2))) if ax is axa else None
        style_ax(ax)
    axa.set_xlabel("QE energy field (source unit audit pending)")
    axa.set_ylabel("model − QE (meV; source column)")
    axa.set_title("A  Held-out energy residual", loc="left")
    axb.set_xlabel("mean QE |F| (eV Å$^{-1}$)")
    axb.set_ylabel("frame force RMSE (meV Å$^{-1}$)")
    axb.set_title("B  Held-out force errors", loc="left")
    handles = [Line2D([], [], marker="o", linestyle="", color=COLORS[k], label=k, markersize=5) for k in COLORS if k in {"Foundation", "FT-600K", "FT-MultiT", "Scratch-600K", "Scratch-MultiT"}]
    axb.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")

    families = ["Scratch-600K", "FT-600K", "Scratch-MultiT", "FT-MultiT"]
    for metric, ax, key in [("Energy RMSE", axc, "energy_rmse_meV_per_atom"), ("Force RMSE", axf, "force_rmse_meV_A")]:
        for i, fam in enumerate(families):
            vals = [f(r, key) for r in summary if r.get("model", "") == fam and r.get("split") == "test"]
            seeds = [f(r, key) for r in summary if r.get("model", "") == fam and r.get("split") == "test"]
            if not vals: continue
            mean, sd = np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0
            ax.errorbar(i, mean, yerr=sd, fmt="o", ms=5, color=COLORS[fam], capsize=3, lw=1.0)
            ax.scatter([i] * len(seeds), seeds, s=18, facecolors="white", edgecolors=COLORS[fam], linewidths=1.1, zorder=5)
        ax.set_xticks(range(len(families)), ["Scratch\n600K", "FT\n600K", "Scratch\nMultiT", "FT\nMultiT"])
        ax.set_ylabel(key.replace("_", " "))
        ax.set_title(("C  " if ax is axc else "F  ") + metric + " by seed", loc="left")
        style_ax(ax)
    axd.set_title("D  Distribution by held-out trajectory", loc="left")
    trajectories = sorted({x[1] for x in all_frame})
    offsets = np.linspace(-.18, .18, len(families))
    for j, traj in enumerate(trajectories):
        for off, fam in zip(offsets, families):
            vals = np.array([x[2] for x in all_frame if x[0] == fam and x[1] == traj and np.isfinite(x[2])])
            if len(vals) == 0: continue
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            x = j + off
            axd.vlines(x, q1, q3, color=COLORS[fam], lw=2.0)
            axd.scatter(x, med, color=COLORS[fam], s=24, zorder=5)
    axd.set_xticks(range(len(trajectories)), [t.replace("2L_", "").replace("_aimd_1", "") for t in trajectories], rotation=20, ha="right")
    axd.set_ylabel("frame force RMSE (meV Å$^{-1}$)")
    style_ax(axd)
    status_box(axe, "PARTIAL", "Only the audited held-out 600 K\ntrajectory groups are available.\nCross-temperature/task strata are not\nasserted by this package.", color="#9A6A00")
    axe.set_title("E  Distribution-shift gate", loc="left")
    axf.set_ylim(bottom=0)
    fig.suptitle("Figure 2 | Near-equilibrium parity is an entry gate", x=.02, ha="left", y=.995, fontsize=17, fontweight="bold")
    assets = save_figure(fig, out, "figure02_near_equilibrium_parity")
    return assets, [summary_path, *per_files]


def load_cr_profiles() -> list[dict[str, str]]:
    return read_rows(WILEY / "analysis/neb_cross_element_triptychs_20261024/data/Cr_fixed_path_representative_profiles.tsv")


def profile_lookup(rows: list[dict[str, str]], path_key: str) -> dict[str, list[dict[str, str]]]:
    out = defaultdict(list)
    for r in rows:
        if r.get("path_key") == path_key:
            out[r.get("series_label", "unknown")].append(r)
    return dict(out)


def fig3(out: Path) -> tuple[dict[str, str], list[Path]]:
    src = WILEY / "analysis/neb_cross_element_triptychs_20261024/data/Cr_fixed_path_representative_profiles.tsv"
    rows = load_cr_profiles(); grouped = profile_lookup(rows, "within_gap")
    order = ["QE reference", "From-scratch (seed 123)", "FT-600K (seed 123)", "FT-MultiT (seed 234)", "Foundation (OMAT-0-M)"]
    line_color = {"QE reference": COLORS["QE reference"], "From-scratch (seed 123)": COLORS["Scratch-600K"], "FT-600K (seed 123)": COLORS["FT-600K"], "FT-MultiT (seed 234)": COLORS["FT-MultiT"], "Foundation (OMAT-0-M)": COLORS["Foundation"]}
    fig = plt.figure(figsize=(10.6, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, .85], hspace=.42, wspace=.32)
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    values = {}
    for label in order:
        sub = grouped.get(label, [])
        if not sub: continue
        sub = sorted(sub, key=lambda r: f(r, "image_index"))
        x = np.array([f(r, "s") for r in sub]); y = np.array([f(r, "relative_energy_eV") for r in sub])
        values[label] = y
        axa.plot(x, y, marker="o" if label != "QE reference" else "D", ms=4.4, lw=2.0 if label != "QE reference" else 1.45,
                 ls="--" if label == "QE reference" else "-", color=line_color[label], markerfacecolor="white", label=label)
    axa.set_xlabel("normalized image coordinate, $s$"); axa.set_ylabel("relative energy (eV)")
    axa.set_title("A  Same fixed QE images", loc="left")
    axa.text(.02, .98, "G1: UNCONVERGED — fixed-image QE reference, not a validated migration barrier", transform=axa.transAxes,
             va="top", fontsize=8.2, color="#B23A48", fontweight="bold")
    axa.legend(frameon=False, fontsize=8.2, loc="upper right"); style_ax(axa)
    qe = values.get("QE reference")
    metrics = []
    if qe is not None:
        for label in order[1:]:
            if label in values:
                err = np.abs(values[label] - qe)
                metrics.append((label, float(np.mean(err)), float(np.max(err))))
    for i, (label, mae, mx) in enumerate(metrics):
        axb.scatter(i, mae, s=45, color=line_color[label], edgecolors="white", linewidths=.6, zorder=5)
        axb.vlines(i, mae, mx, color=line_color[label], lw=1.2)
        axb.scatter(i, mx, marker="^", s=22, color=line_color[label], zorder=5)
    axb.set_xticks(range(len(metrics)), [x[0].replace(" (seed ", "\nseed ").replace(")", "") for x in metrics], rotation=25, ha="right")
    axb.set_ylabel("|model − QE| (eV)"); axb.set_title("B  Diagnostic error on fixed images", loc="left"); style_ax(axb)
    for label in order[1:]:
        if label in values and qe is not None:
            axc.plot(np.arange(10), values[label] - qe, marker="o", ms=3.6, lw=1.5, color=line_color[label], label=label)
    axc.axhline(0, color="#222222", ls=(0, (4, 2)), lw=1.0)
    axc.set_xlabel("image index"); axc.set_ylabel("model − QE residual (eV)"); axc.set_title("C  Residual diagnostic and maximum-image check", loc="left"); style_ax(axc)
    axc.legend(frameon=False, ncol=4, fontsize=8.2, loc="upper center")
    fig.suptitle("Figure 3 | Within-gap: paired fixed-image evidence", x=.02, ha="left", y=.995, fontsize=17, fontweight="bold")
    assets = save_figure(fig, out, "figure03_within_gap_fixed_images")
    return assets, [src]


def fig4(out: Path) -> tuple[dict[str, str], list[Path]]:
    src = WILEY / "analysis/neb_cross_element_triptychs_20261024/data/Cr_fixed_path_representative_profiles.tsv"
    rows = load_cr_profiles()
    paths = [("within_gap", "Within-gap"), ("gap_to_ql", "Gap-to-QL / into-QL"), ("within_ql", "Within-QL")]
    model_order = ["From-scratch (seed 123)", "FT-600K (seed 123)", "FT-MultiT (seed 234)", "Foundation (OMAT-0-M)"]
    line_color = {"QE reference": COLORS["QE reference"], "From-scratch (seed 123)": COLORS["Scratch-600K"], "FT-600K (seed 123)": COLORS["FT-600K"], "FT-MultiT (seed 234)": COLORS["FT-MultiT"], "Foundation (OMAT-0-M)": COLORS["Foundation"]}
    fig = plt.figure(figsize=(11.2, 6.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, .72], hspace=.42, wspace=.28)
    diffs=[]
    for col, (key, title) in enumerate(paths):
        ax = fig.add_subplot(gs[0, col]); grouped = profile_lookup(rows, key); vals={}
        for label in ["QE reference", *model_order]:
            sub=sorted(grouped.get(label, []), key=lambda r:f(r,"image_index"))
            if not sub: continue
            x=np.array([f(r,"s") for r in sub]); y=np.array([f(r,"relative_energy_eV") for r in sub]); vals[label]=y
            ax.plot(x,y,marker="D" if label=="QE reference" else "o",ms=3.8,lw=1.45 if label=="QE reference" else 1.8,ls="--" if label=="QE reference" else "-",color=line_color[label],markerfacecolor="white",label=label)
        ax.set_title(chr(97+col)+"  "+title, loc="left"); ax.set_xlabel("path coordinate, $s$"); ax.set_ylabel("relative E (eV)"); style_ax(ax)
        if col==0: ax.legend(frameon=False, fontsize=7.7, loc="upper left")
        if "QE reference" in vals and "From-scratch (seed 123)" in vals and "FT-600K (seed 123)" in vals:
            qe=vals["QE reference"]; scratch=np.mean(np.abs(vals["From-scratch (seed 123)"]-qe)); ft=np.mean(np.abs(vals["FT-600K (seed 123)"]-qe)); diffs.append((title, scratch-ft))
    axd=fig.add_subplot(gs[1,:])
    if diffs:
        names=[d[0] for d in diffs]; vals=[d[1] for d in diffs]
        axd.axhline(0,color="#222222",lw=1.0,ls=(0,(4,2)))
        axd.bar(range(len(vals)), vals, color=["#6E9E75" if v>0 else "#B23A48" for v in vals], alpha=.85, edgecolor="#111111", lw=.5)
        axd.scatter(range(len(vals)), vals, color="#222222", s=18, zorder=5)
        axd.set_xticks(range(len(vals)), names); axd.set_ylabel("scratch MAE − FT MAE (eV)")
    axd.set_title("D  Paired diagnostic gain (positive = FT has smaller fixed-image error)", loc="left"); style_ax(axd)
    fig.suptitle("Figure 4 | Where target adaptation stops transferring", x=.02, ha="left", y=.995, fontsize=17, fontweight="bold")
    fig.text(.02, -.035, "QE references are retained with COMPLETED_BUT_SCF_WARNING status; values are diagnostics, not validated barriers.", fontsize=9, color="#555555")
    assets=save_figure(fig,out,"figure04_cross_environment_transfer"); return assets,[src]


def fig5(out: Path) -> tuple[dict[str, str], list[Path]]:
    src=WILEY/"analysis/neb_cross_element_triptychs_20261024/data/Cr_four_model_self_relaxed_neb_profiles.tsv"
    rows=read_rows(src); paths=[("within_gap","Within-gap"),("gap_to_ql","Gap-to-QL"),("within_ql","Within-QL")]
    fig=plt.figure(figsize=(11.2,6.3)); gs=fig.add_gridspec(2,3,height_ratios=[1.0,.78],hspace=.46,wspace=.30)
    series_order=[]
    for r in rows:
        if r.get("kind")=="model" and r.get("series_label") not in series_order: series_order.append(r.get("series_label"))
    series_colors={"From-scratch (seed 123)":COLORS["Scratch-600K"],"FT-600K (seed 123)":COLORS["FT-600K"],"FT-MultiT (seed 234)":COLORS["FT-MultiT"],"Foundation (OMAT-0-M)":COLORS["Foundation"]}
    status_map={"CONVERGED_MODEL_OBSERVABLE":1,"FAILURE_MODE_NONCONVERGED":0}
    status_grid=[]; grid_labels=[]
    for col,(key,title) in enumerate(paths):
        ax=fig.add_subplot(gs[0,col]); sub=[r for r in rows if r.get("kind")=="model" and r.get("path_key")==key]
        for label in series_order:
            q=sorted([r for r in sub if r.get("series_label")==label],key=lambda r:f(r,"image_index"))
            if not q: continue
            x=np.array([f(r,"s") for r in q]); y=np.array([f(r,"relative_energy_eV") for r in q]); st=q[0].get("status","")
            ax.plot(x,y,marker="o",ms=3.8,lw=1.8,color=series_colors.get(label,"#555555"),alpha=.9 if st.startswith("CONVERGED") else .5,label=label)
            if label not in grid_labels: grid_labels.append(label)
            status_grid.append((key,label,status_map.get(st,np.nan)))
        ax.set_title(chr(97+col)+"  "+title,loc="left"); ax.set_xlabel("model-path coordinate, $s$"); ax.set_ylabel("model relative E (eV)"); style_ax(ax)
        if col==0: ax.legend(frameon=False,fontsize=8,loc="upper left")
    axd=fig.add_subplot(gs[1,0:2]); axd.set_title("D  Recorded profile status (diagnostic only)",loc="left")
    mat=np.full((len(grid_labels),len(paths)),np.nan)
    for j,(key,_) in enumerate(paths):
        for i,label in enumerate(grid_labels):
            hits=[v for k,l,v in status_grid if k==key and l==label]
            if hits: mat[i,j]=hits[0]
    im=axd.imshow(mat,cmap=matplotlib.colors.ListedColormap(["#D65C5C","#6E9E75"]),vmin=0,vmax=1,aspect="auto")
    axd.set_xticks(range(len(paths)),[p[1] for p in paths],rotation=20,ha="right"); axd.set_yticks(range(len(grid_labels)),[x.replace(" (seed ","\nseed ").replace(")","") for x in grid_labels])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i,j]): axd.text(j,i,"PASS" if mat[i,j] else "FAIL",ha="center",va="center",fontsize=8,fontweight="bold",color="white")
    style_ax(axd)
    status_box(fig.add_subplot(gs[1,2]),"BLOCKED FOR PUBLICATION","Campaign is endpoint-only /\nprecanonical and stopped for\nendpoint repair. Per-run fmax,\niterations, endpoint identity and\ngeometry alignment are incomplete;\ndo not report success rate or barrier.")
    fig.suptitle("Figure 5 | Model-generated NEB paths are a separate diagnostic",x=.02,ha="left",y=.995,fontsize=17,fontweight="bold")
    fig.text(.02,-.035,"Current profiles are retained as model observables only; they are not validated DFT MEP barriers.",fontsize=9,color="#555555")
    assets=save_figure(fig,out,"figure05_generated_neb_diagnostic"); return assets,[src]


def fig6(out: Path) -> tuple[dict[str, str], list[Path]]:
    root=WILEY/"analysis/md_1ns_updated_publication_figure_20261025"; status_path=root/"MD15_FIGURE_DATA_R06G.tsv"; summary_path=root/"MD15_CN_GROUP_SUMMARY_R06G.tsv"; analysis=root/"md15_analysis_r03"
    status=read_rows(status_path); summary=read_rows(summary_path); by_id={r["model_id"]:r for r in status}
    fig=plt.figure(figsize=(11.6,7.4)); gs=fig.add_gridspec(2,3,hspace=.44,wspace=.33)
    axa,axb,axc=[fig.add_subplot(gs[0,i]) for i in range(3)]; axd,axe,axf=[fig.add_subplot(gs[1,i]) for i in range(3)]
    family_colors={"Foundation":"#5C8FB8","FT-600K":"#C4547A","FT-MultiT":"#A83C63","Scratch":"#E68F1D"}
    file_paths=[]
    for p in sorted((analysis/"series").glob("*.csv")):
        model=p.stem; file_paths.append(p); r=by_id.get(model); fam=r.get("family","") if r else ""; color=family_colors.get(fam,"#777777"); rows=read_rows(p)
        t=np.array([f(x,"time_ps") for x in rows]); temp=np.array([f(x,"temperature_K") for x in rows]); mask=np.isfinite(t)&np.isfinite(temp)
        if mask.any():
            tx, ty = downsample_xy(t[mask], np.clip(temp[mask], 520, 700)); axa.plot(tx,ty,color=color,alpha=.48,lw=.65)
            for side in (temp[mask] < 520, temp[mask] > 700):
                if side.any(): axa.scatter(t[mask][side],np.where(temp[mask][side]>700,700,520),s=3,color="#B23A48",alpha=.7)
        for d,ax,xkey,ykey in [("msd",axf,"lag_ps","msd3d_A2"),("rdf",axd,"r_A","g_CrCr"),("vacf",axe,"lag_ps","vacf_normalized")]:
            q=read_rows(analysis/d/(p.name))
            x=np.array([f(z,xkey) for z in q]); y=np.array([f(z,ykey) for z in q]); m=np.isfinite(x)&np.isfinite(y)
            if m.any():
                dx, dy = downsample_xy(x[m], y[m])
                ax.plot(dx,dy,color=color,alpha=.40,lw=.75)
    axa.set_xlabel("time (ps)"); axa.set_ylabel("instantaneous T (K)"); axa.set_ylim(520,700); axa.set_title("A  Temperature traces",loc="left",fontsize=11.5); style_ax(axa)
    ids=[r["model_id"] for r in status]; horizon=np.array([f(by_id[i],"last_time_ps") for i in ids]); y=np.arange(len(ids));
    for yi,i in zip(y,ids):
        r=by_id[i]; fam=r.get("family",""); color=family_colors.get(fam,"#777777"); axb.barh(yi,f(r,"last_time_ps"),color=color,alpha=.85,height=.72)
        if r.get("status") == "SCIENTIFIC_FAILURE": axb.scatter(f(r,"last_time_ps"),yi,color="#B23A48",marker="x",s=35,zorder=5)
    axb.axvline(1000,color="#222222",ls=(0,(4,2)),lw=1.0); axb.set_yticks(y, [by_id[i].get("display_label",i) for i in ids],fontsize=7.5); axb.set_xlabel("observed duration (ps)"); axb.set_title("B  Observed horizon",loc="left",fontsize=11.5); style_ax(axb)
    pairs=sorted({r["pair"] for r in summary}); groups=sorted({r["group"] for r in summary}); xbase=np.arange(len(pairs)); width=.18
    for j,g in enumerate(groups):
        q=[next((r for r in summary if r["group"]==g and r["pair"]==p),None) for p in pairs]; vals=np.array([f(r,"mean_neighbors_per_center") if r else np.nan for r in q]); errs=np.array([f(r,"std_neighbors_per_center") if r else 0 for r in q]); axc.errorbar(xbase+(j-1.5)*width,vals,yerr=errs,fmt="o",color=family_colors.get(g,"#777777"),label=g,capsize=2,lw=1.0)
    axc.set_xticks(xbase,pairs,rotation=25,ha="right"); axc.set_ylabel("neighbors per center"); axc.set_title("C  Coordination",loc="left",fontsize=11.5); axc.legend(frameon=False,fontsize=7.5); style_ax(axc)
    axd.set_xlabel("r (Å)"); axd.set_ylabel("g$_{CrCr}$(r)"); axd.set_title("D  RDF",loc="left"); style_ax(axd)
    axe.set_xlabel("lag (ps)"); axe.set_ylabel("normalized VACF"); axe.set_title("E  Velocity memory",loc="left"); style_ax(axe)
    axf.set_xlabel("lag (ps)"); axf.set_ylabel("MSD (Å$^2$)"); axf.set_title("F  Observed Cr motion",loc="left"); style_ax(axf)
    fig.suptitle("Figure 6 | Matched MD stability, structure and motion",x=.02,ha="left",y=.995,fontsize=17,fontweight="bold")
    fig.text(.02,-.035,"Curves stop at the recorded horizon; no failed trajectory is extended to 1 ns and no diffusion ranking is inferred.",fontsize=9,color="#555555")
    assets=save_figure(fig,out,"figure06_matched_md_diagnostics"); return assets,[status_path,summary_path,*file_paths]


def fig7(out: Path) -> tuple[dict[str, str], list[Path]]:
    root=WILEY/"analysis/foundation_interlayer_sliding_10model_20261024/results/tables"; img_path=root/"IMAGE_PROFILES.tsv"; qe_path=root/"QE_REFERENCE.tsv"; srcs=[img_path,qe_path]
    img=read_rows(img_path); qe=read_rows(qe_path)
    model_class={"foundation_agnesi_large":"MACE-MP","foundation_agnesi_medium":"MACE-MP","foundation_agnesi_medium_density":"MACE-MP","foundation_agnesi_small_density":"MACE-MP","foundation_mp0b3":"MACE-MP","foundation_omat0_medium":"MACE-OMAT-0","foundation_omat0_small":"MACE-OMAT-0","foundation_matpes_pbe_omat_ft":"MACE-OMAT-0","foundation_mh0":"MACE-MH","foundation_mh1":"MACE-MH"}
    for r in img: r["model_class"]=model_class.get(r.get("model_id"),"unknown")
    obs=sorted({r.get("observable") for r in img}); fixed=next((x for x in obs if "FIXED" in x),obs[0]); selfobs=next((x for x in obs if "SELF" in x),obs[-1])
    pal={"categorical":[COLORS["MACE-MP"],COLORS["MACE-OMAT-0"],COLORS["MACE-MH"]],"roles":{k:COLORS[k] for k in ["MACE-MP","MACE-OMAT-0","MACE-MH"]},"reference":"#222222"}
    fig,axes=plt.subplots(2,2,figsize=(10.4,8.2)); systems=[("bulk","Bulk"),("cr_in_gap","Cr-in-gap")]
    labels=[]
    for row,(system,system_label) in enumerate(systems):
        for col,observable in enumerate([fixed,selfobs]):
            ax=axes[row,col]; q=[r for r in img if r.get("system")==system and r.get("observable")==observable]
            profiles=aggregate_profile_rows([{"s":f(r,"image_fraction"),"series":r["model_class"],"replicate":r["model_id"],"value":f(r,"relative_energy_eV")} for r in q],series_order=["MACE-MP","MACE-OMAT-0","MACE-MH"])
            ref=[r for r in qe if r.get("system")==system]; ref=sorted(ref,key=lambda r:f(r,"image_index"))
            plot_profile_ensemble(ax,profiles,x=np.array([f(r,"image_fraction") for r in sorted(q,key=lambda r:f(r,"image_index")) if r.get("model_class")=="MACE-MP"][::5]) if False else None,reference={"y":np.array([f(r,"relative_energy_eV") for r in ref]),"label":"QE reference · SCF warning"},palette=pal,style={"font_family":FONT,"tick_labels_bold":True},title=f"{chr(97+row*2+col)}  {system_label} · {'fixed images' if col==0 else 'generated path'}",xlabel="registry displacement fraction" if row==1 else None,ylabel="relative E (eV)",panel_label=None)
            if col==1: ax.text(.98,.04,"endpoint-only proxy",transform=ax.transAxes,ha="right",fontsize=8,color="#9A6A00")
    handles=[Line2D([],[],color=COLORS[k],marker="o",markerfacecolor="white",lw=2,ms=4,label=k) for k in ["MACE-MP","MACE-OMAT-0","MACE-MH"]]
    handles.append(Line2D([],[],color="#222222",marker="D",markerfacecolor="white",lw=1.45,ls="--",ms=3.6,label="QE reference · SCF warning"))
    fig.legend(handles=handles,loc="upper center",ncol=4,frameon=False,bbox_to_anchor=(.52,.985),fontsize=10)
    fig.subplots_adjust(left=.09,right=.98,bottom=.08,top=.90,wspace=.30,hspace=.40)
    fig.suptitle("Figure 7 | Interlayer sliding: fixed images versus generated paths",x=.02,ha="left",y=1.02,fontsize=17,fontweight="bold")
    fig.text(.02,-.035,"Bands are pointwise sample SD across model identities within class; they are not training-seed error bars.",fontsize=9,color="#555555")
    assets=save_figure(fig,out,"figure07_interlayer_sliding_diagnostics"); return assets,srcs


def fig8(out: Path) -> tuple[dict[str, str], list[Path]]:
    src=WILEY/"analysis/migrationbench_universe_publication_20261026/derived/structure_universe_points.csv"; rows=read_rows(src)
    fig=plt.figure(figsize=(11.0,6.5)); gs=fig.add_gridspec(2,2,hspace=.38,wspace=.30); ax=fig.add_subplot(gs[0,0])
    layer_colors={"L0_reference":"#8DA9BF","L2_DFT_path":"#C4547A","L3_interlayer_self":"#E68F1D","L3_self_relaxed_NEB":"#6E9E75"}
    for layer,color in layer_colors.items():
        sub=[r for r in rows if r.get("task_layer")==layer]; x=np.array([f(r,"x_pca") for r in sub]); y=np.array([f(r,"y_pca") for r in sub]); m=np.isfinite(x)&np.isfinite(y)
        if m.any(): ax.scatter(x[m],y[m],s=8,alpha=.42,color=color,label=layer.replace("_"," "))
    ax.set_xlabel("SOAP PCA 1"); ax.set_ylabel("SOAP PCA 2"); ax.set_title("A  Task-aware configuration map",loc="left"); ax.legend(frameon=False,fontsize=8); style_ax(ax)
    status_box(fig.add_subplot(gs[0,1]),"BLOCKED","No verified geometry_id-resolved join\nbetween descriptor distance and the\nFigure 3–4 QE error table.\nDo not substitute PCA distance for error.",color="#B23A48")
    status_box(fig.add_subplot(gs[1,0]),"BLOCKED","High-error image threshold and\nprospective held-out path are not\nfrozen with a common geometry key.\nNo recognition rate is reported.",color="#B23A48")
    status_box(fig.add_subplot(gs[1,1]),"BLOCKED","Seed disagreement and measured\nerror cannot be paired from current\nartifacts without a validated join.\nKeep as a supplemental map only.",color="#B23A48")
    fig.suptitle("Figure 8 | Where new QE data may be needed (conditional)",x=.02,ha="left",y=.995,fontsize=17,fontweight="bold")
    fig.text(.02,-.035,"Only panel A is supported by the current descriptor manifest; panels B–D are explicit placeholders, not conclusions.",fontsize=9,color="#555555")
    assets=save_figure(fig,out,"figure08_qe_diagnostic_conditional"); return assets,[src]


def write_tables(out: Path) -> list[Path]:
    tables=out/"tables"; tables.mkdir(parents=True,exist_ok=True)
    t1=tables/"TABLE1_MODEL_REFERENCE_AUDIT.tsv"
    rows=[
        ["route","source registry","observed model/seed scope","reference/protocol","current status","publication action"],
        ["Scratch / FT parity","analysis/rmse_train_test_20261026/MODEL_REGISTRY.tsv","4 families × 3 seeds; train/test summary","QE parity; force and energy fields","PUBLICATION_PROVISIONAL","resolve energy unit and held-out strata before claims"],
        ["Cr fixed images","analysis/neb_cross_element_triptychs_20261024/data/Cr_fixed_path_representative_profiles.tsv","from-scratch, FT-600K, FT-MultiT, Foundation; one recorded seed each","fixed images; QE SCF-warning reference","SUPPORTING_DIAGNOSTIC","retain G1 warning; define metric; do not call barrier"],
        ["Foundation self-relaxed paths","analysis/neb_cross_element_triptychs_20261024/foundation_self_relaxed_4x3_20261024/MODEL_REGISTRY.tsv","9 checkpoints × 4 elements × 3 path classes planned","endpoint-only IDPP initialization; campaign stopped for repair","EXCLUDE_FROM_MAIN_CLAIM","recover canonical run logs and geometry/endpoint gates"],
        ["1 ns MD","analysis/md_1ns_updated_publication_figure_20261025/md15_inputs.tsv","15 model IDs; 3 seeds for FT/Scratch groups","observed horizon, CN/RDF/VACF/MSD","PROVISIONAL_RAW_TRAJECTORIES_EXTERNAL","audit common initial state and trajectory integrity"],
        ["Interlayer sliding","analysis/foundation_interlayer_sliding_10model_20261024/MODEL_REGISTRY.tsv","10 checkpoints; class aggregation","fixed image + endpoint-only self proxy; QE SCF warning","RECONSTRUCTION_DIAGNOSTIC","retain class-SD semantics; do not rank physical barriers"],
    ]
    with t1.open("w",newline="") as h: csv.writer(h,delimiter="\t").writerows(rows)
    t2=tables/"TABLE2_TASK_DECISION_MATRIX.tsv"
    rows=[
        ["task","observable","reference","acceptance gate","denominator","current status","decision"],
        ["Near-equilibrium","energy/force parity","QE pointwise","finite arrays; exact model/data hashes; unit audit; held-out trajectory split","frames and force components","PENDING","do not pool across task families"],
        ["Within-gap fixed path","relative image energy","QE image set with SCF warning","same geometry IDs; explicit residual metric; G1 warning","10 fixed images per route","PENDING","diagnostic only"],
        ["Gap-to-QL / within-QL","fixed-path relative energy","same QE protocol","same image IDs and QE protocol across task","10 fixed images per path","PENDING","do not compare mixed protocols"],
        ["Generated NEB","model-generated path + fmax","model NEB log; QE cross-evaluation if used","full iteration history; convergence; endpoint identity; geometry alignment","attempts, not only completed rows","NOT_VALIDATED","hold main-text success/barrier claim"],
        ["1 ns MD","survival + structure + motion","matched trajectories","finite/continuous trajectory; actual horizon; common ensemble/step","all submitted runs, including failures","PROVISIONAL","stop curves at observed horizon"],
        ["Interlayer sliding","registry profile + force","fixed images and separate generated path","same coordinate/zero; SCF status; self-path convergence","valid images per category","DIAGNOSTIC","label proxy and class-SD"],
        ["QE acquisition diagnostic","distance/disagreement versus measured error","geometry-keyed QE error","frozen high-error threshold; prospective held-out join","independent held-out paths","PENDING","supplemental map only until validated"],
    ]
    with t2.open("w",newline="") as h: csv.writer(h,delimiter="\t").writerows(rows)
    return [t1,t2]


def write_manifest(out: Path, figure_assets: dict[str, dict[str, str]], sources: list[Path]) -> Path:
    manifest={"schema_version":"atomic_migration_blueprint_figure_package_v1","status":"REVIEW_PACKAGE_NOT_SCIENTIFIC_ACCEPTANCE","generated_by":str(Path(__file__).resolve()),"source_root":str(WILEY),"figures":figure_assets,"source_hashes":{str(p):sha256(p) for p in sources if p.exists()},"notes":["Canva connector was unavailable at build time and no external Canva design was overwritten.","All blocked panels are explicit placeholders; no synthetic observations are used.","SCF-warning, endpoint-only proxy, legacy and provisional states remain separate."]}
    p=out/"manifests/ATOMIC_MIGRATION_FIGURE_MANIFEST.json"; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n"); return p


def build_composite(out: Path, figure_assets: dict[str, dict[str, str]]) -> dict[str, str]:
    fig,axes=plt.subplots(2,4,figsize=(16,8.5)); names=[f"Figure {i}" for i in range(1,9)]
    statuses=["PARTIAL","PROVISIONAL","G1 WARNING","DIAGNOSTIC","BLOCKED","PROVISIONAL","SCF WARNING / PROXY","CONDITIONAL"]
    for ax,(name,status) in zip(axes.flat,zip(names,statuses)):
        path=Path(figure_assets[f"figure0{names.index(name)+1}"]["png"])
        img=plt.imread(path); ax.imshow(img); ax.set_title(f"{name} · {status}",fontsize=12,fontweight="bold",loc="left"); ax.axis("off")
    fig.suptitle("Atomic Migration Figure Blueprint | review canvas board",fontsize=20,fontweight="bold",x=.02,ha="left",y=.995)
    fig.text(.02,.01,"Green/black panels are data-backed diagnostics; red labels are explicit release gates or placeholders. See the audit package for publication decisions.",fontsize=10,color="#555555")
    return save_figure(fig,out,"atomic_migration_blueprint_canvas_board",dpi=300)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,default=OUT_DEFAULT); args=parser.parse_args(); out=args.out
    set_house_style(); figdir=out/"figures"; figdir.mkdir(parents=True,exist_ok=True)
    builders=[fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8]; assets={}; sources=[]
    for i,builder in enumerate(builders,1):
        a,s=builder(figdir); assets[f"figure0{i}"]=a; sources.extend(s)
    assets["composite"]=build_composite(figdir,assets)
    tables=write_tables(out); sources.extend(tables)
    manifest=write_manifest(out,assets,sources)
    print(json.dumps({"manifest":str(manifest),"figures":assets,"tables":[str(p) for p in tables]},indent=2))


if __name__ == "__main__":
    main()
