#!/usr/bin/env python3
"""Render the 15-model, evidence-bounded 1 ns MD figure.

The five plotted families are represented by 3 Foundation checkpoints plus
three training seeds for each of FT-600K, FT-MultiT, Scratch-600K and
Scratch-MultiT.  A trajectory is plotted only over its observed horizon; a
scientific-threshold failure is never extended to 1 ns.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "md15_analysis_r03"
PARTIAL_DATA = HERE / "md15_partial_rdf_r05" / "output"
GLOBAL_DATA = HERE / "md15_global_rdf_r06" / "output"
RESULTS = HERE / "results"
RELEASE_TAG = "R02"
STEM = RESULTS / f"fig_md_1ns_updated_A_to_F_15models_{RELEASE_TAG}"
STATUS_TSV = HERE / "MD15_STATUS_AUDIT_R02.tsv"
REMOTE_SUMMARY = DATA / "analysis_summary.tsv"
# Panel switches are kept explicit so each additive release can preserve the
# older figure while changing only the requested panel.
D_MODE = "vacf"
F_MODE = "dxy_ratio"
E_MODE = "raw_rdf"

# The RDF files are normalized with the fixed cell volume used by the MD15
# analysis.  The CN panel integrates the already validated Cr--Cr RDF; it does
# not introduce a second raw-trajectory normalization.
CELL_VOLUME_A3 = 150190.66347415018
CN_CUTOFF_A = 2.5
CN_PAIR_CUTOFFS = {
    "Cr-Sb": 3.2,
    "Cr-Te": 4.1,
    "Sb-Te": 4.2,
}
CN_PAIR_LABELS = {
    "Cr-Sb": "Cr–Sb",
    "Cr-Te": "Cr–Te",
    "Sb-Te": "Sb–Te",
}
CN_PAIR_COLORS = {
    "Cr-Sb": "#2E5B8A",
    "Cr-Te": "#67ACE5",
    "Sb-Te": "#E07A4D",
}
CN_PAIR_DATA_KEYS = {
    "Cr-Sb": "d_cn_cr_sb_neighbors_per_center",
    "Cr-Te": "d_cn_cr_te_neighbors_per_center",
    "Sb-Te": "d_cn_sb_te_neighbors_per_center",
}
CN_GROUPS = {
    "Foundation": ["Foundation_MPA0", "Foundation_MP0b3", "Foundation_OMAT0"],
    "FT-600K": ["FT600K_A2_seed123", "FT600K_A2_seed234", "FT600K_A2_seed345"],
    "FT-MultiT": ["FTMultiT_A2_seed123", "FTMultiT_A2_seed234", "FTMultiT_A2_seed345"],
    "Scratch": [
        "Scratch600K_Lmax1_seed123", "Scratch600K_Lmax1_seed234", "Scratch600K_Lmax1_seed345",
        "ScratchMultiT_Lmax1_seed123", "ScratchMultiT_Lmax1_seed234", "ScratchMultiT_Lmax1_seed345",
    ],
}
CN_GROUP_BY_MODEL = {model_id: group for group, ids in CN_GROUPS.items() for model_id in ids}
GLOBAL_GROUP_COLORS = {
    "Foundation": "#CF4520",
    "FT-600K": "#002E71",
    "FT-MultiT": "#67ACE5",
    "Scratch": "#F4A340",
}

COLORS = {
    "Foundation": "#CF4520",
    "FT-600K": "#002E71",
    "FT-MultiT": "#67ACE5",
    "Scratch-600K": "#FF9F1B",
    "Scratch-MultiT": "#F4C95D",
}

MODELS = [
    ("Foundation_MPA0", "Foundation", "MPA-0", 0, "v", "-"),
    ("Foundation_MP0b3", "Foundation", "MP-0b3", 0, "v", "--"),
    ("Foundation_OMAT0", "Foundation", "OMAT-0", 0, "v", ":"),
    ("FT600K_A2_seed123", "FT-600K", "seed 123", 123, "o", "--"),
    ("FT600K_A2_seed234", "FT-600K", "seed 234", 234, "s", ":"),
    ("FT600K_A2_seed345", "FT-600K", "seed 345", 345, "^", "-"),
    ("FTMultiT_A2_seed123", "FT-MultiT", "seed 123", 123, "o", "--"),
    ("FTMultiT_A2_seed234", "FT-MultiT", "seed 234", 234, "s", ":"),
    ("FTMultiT_A2_seed345", "FT-MultiT", "seed 345", 345, "^", "-"),
    ("Scratch600K_Lmax1_seed123", "Scratch-600K", "seed 123", 123, "o", "--"),
    ("Scratch600K_Lmax1_seed234", "Scratch-600K", "seed 234", 234, "s", ":"),
    ("Scratch600K_Lmax1_seed345", "Scratch-600K", "seed 345", 345, "^", "-"),
    ("ScratchMultiT_Lmax1_seed123", "Scratch-MultiT", "seed 123", 123, "o", "--"),
    ("ScratchMultiT_Lmax1_seed234", "Scratch-MultiT", "seed 234", 234, "s", ":"),
    ("ScratchMultiT_Lmax1_seed345", "Scratch-MultiT", "seed 345", 345, "^", "-"),
]
MODEL_INFO = {m[0]: m for m in MODELS}
CSV_CACHE = {}


def read_csv(path):
    path = Path(path)
    key = str(path)
    if key in CSV_CACHE:
        return CSV_CACHE[key]
    # Read the small derived CSV in one filesystem operation.  The local
    # workspace may be backed by a file-provider layer where thousands of
    # incremental BufferedReader reads are disproportionately slow.
    rows = list(csv.DictReader(path.read_text().splitlines()))
    CSV_CACHE[key] = rows
    return rows


def read_status():
    out = {}
    if not STATUS_TSV.exists():
        return out
    with STATUS_TSV.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[row["model_id"]] = row
    return out


def load_series(model_id):
    rows = read_csv(DATA / "series" / f"{model_id}.csv")
    cols = {k: np.asarray([float(r[k]) if r[k] not in ("", "nan", "NaN") else np.nan for r in rows], dtype=float) for k in rows[0]}
    return cols


def load_msd(model_id):
    rows = read_csv(DATA / "msd" / f"{model_id}.csv")
    return (np.asarray([float(r["lag_ps"]) for r in rows]),
            np.asarray([float(r["msdxy_A2"]) for r in rows]))


def load_vacf(model_id):
    rows = read_csv(DATA / "vacf" / f"{model_id}.csv")
    return (np.asarray([float(r["lag_ps"]) for r in rows]),
            np.asarray([float(r["vacf_normalized"]) for r in rows]))


def load_rdf(model_id):
    rows = read_csv(DATA / "rdf" / f"{model_id}.csv")
    return (np.asarray([float(r["r_A"]) for r in rows]),
            np.asarray([float(r["g_CrCr"]) for r in rows]))


def load_global_rdf(model_id):
    rows = read_csv(GLOBAL_DATA / model_id / f"{model_id}.csv")
    return (
        np.asarray([float(r["r_A"]) for r in rows]),
        np.asarray([float(r["g_global"]) for r in rows]),
    )


def smooth_curve(y, sigma_bins=1.15):
    """Conservative Gaussian smoothing for display only; raw RDF is retained."""
    radius = max(2, int(np.ceil(3.0 * sigma_bins)))
    grid = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (grid / sigma_bins) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(np.asarray(y, dtype=float), (radius, radius), mode="edge")
    return np.clip(np.convolve(padded, kernel, mode="valid"), 0.0, None)


def load_short_range_cn(model_id, cutoff=CN_CUTOFF_A):
    """Integrate the validated Cr--Cr RDF up to one fixed cutoff radius."""
    rows = read_csv(DATA / "rdf" / f"{model_id}.csv")
    if not rows:
        return float("nan")
    centers = np.asarray([float(r["r_A"]) for r in rows], dtype=float)
    dr = float(np.median(np.diff(centers))) if len(centers) > 1 else 0.1
    ncr = float(rows[0]["mean_nCr"])
    rho_cr = ncr / CELL_VOLUME_A3
    cn = 0.0
    for center, row in zip(centers, rows):
        g = float(row["g_CrCr"])
        if not np.isfinite(g):
            continue
        r0 = max(0.0, center - dr / 2.0)
        r1 = center + dr / 2.0
        if r1 <= cutoff + 1e-12:
            cn += 4.0 * np.pi * rho_cr * g * (r1**3 - r0**3) / 3.0
    return float(cn)


def load_partial_cn(model_id, pair, cutoff):
    """Integrate a species-resolved partial RDF to a fixed pair cutoff."""
    rows = [
        row for row in read_csv(PARTIAL_DATA / model_id / f"{model_id}.csv")
        if row["pair"] == pair
    ]
    if not rows:
        return float("nan")
    centers = np.asarray([float(r["r_A"]) for r in rows], dtype=float)
    dr = float(np.median(np.diff(centers))) if len(centers) > 1 else 0.05
    rho_partner = float(rows[0]["mean_n_partner"]) / float(rows[0]["mean_volume_A3"])
    cn = 0.0
    for center, row in zip(centers, rows):
        g = float(row["g_pair"])
        if not np.isfinite(g):
            continue
        r0 = max(0.0, center - dr / 2.0)
        r1 = center + dr / 2.0
        if r1 <= cutoff + 1e-12:
            cn += 4.0 * np.pi * rho_partner * g * (r1**3 - r0**3) / 3.0
    return float(cn)


def cn_group_summary():
    summary = []
    per_model = {}
    for model_id, *_ in MODELS:
        per_model[model_id] = {
            pair: load_partial_cn(model_id, pair, cutoff)
            for pair, cutoff in CN_PAIR_CUTOFFS.items()
        }
    for group, model_ids in CN_GROUPS.items():
        for pair in CN_PAIR_CUTOFFS:
            values = np.asarray([per_model[mid][pair] for mid in model_ids], dtype=float)
            values = values[np.isfinite(values)]
            summary.append({
                "group": group,
                "pair": pair,
                "n_models": int(values.size),
                "mean_neighbors_per_center": float(np.mean(values)) if values.size else float("nan"),
                "std_neighbors_per_center": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            })
    return per_model, summary


def style_for(model_id):
    _, family, _, _, marker, linestyle = MODEL_INFO[model_id]
    return COLORS[family], marker, linestyle


def finite_xy(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def draw_observed(ax, model_id, x, y, linewidth=1.0, alpha=0.85, clip=None, zorder=2):
    color, marker, linestyle = style_for(model_id)
    x, y = finite_xy(x, y)
    if not len(x):
        return
    if clip is not None:
        lo, hi = clip
        y_draw = np.clip(y, lo, hi)
    else:
        y_draw = y
    ax.plot(x, y_draw, color=color, lw=linewidth, ls=linestyle, alpha=alpha, zorder=zorder)
    if clip is not None and (np.any(y < lo) or np.any(y > hi)):
        idx = np.where((y < lo) | (y > hi))[0][0]
        ax.plot(x[idx], np.clip(y[idx], lo, hi), marker="v", ms=4.3, color=color, mec="white", mew=.45, zorder=5)


def axes_style(ax, title, letter, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=13.5, fontweight="bold", pad=7)
    ax.text(-0.115, 1.06, letter, transform=ax.transAxes, fontsize=19, fontweight="bold", va="top")
    ax.tick_params(which="major", labelsize=10.3, width=1.05, length=4.3, direction="out")
    ax.tick_params(which="minor", width=.75, length=2.4, direction="out")
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.05)
        spine.set_color("#222222")
    ax.grid(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11.6, fontweight="bold", labelpad=5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11.6, fontweight="bold", labelpad=5)


def add_time_inset(ax, panel):
    """Add a compact early-time zoom matching the supplied G-panel grammar."""
    # A/B/C use the lower-right corner because their main traces occupy the
    # upper-right steady-state band.  F retains the reference G-panel
    # placement in the upper-right corner.
    inset_position = [0.56, 0.08, 0.38, 0.34] if panel in {"A", "B", "C"} else [0.56, 0.56, 0.38, 0.34]
    inset = ax.inset_axes(inset_position, facecolor="white", zorder=10)
    if panel == "A":
        getter = lambda model_id: (load_series(model_id)["time_ps"], load_series(model_id)["temperature_K"])
        xlim, ylim = (190, 270), (450, 720)
    elif panel == "B":
        getter = lambda model_id: (load_series(model_id)["time_ps"], load_series(model_id)["pressure_GPa"])
        xlim, ylim = (190, 270), (-2.0, 2.0)
    elif panel == "C":
        getter = load_msd
        xlim, ylim = (0, 80), (0, 20)
    elif panel == "F":
        getter = load_vacf
        xlim, ylim = (0, 8), (-0.2, 1.05)
    else:
        raise ValueError(f"unsupported inset panel: {panel}")
    for model_id, *_ in MODELS:
        x, y = getter(model_id)
        x, y = finite_xy(x, y)
        mask = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
        if not np.any(mask):
            continue
        color, _, linestyle = style_for(model_id)
        inset.plot(x[mask], y[mask], color=color, lw=.72, ls=linestyle, alpha=.9)
    if panel in {"B", "F"}:
        inset.axhline(0, color="#666666", ls=(0, (2, 2)), lw=.55, zorder=0)
    inset.set_xlim(*xlim)
    inset.set_ylim(*ylim)
    inset.tick_params(which="major", labelsize=7.4, width=.65, length=2.0, direction="out", pad=1.0)
    inset.tick_params(which="minor", width=.45, length=1.2, direction="out")
    for text in inset.get_xticklabels() + inset.get_yticklabels():
        text.set_fontweight("bold")
    for spine in inset.spines.values():
        spine.set_linewidth(.75)
        spine.set_color("#222222")
    inset.grid(False)
    return inset


def fit_slope(model_id, lo, hi):
    x, y = load_msd(model_id)
    mask = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
    if np.count_nonzero(mask) < 8:
        return None
    slope = np.polyfit(x[mask], y[mask], 1)[0]
    return slope / 4.0e4 if np.isfinite(slope) else None


def model_key_handles():
    handles, labels = [], []
    for model_id, family, short, _, marker, linestyle in MODELS:
        handles.append(Line2D([0], [0], color=COLORS[family], lw=1.7, ls=linestyle, marker=marker, ms=3.7,
                              mfc="white" if marker == "v" else COLORS[family], mec=COLORS[family]))
        prefix = {"Foundation": "F", "FT-600K": "FT600K", "FT-MultiT": "FTMultiT", "Scratch-600K": "S600K", "Scratch-MultiT": "SMultiT"}[family]
        labels.append(f"{prefix} · {short}")
    return handles, labels


def render():
    if not REMOTE_SUMMARY.exists():
        raise FileNotFoundError(f"missing remote analysis output: {REMOTE_SUMMARY}")
    status = read_status()
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, aa = plt.subplots(2, 3, figsize=(14.2, 8.25), dpi=400)
    aa = aa.ravel()

    # A: absolute simulation time.  Off-scale catastrophic endpoints are
    # clipped only for display and marked at the clipping boundary.
    for model_id, *_ in MODELS:
        s = load_series(model_id)
        draw_observed(aa[0], model_id, s["time_ps"], s["temperature_K"], linewidth=1.0, alpha=.84, clip=(250, 820))
    aa[0].axhline(600, color="#666666", ls=(0, (3, 2)), lw=.85)
    aa[0].set_xlim(190, 1010)
    aa[0].set_ylim(250, 820)
    axes_style(aa[0], "Temperature stability", "A", "Time (ps)", "Temperature (K)")
    add_time_inset(aa[0], "A")

    # B: pressure from the raw stress tensor, P = -tr(sigma)/3.
    for model_id, *_ in MODELS:
        s = load_series(model_id)
        draw_observed(aa[1], model_id, s["time_ps"], s["pressure_GPa"], linewidth=1.0, alpha=.84, clip=(-2.0, 2.0))
    aa[1].axhline(0, color="#666666", ls=(0, (3, 2)), lw=.85)
    aa[1].set_xlim(190, 1010)
    aa[1].set_ylim(-2.0, 2.0)
    axes_style(aa[1], "Pressure stability", "B", "Time (ps)", "Hydrostatic pressure (GPa)")
    add_time_inset(aa[1], "B")

    # C: per-model xy-MSD from the first retained production frame.
    for model_id, *_ in MODELS:
        x, y = load_msd(model_id)
        draw_observed(aa[2], model_id, x, y, linewidth=1.0, alpha=.84, clip=(0, 80))
    aa[2].set_xlim(0, 800)
    aa[2].set_ylim(0, 80)
    axes_style(aa[2], "Mean-square displacement", "C", "Lag time (ps)", r"MSD$_{xy}$ ($\AA^2$)")
    add_time_inset(aa[2], "C")

    if D_MODE == "cn_short":
        # D: short-range Cr--Cr coordination/contact number derived from the
        # same zero-inclusive RDF used in panel E.  A horizontal model list
        # keeps all 15 models individually visible without collapsing them to
        # family means.
        positions = np.arange(len(MODELS))
        labels = [
            "F–MPA-0", "F–MP-0b3", "F–OMAT-0",
            "FT6–123", "FT6–234", "FT6–345",
            "FTM–123", "FTM–234", "FTM–345",
            "S6–123", "S6–234", "S6–345",
            "SMT–123", "SMT–234", "SMT–345",
        ]
        for pos, (model_id, *_rest) in zip(positions, MODELS):
            color, marker, _ = style_for(model_id)
            value = load_short_range_cn(model_id)
            aa[3].plot(
                value, pos, marker=marker, linestyle="None", ms=5.5,
                color=color, mfc="white" if marker == "v" else color,
                mec=color, mew=1.0, zorder=5,
            )
        aa[3].set_yticks(positions, labels)
        aa[3].set_xlim(-.03, .82)
        aa[3].set_ylim(-1, len(MODELS))
        aa[3].tick_params(axis="y", labelsize=8.8, pad=2)
        axes_style(
            aa[3], "Short-range Cr–Cr coordination", "D",
            r"$N_{\mathrm{Cr-Cr}}$ ($r<2.5\,\AA$; neighbors/Cr)", "Model",
        )
    elif D_MODE == "cn_bars":
        # D: pair-resolved CN bars.  The four x-axis groups follow the
        # established model classes; the six Scratch trajectories are pooled
        # because the requested figure has four clusters.
        _, cn_summary = cn_group_summary()
        group_names = list(CN_GROUPS)
        positions = np.arange(len(group_names), dtype=float)
        width = 0.22
        offsets = np.linspace(-width, width, len(CN_PAIR_CUTOFFS))
        for offset, pair in zip(offsets, CN_PAIR_CUTOFFS):
            rows = [row for row in cn_summary if row["pair"] == pair]
            means = [next(row["mean_neighbors_per_center"] for row in rows if row["group"] == group) for group in group_names]
            stds = [next(row["std_neighbors_per_center"] for row in rows if row["group"] == group) for group in group_names]
            aa[3].bar(
                positions + offset, means, width=width * 0.92,
                yerr=stds, capsize=2.0,
                color=CN_PAIR_COLORS[pair], edgecolor="#222222", linewidth=.55,
                error_kw={"ecolor": "#222222", "elinewidth": .8, "capthick": .8},
                label=CN_PAIR_LABELS[pair], zorder=3,
            )
        aa[3].set_xticks(positions, ["Foundation", "FT-600K", "FT-MultiT", "Scratch"])
        aa[3].set_xlim(-.55, len(group_names) - .45)
        aa[3].set_ylim(0, 7.0)
        axes_style(
            aa[3], "Local coordination number", "D",
            "Model class", "CN (neighbors/center atom)",
        )
        aa[3].legend(
            loc="upper left", frameon=False, fontsize=9.0,
            handlelength=1.35, borderaxespad=.2, labelspacing=.25,
        )
    else:
        # D: multi-origin normalized VACF computed from every fourth trajectory
        # frame; short trajectories retain their short lag range.
        for model_id, *_ in MODELS:
            x, y = load_vacf(model_id)
            draw_observed(aa[3], model_id, x, y, linewidth=1.0, alpha=.84, clip=(-.35, 1.05))
        aa[3].axhline(0, color="#666666", ls=(0, (3, 2)), lw=.85)
        aa[3].set_xlim(0, 800)
        aa[3].set_ylim(-.35, 1.05)
        axes_style(aa[3], "Velocity autocorrelation", "D", "Lag time (ps)", "Normalized VACF")

    # E: two additive display modes are available for the RDF replacement.
    # Both retain the raw source data: R06S smooths the 15 Cr--Cr curves only
    # for display, while R06G recomputes an all-atom RDF from the raw Rockfish
    # trajectories and adds model-class mean +/- sample-SD envelopes.
    if E_MODE == "smooth_crcr":
        for model_id, *_ in MODELS:
            x, y = load_rdf(model_id)
            draw_observed(aa[4], model_id, x, smooth_curve(y), linewidth=1.08, alpha=.82, zorder=2)
        aa[4].axhline(0, color="#999999", lw=.75, zorder=1)
        aa[4].axhline(1, color="#666666", ls=(0, (3, 2)), lw=.85)
        aa[4].set_xlim(0, 10)
        aa[4].set_yscale("symlog", linthresh=1.0, linscale=1.0)
        aa[4].set_ylim(0, 200)
        axes_style(aa[4], r"Smoothed Cr–Cr radial distribution", "E", r"r ($\AA$)", r"$g_{\mathrm{Cr-Cr}}(r)$")
    elif E_MODE == "global_grouped":
        global_curves = {}
        for model_id, *_ in MODELS:
            x, y = load_global_rdf(model_id)
            global_curves[model_id] = (x, y)
            # Thin, low-alpha curves preserve the complete 15-model evidence
            # without making the structural summary visually dominant.
            model_color, _, model_linestyle = style_for(model_id)
            aa[4].plot(
                x, y, color=model_color, ls=model_linestyle,
                lw=.55, alpha=.22, zorder=1,
            )
        x0 = next(iter(global_curves.values()))[0]
        group_handles = []
        for group, model_ids in CN_GROUPS.items():
            curves = np.asarray([global_curves[mid][1] for mid in model_ids], dtype=float)
            mean = np.nanmean(curves, axis=0)
            std = np.nanstd(curves, axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(mean)
            mean_display = smooth_curve(mean, sigma_bins=1.15)
            std_display = smooth_curve(std, sigma_bins=1.15)
            color = GLOBAL_GROUP_COLORS[group]
            lower = np.clip(mean_display - std_display, 0.0, None)
            upper = np.clip(mean_display + std_display, 0.0, None)
            aa[4].fill_between(x0, lower, upper, color=color, alpha=.13, lw=0, zorder=2)
            aa[4].plot(x0, mean_display, color=color, lw=1.65, zorder=3)
            group_handles.append(Line2D([0], [0], color=color, lw=1.65, label=group))
        aa[4].axhline(0, color="#999999", lw=.75, zorder=0)
        aa[4].axhline(1, color="#666666", ls=(0, (3, 2)), lw=.85, zorder=0)
        aa[4].set_xlim(0, 10)
        aa[4].set_ylim(0, 8)
        axes_style(aa[4], r"All-atom global radial distribution", "E", r"r ($\AA$)", r"$g_{\mathrm{all}}(r)$")
        aa[4].legend(
            handles=group_handles, loc="upper right", frameon=False, fontsize=8.6,
            ncol=2, handlelength=1.35, columnspacing=.8, handletextpad=.35,
            borderaxespad=.25, labelspacing=.25,
        )
    else:
        for model_id, *_ in MODELS:
            x, y = load_rdf(model_id)
            draw_observed(aa[4], model_id, x, y, linewidth=1.05, alpha=.83, zorder=2)
        aa[4].axhline(0, color="#999999", lw=.75, zorder=1)
        aa[4].axhline(1, color="#666666", ls=(0, (3, 2)), lw=.85)
        aa[4].set_xlim(0, 10)
        aa[4].set_yscale("symlog", linthresh=1.0, linscale=1.0)
        aa[4].set_ylim(0, 80)
        axes_style(aa[4], r"Cr–Cr radial distribution", "E", r"r ($\AA$)", r"$g_{\mathrm{Cr-Cr}}(r)$")

    if F_MODE == "vacf":
        # F: the original VACF panel requested by the user.  This branch
        # deliberately reuses the same raw per-model VACF series as panel D;
        # all other panels and their data remain unchanged.
        for model_id, *_ in MODELS:
            x, y = load_vacf(model_id)
            draw_observed(aa[5], model_id, x, y, linewidth=1.0, alpha=.84, clip=(-.35, 1.05))
        aa[5].axhline(0, color="#666666", ls=(0, (3, 2)), lw=.85)
        aa[5].set_xlim(0, 800)
        aa[5].set_ylim(-.35, 1.05)
        axes_style(aa[5], "Velocity autocorrelation", "F", "Lag time (ps)", "Normalized VACF")
        add_time_inset(aa[5], "F")
    else:
        # F: Dxy late/early window ratio.  A ratio is only estimable when both
        # windows are observed; all 15 models remain represented by a colored x
        # at the bottom for the scientifically undefined cases.
        positions = np.arange(len(MODELS))
        missing_y = .06
        for i, (model_id, family, *_rest) in enumerate(MODELS):
            color, marker, _ = style_for(model_id)
            d1 = fit_slope(model_id, 200, 500)
            d2 = fit_slope(model_id, 500, 900)
            if d1 is not None and d2 is not None and d1 > 0 and d2 > 0:
                ratio = d2 / d1
                aa[5].errorbar(i, ratio, fmt=marker, color=color, mfc="white" if marker == "v" else color,
                                mec=color, ms=5.0, mew=1.0, capsize=2.2, lw=1.0, zorder=5)
            else:
                aa[5].plot(i, missing_y, marker="x", color=color, ms=5.4, mew=1.35, zorder=5)
        aa[5].axhline(1, color="#666666", ls=(0, (3, 2)), lw=.85)
        aa[5].set_xlim(-.7, len(MODELS) - .3)
        aa[5].set_xticks(positions, ["F–MPA", "F–MP", "F–OMAT", "600K–123", "600K–234", "600K–345",
                                     "MT–123", "MT–234", "MT–345", "S600K–123", "S600K–234", "S600K–345",
                                     "SMT–123", "SMT–234", "SMT–345"], rotation=62, ha="right")
        aa[5].set_yscale("log")
        aa[5].set_ylim(.05, 20)
        aa[5].text(.02, .04, "× = no two-window estimate before failure", transform=aa[5].transAxes, fontsize=9.2, color="#555555")
        axes_style(aa[5], r"$D_{xy}$ window ratio", "F", None, r"500–900 / 200–500 ps")

    handles, labels = model_key_handles()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, .002),
               fontsize=9.3, handlelength=2.05, columnspacing=1.25, handletextpad=.45)
    fig.subplots_adjust(left=.075, right=.99, top=.965, bottom=.285, wspace=.34, hspace=.44)
    RESULTS.mkdir(exist_ok=True)
    outs = [STEM.with_suffix(ext) for ext in (".pdf", ".png", ".svg")]
    fig.savefig(outs[0], bbox_inches="tight", facecolor="white")
    fig.savefig(outs[1], dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(outs[2], bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary_rows = []
    for model_id, family, short, seed, *_ in MODELS:
        row = {
            "model_id": model_id,
            "family": family,
            "display_label": short,
        }
        if D_MODE == "cn_short":
            row["d_cn_crcr_lt_2p5_A_neighbors_per_Cr"] = load_short_range_cn(model_id)
        elif D_MODE == "cn_bars":
            row["d_cn_group"] = CN_GROUP_BY_MODEL[model_id]
            row.update({
                CN_PAIR_DATA_KEYS[pair]: load_partial_cn(model_id, pair, cutoff)
                for pair, cutoff in CN_PAIR_CUTOFFS.items()
            })
        if F_MODE == "vacf":
            x, y = load_vacf(model_id)
            row.update({
                "f_vacf_max_lag_ps": float(np.max(x)) if len(x) else "",
                "f_vacf_initial": float(y[0]) if len(y) else "",
                "f_vacf_min": float(np.min(y)) if len(y) else "",
                "f_vacf_max": float(np.max(y)) if len(y) else "",
            })
        else:
            d1 = fit_slope(model_id, 200, 500)
            d2 = fit_slope(model_id, 500, 900)
            row.update({
                "dxy_200_500_cm2_s": "" if d1 is None else d1,
                "dxy_500_900_cm2_s": "" if d2 is None else d2,
                "dxy_ratio": "" if d1 is None or d2 is None or d1 <= 0 else d2 / d1,
            })
        row.update({k: status.get(model_id, {}).get(k, "") for k in ("last_step", "last_time_ps", "status", "reason", "metrics_core_finite")})
        summary_rows.append(row)
    summary_path = HERE / f"MD15_FIGURE_DATA_{RELEASE_TAG}.tsv"
    with summary_path.open("w", newline="") as f:
        fields = list(summary_rows[0])
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader(); w.writerows(summary_rows)

    cn_summary_path = None
    if D_MODE == "cn_bars":
        _, cn_group_rows = cn_group_summary()
        cn_summary_path = HERE / f"MD15_CN_GROUP_SUMMARY_{RELEASE_TAG}.tsv"
        with cn_summary_path.open("w", newline="") as f:
            fields = ["group", "pair", "cutoff_A", "n_models", "mean_neighbors_per_center", "std_neighbors_per_center"]
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            w.writeheader()
            for row in cn_group_rows:
                w.writerow({**row, "cutoff_A": CN_PAIR_CUTOFFS[row["pair"]]})

    manifest = {
        "figure": STEM.name,
        "panel_order": {
            "A": "Temperature stability",
            "B": "Pressure stability",
            "C": "MSD_xy",
            "D": (
                "Pair-resolved CN grouped bars" if D_MODE == "cn_bars"
                else "Short-range Cr-Cr coordination" if D_MODE == "cn_short"
                else "Normalized VACF"
            ),
            "E": (
                "Smoothed Cr-Cr RDF" if E_MODE == "smooth_crcr"
                else "All-atom global RDF" if E_MODE == "global_grouped"
                else "Cr-Cr RDF"
            ),
            "F": "Normalized VACF" if F_MODE == "vacf" else "Dxy late/early window ratio",
        },
        "model_count": len(MODELS),
        "model_matrix": "3 Foundation checkpoints + 3 seeds x FT-600K + 3 seeds x FT-MultiT + 3 seeds x Scratch-600K + 3 seeds x Scratch-MultiT",
        "style_reference": "FIG168_MODEL_STYLE_YELLOW_SCRATCH_R04.json",
        "palette": COLORS,
        "trajectory_policy": "plot each model only through its observed raw trajectory horizon; failures are retained and never extended",
        "temperature_definition": "instantaneous kinetic temperature recomputed from extxyz momenta with ASE masses and k_B",
        "pressure_definition": "P = -trace(stress)/3, converted from eV/A^3 to GPa with 160.21766208",
        "msd_definition": "xy mean-square displacement from the first retained production frame with triclinic fractional unwrapping",
        "vacf_definition": "normalized multi-origin VACF using every fourth retained frame; short trajectories retain their observed lag range",
        "rdf_definition": (
            "All-atom ordered-neighbor RDF from raw trajectories, triclinic minimum-image distances, rmax=10 A, dr=0.1 A, every 20th eligible frame"
            if E_MODE == "global_grouped"
            else "Cr-Cr pair RDF from raw trajectories, triclinic minimum-image distances, rmax=12 A, dr=0.1 A, every 20th eligible frame; panel displays 0-10 A"
        ),
        "rdf_zero_policy": "retain g(r)=0 and use a zero-inclusive symlog ordinate; the 0-2 A baseline is not converted to NaN",
        "cn_definition": "Cr-Cr coordination/contact number integrated from the validated Cr-Cr RDF as 4*pi*rho_Cr*integral[g_CrCr(r) r^2 dr]",
        "cn_cutoff_A": CN_CUTOFF_A,
        "cn_normalization": "mean Cr number density from the same fixed-cell RDF normalization; units are neighbors per Cr",
        "d_panel_mode": D_MODE,
        "e_panel_mode": E_MODE,
        "f_panel_mode": F_MODE,
        "inset_panels": ["A", "B", "C", "F"],
        "inset_definition": "Compact early-time zooms use the same 15 model curves and palette: A/B/C lower-right, F upper-right; windows are A/B 190-270 ps, C 0-80 ps, F 0-8 ps",
        "source_summary": str(REMOTE_SUMMARY),
        "source_status": str(STATUS_TSV),
        "source_inputs": str(HERE / "md15_inputs.tsv"),
        "outputs": [],
    }
    if D_MODE == "cn_bars":
        manifest.update({
            "cn_definition": "Pair-resolved coordination number N_A-B(rc) = 4*pi*rho_B*integral_0^rc[g_A-B(r) r^2 dr] from raw Rockfish trajectories",
            "cn_pair_cutoffs_A": CN_PAIR_CUTOFFS,
            "cn_pair_cutoff_rule": "common cutoffs fixed from the first minimum of the pooled stable-reference partial RDF: Foundation MPA-0, Foundation MP-0b3, Foundation OMAT-0 and FT600K seed234",
            "cn_pairs": {pair: {"center": pair.split("-")[0], "partner": pair.split("-")[1]} for pair in CN_PAIR_CUTOFFS},
            "cn_grouping": CN_GROUPS,
            "cn_error_bar": "sample standard deviation across model-internal points; Scratch pools 600K and MultiT seeds into the requested fourth cluster",
            "cn_group_summary": str(cn_summary_path) if cn_summary_path else "",
        })
    if E_MODE == "smooth_crcr":
        manifest.update({
            "rdf_display_smoothing": "Gaussian convolution applied to each Cr-Cr RDF curve for display only; sigma=1.15 bins at dr=0.1 A; raw RDF files are unchanged",
            "rdf_display_ylim": [0, 200],
        })
    elif E_MODE == "global_grouped":
        manifest.update({
            "global_rdf_definition": "All-atom ordered-neighbor RDF from every atom pair in the raw trajectory, normalized by sum[N(N-1)/V] and spherical shell volume",
            "global_rdf_sampling": "Raw Rockfish extxyz trajectories, triclinic minimum-image distances, rmax=10 A, dr=0.1 A, every 20th eligible frame",
            "global_rdf_display": "15 model-specific curves are retained as thin background lines; four model-class curves show mean +/- sample standard deviation with display-only Gaussian smoothing",
            "global_rdf_grouping": CN_GROUPS,
            "global_rdf_display_ylim": [0, 8],
        })
    source_paths = [
        HERE / "md15_inputs.tsv",
        HERE / "compute_md15.cpp",
        HERE / "render_md15_figure_R02.py",
        HERE / f"render_md15_figure_{RELEASE_TAG}.py",
        REMOTE_SUMMARY,
        STATUS_TSV,
        summary_path,
        HERE / "remote_evidence_R02" / "FT_MODEL_REGISTRY.tsv",
        HERE / "remote_evidence_R02" / "SCRATCH_MODEL_REGISTRY.tsv",
        HERE / "remote_evidence_R02" / "FT_CAMPAIGN_MANIFEST.json",
        HERE / "remote_evidence_R02" / "MODEL_REGISTRY.tsv",
        HERE / "remote_evidence_R02" / "CAMPAIGN_MANIFEST.json",
        HERE / f"FIGURE_CAPTION_15MODELS_{RELEASE_TAG}.md",
        HERE / f"MD15_STATUS_REPORT_{RELEASE_TAG}.md",
    ]
    if D_MODE == "cn_bars":
        source_paths.extend([
            HERE / "compute_partial_rdf_r05.cpp",
            HERE / "run_partial_rdf_r05.sh",
            cn_summary_path,
        ])
        source_paths.extend(
            path
            for model_id, *_ in MODELS
            for path in (
                PARTIAL_DATA / model_id / f"{model_id}.csv",
                PARTIAL_DATA / model_id / f"{model_id}.meta.tsv",
                PARTIAL_DATA / model_id / "partial_rdf_summary.tsv",
            )
        )
    if E_MODE == "global_grouped":
        source_paths.extend([
            HERE / "compute_global_rdf_r06.cpp",
            HERE / "run_global_rdf_r06.sh",
        ])
        source_paths.extend(
            path
            for model_id, *_ in MODELS
            for path in (
                GLOBAL_DATA / model_id / f"{model_id}.csv",
                GLOBAL_DATA / model_id / f"{model_id}.meta.tsv",
                GLOBAL_DATA / model_id / "global_rdf_summary.tsv",
            )
        )
    for path in source_paths:
        if path.exists():
            manifest.setdefault("source_artifacts", []).append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for path in outs:
        manifest["outputs"].append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (HERE / f"FIGURE_MANIFEST_15MODELS_{RELEASE_TAG}.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    render()
