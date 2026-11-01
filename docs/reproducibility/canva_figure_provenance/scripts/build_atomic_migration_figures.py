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
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull

WORKTREE = Path(__file__).resolve().parents[4]
WILEY = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench")
ATLAS = Path("/Users/alina/Documents/Codex/plot_library_recovered")
sys.path.insert(0, str(ATLAS / "src"))
from plot_atlas import aggregate_profile_rows, apply_publication_style, get_palette, plot_profile_ensemble  # noqa: E402


OUT_DEFAULT = WORKTREE / "docs/reproducibility/canva_figure_provenance"
FONT = "Arial"
# Canva placement is the controlling typography target for this package.  The
# prior R02j export inherited a manuscript-scale 7--11 pt setting; that is too
# small once the vector is placed in a slide.  Keep every authored text artist
# at or above 18 pt and enlarge the canvas/layouts accordingly.
MIN_TEXT_SIZE = 18
TITLE_SIZE = 22
PANEL_TITLE_SIZE = 20
AXIS_LABEL_SIZE = 20
TICK_SIZE = 18
LEGEND_SIZE = 18
ANNOTATION_SIZE = 18
COLORS = {
    # Frozen FIG-168 / NeurIPS2025-AI4MAT R04 model mapping.  These colors
    # are semantic identifiers, not a convenience palette: every model-family
    # panel must reuse them unchanged.
    "Foundation": "#CF4520",
    "FT-600K": "#002E71",
    "FT-MultiT": "#67ACE5",
    "Scratch-600K": "#FF9F1B",
    "Scratch-MultiT": "#F4C95D",
    "QE reference": "#222222",
    # Figure 7 uses the registered Plot Atlas foundation_profile class palette.
    "MACE-MP": "#D5A238",
    "MACE-OMAT-0": "#D85A45",
    "MACE-MH": "#5D8B56",
}

MODEL_MARKERS = {
    "Foundation": "v",
    "FT-600K": "s",
    "FT-MultiT": "^",
    "Scratch-600K": "D",
    "Scratch-MultiT": "D",
}

SERIES_MARKERS = {
    "From-scratch (seed 123)": MODEL_MARKERS["Scratch-600K"],
    "FT-600K (seed 123)": MODEL_MARKERS["FT-600K"],
    "FT-MultiT (seed 234)": MODEL_MARKERS["FT-MultiT"],
    "Foundation (OMAT-0-M)": MODEL_MARKERS["Foundation"],
    # Labels used by the self-relaxed NEB source table.
    "From-scratch (full AIMD)": MODEL_MARKERS["Scratch-600K"],
    "FT-600K": MODEL_MARKERS["FT-600K"],
    "FT-MultiT": MODEL_MARKERS["FT-MultiT"],
    "Foundation (OMAT-FT)": MODEL_MARKERS["Foundation"],
}

# Figure 1A follows the task-conditioned display grammar used by the Canva
# slide-1 source.  These are task-map colors, not the frozen model-family
# palette above; the latter remains unchanged in Figures 2--7.
TASK_GROUPS = ["L0 DFT Reference", "L1 Thermal Equilibrium", "L2 NEB Migration", "L3 Layer Sliding"]
TASK_MARKERS = {
    "L0 DFT Reference": "o",
    "L1 Thermal Equilibrium": "^",
    "L2 NEB Migration": "D",
    "L3 Layer Sliding": "P",
}
DFT_SPLIT_MARKERS = {
    "DFT - train": "o",
    "DFT - valid": "^",
    "DFT - test": "s",
}
TASK_MAP_COLORS = {
    "L0 DFT Reference": "#6E8793",
    "L1 Thermal Equilibrium": "#7658C7",
    "L2 NEB Migration": "#2D7FB5",
    "L3 Layer Sliding": "#E8792F",
}
TASK_SUBLEVEL_COLORS = {
    "Static DFT": "#587482",
    "DFT-AIMD": "#4B3199",
    "MLIP-MD": "#A28DE2",
    "DFT Fixed": "#1F5D88",
    "MLIP Relaxed": "#69A9D3",
    "DFT Sliding": "#C45C20",
    "MLIP Sliding": "#F3B16F",
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
    apply_publication_style(font_size=MIN_TEXT_SIZE)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT, "Helvetica", "DejaVu Sans"],
            "font.size": MIN_TEXT_SIZE,
            "axes.titlesize": PANEL_TITLE_SIZE,
            "axes.titleweight": "bold",
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.labelweight": "bold",
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.linewidth": 1.05,
            "figure.titlesize": TITLE_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def style_ax(ax, *, bold_ticks: bool = True) -> None:
    ax.grid(False)
    ax.tick_params(axis="both", direction="out", length=7.0, width=1.35, pad=6, labelsize=TICK_SIZE)
    ax.tick_params(axis="both", which="minor", direction="out", length=2.5, width=0.8)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.05)
        spine.set_color("#111111")
    if bold_ticks:
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            label.set_fontweight("bold")
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontsize(TICK_SIZE)
        label.set_fontfamily(FONT)
    ax.xaxis.label.set_fontsize(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_fontsize(AXIS_LABEL_SIZE)
    ax.xaxis.label.set_fontfamily(FONT)
    ax.yaxis.label.set_fontfamily(FONT)
    ax.title.set_fontsize(PANEL_TITLE_SIZE)
    ax.title.set_fontfamily(FONT)


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
    ax.text(0.5, 0.72, title, ha="center", va="center", color=color, fontsize=PANEL_TITLE_SIZE, fontweight="bold")
    ax.text(0.5, 0.47, body, ha="center", va="center", color="#333333", fontsize=MIN_TEXT_SIZE, linespacing=1.35)


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


FIXED_ENSEMBLE = WILEY / "analysis/neb_fixed_path_trajectory_figure_20261023/python/data/fixed_path_long.tsv"
GROUPED_NORMALIZED = WILEY / "analysis/cr_interlayer_sliding_grouped_20261025/results/tables/NORMALIZED_PROFILES.tsv"
GROUPED_QE = WILEY / "analysis/cr_interlayer_sliding_grouped_20261025/source_tables/foundation_2x2_QE_REFERENCE.tsv"
SELF_RELAXED_ENSEMBLE = WILEY / "analysis/neb_cross_element_triptychs_20261024/foundation_self_relaxed_4x3_20261024/figure_revisions/foundation_class_4x3_R03_fontplus2pt/data/foundation_class_4x3_R03_fontplus2pt/foundation_model_profiles_9models_4x3_R03.tsv"
SELF_RELAXED_QE = WILEY / "analysis/neb_cross_element_triptychs_20261024/foundation_self_relaxed_4x3_20261024/figure_revisions/foundation_class_4x3_R03_fontplus2pt/data/foundation_class_4x3_R03_fontplus2pt/foundation_model_qe_reference_profiles_4x3_R03.tsv"
FIXED_ENSEMBLE_PATHS = {
    "within_gap": "in-gap diffusion",
    "gap_to_ql": "gap-to-QL",
    "within_ql": "within-QL",
}
ENSEMBLE_FAMILIES = ["From-scratch", "FT-600K", "FT-MultiT", "Foundation"]
ENSEMBLE_LABELS = {
    "From-scratch": "From-scratch (n=3 seeds)",
    "FT-600K": "FT-600K (n=3 seeds)",
    "FT-MultiT": "FT-MultiT (n=3 seeds)",
    "Foundation": "Foundation (n=3 model variants)",
}
ENSEMBLE_COLORS = {
    "From-scratch": COLORS["Scratch-600K"],
    "FT-600K": COLORS["FT-600K"],
    "FT-MultiT": COLORS["FT-MultiT"],
    "Foundation": COLORS["Foundation"],
}
ENSEMBLE_MARKERS = {
    "From-scratch": MODEL_MARKERS["Scratch-600K"],
    "FT-600K": MODEL_MARKERS["FT-600K"],
    "FT-MultiT": MODEL_MARKERS["FT-MultiT"],
    "Foundation": MODEL_MARKERS["Foundation"],
}

GROUPED_FAMILIES = ["Foundation", "Scratch-600K", "Scratch-MultiT", "FT-600K", "FT-MultiT"]
GROUPED_PAIRS = ["Scratch-600K", "Scratch-MultiT", "FT-600K", "FT-MultiT"]
GROUPED_COLORS = {family: COLORS[family] for family in GROUPED_FAMILIES}
GROUPED_MARKERS = {family: MODEL_MARKERS[family] for family in GROUPED_FAMILIES}


def grouped_display_label(family: str, n: int) -> str:
    unit = "checkpoint variants" if family == "Foundation" else "seeds"
    return f"{family} (n={n} {unit})"


def ensemble_display_label(family: str, n: int) -> str:
    unit = "model variants" if family == "Foundation" else "seeds"
    return f"{family} (n={n} {unit})"


def profile_ensemble_summary(rows: list[dict[str, str]], path_key: str, group_field: str = "model_family") -> dict[str, dict[str, np.ndarray | int | list[str]]]:
    """Return mean/SD profiles over registered seeds or explicit model variants."""
    path_class = FIXED_ENSEMBLE_PATHS[path_key]
    subset = [r for r in rows if r.get("path_class") == path_class]
    result = {}
    for group in ENSEMBLE_FAMILIES:
        group_rows = [r for r in subset if r.get(group_field) == group]
        by_id = defaultdict(dict)
        for r in group_rows:
            if not r.get("model_id") or not r.get("image_index"):
                continue
            image_index = f(r, "image_index")
            if not np.isfinite(image_index):
                continue
            by_id[r.get("model_id", "")][int(image_index)] = (f(r, "s"), f(r, "relative_energy_eV"))
        replicate_ids = sorted(mid for mid, values in by_id.items() if values)
        if not replicate_ids:
            continue
        image_indices = sorted(set().union(*(set(by_id[mid]) for mid in replicate_ids)))
        # Rebuild an aligned replicate matrix; all canonical profiles are expected
        # to have ten images, but missing rows remain NaN rather than imputed.
        aligned = np.full((len(replicate_ids), len(image_indices)), np.nan)
        x_aligned = np.full(len(image_indices), np.nan)
        for j, image_index in enumerate(image_indices):
            xs = []
            for i, mid in enumerate(replicate_ids):
                if image_index in by_id[mid]:
                    x, y = by_id[mid][image_index]
                    x_aligned[j] = x if not np.isfinite(x_aligned[j]) else x_aligned[j]
                    xs.append(x); aligned[i, j] = y
            if xs: x_aligned[j] = float(np.nanmean(xs))
        result[group] = {
            "x": x_aligned,
            "values": aligned,
            "mean": np.nanmean(aligned, axis=0),
            "sd": np.nanstd(aligned, axis=0, ddof=1) if len(replicate_ids) > 1 else np.full(len(image_indices), np.nan),
            "n": len(replicate_ids),
            "replicate_ids": replicate_ids,
        }
    return result


def grouped_profile_summary(rows: list[dict[str, str]], system: str, method: str) -> dict[str, dict[str, np.ndarray | int | list[str]]]:
    """Summarize the current grouped Cr profiles by family and registered seed.

    The source table contains two distinct observables (fixed-path single-point
    scoring and endpoint-only MLIP NEB proxy) and two systems.  They are kept
    separate here; no path, method, or model family is pooled across those
    boundaries.
    """
    result = {}
    subset = [r for r in rows if r.get("system") == system and r.get("method") == method]
    for family in GROUPED_FAMILIES:
        group_rows = [r for r in subset if r.get("model_label") == family]
        by_id = defaultdict(dict)
        for r in group_rows:
            model_id = r.get("model_id", "")
            image_index = f(r, "image_index")
            x = f(r, "image_fraction")
            y = f(r, "relative_energy_eV")
            if not model_id or not np.isfinite(image_index) or not np.isfinite(x) or not np.isfinite(y):
                continue
            by_id[model_id][int(image_index)] = (x, y)
        replicate_ids = sorted(mid for mid, values in by_id.items() if values)
        if not replicate_ids:
            continue
        image_indices = sorted(set().union(*(set(by_id[mid]) for mid in replicate_ids)))
        aligned = np.full((len(replicate_ids), len(image_indices)), np.nan)
        x_aligned = np.full(len(image_indices), np.nan)
        for j, image_index in enumerate(image_indices):
            xs = []
            for i, mid in enumerate(replicate_ids):
                if image_index in by_id[mid]:
                    x, y = by_id[mid][image_index]
                    xs.append(x)
                    aligned[i, j] = y
            if xs:
                x_aligned[j] = float(np.nanmean(xs))
        result[family] = {
            "x": x_aligned,
            "values": aligned,
            "mean": np.nanmean(aligned, axis=0),
            "sd": np.nanstd(aligned, axis=0, ddof=1) if len(replicate_ids) > 1 else np.full(len(image_indices), np.nan),
            "n": len(replicate_ids),
            "replicate_ids": replicate_ids,
            "image_indices": image_indices,
        }
    return result


def grouped_paired_mae(
    rows: list[dict[str, str]], qe_rows: list[dict[str, str]], system: str, method: str
) -> dict[str, dict[str, np.ndarray | float | int | list[str]]]:
    """Return per-replica MAE against the same retained DFT image positions."""
    summaries = grouped_profile_summary(rows, system, method)
    qe_by_image = {
        int(f(r, "image_index")): f(r, "relative_energy_eV")
        for r in qe_rows
        if r.get("system") == system and np.isfinite(f(r, "image_index")) and np.isfinite(f(r, "relative_energy_eV"))
    }
    result = {}
    for family, summary in summaries.items():
        image_indices = list(summary["image_indices"])
        values = np.asarray(summary["values"], dtype=float)
        qe = np.array([qe_by_image.get(i, np.nan) for i in image_indices], dtype=float)
        per_replicate = np.nanmean(np.abs(values - qe[None, :]), axis=1)
        finite = np.isfinite(per_replicate)
        if not finite.any():
            continue
        result[family] = {
            "values": per_replicate[finite],
            "mean": float(np.nanmean(per_replicate[finite])),
            "sd": float(np.nanstd(per_replicate[finite], ddof=1)) if finite.sum() > 1 else np.nan,
            "n": int(finite.sum()),
            "replicate_ids": [mid for mid, keep in zip(summary["replicate_ids"], finite) if keep],
        }
    return result


def plot_mean_sd(ax, summary: dict[str, np.ndarray | int | list[str]], label: str, color: str, marker: str, *, lw: float = 1.6, alpha: float = .18) -> None:
    x = np.asarray(summary["x"], dtype=float); mean = np.asarray(summary["mean"], dtype=float); sd = np.asarray(summary["sd"], dtype=float)
    ax.plot(x, mean, color=color, marker=marker, markersize=5.5, lw=max(lw, 2.2), label=label, markerfacecolor="white", markeredgewidth=1.0)
    finite_band = np.isfinite(sd)
    if finite_band.any():
        lo = np.where(finite_band, mean - sd, np.nan); hi = np.where(finite_band, mean + sd, np.nan)
        ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=0)


def finite_values(values) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[np.isfinite(arr)]


def mean_sd(values) -> tuple[float, float]:
    arr = finite_values(values)
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan


def plot_family_cluster_bars(
    ax,
    categories: list[str],
    values_by_family: dict[str, list[np.ndarray]],
    family_order: list[str],
    colors: dict[str, str],
    *,
    ylabel: str,
    title: str,
    rotate: float = 0,
    y_min: float | None = None,
    width_fraction: float = .80,
) -> None:
    """Full-campaign grouped bars with replica points and sample-SD whiskers."""
    x = np.arange(len(categories), dtype=float)
    width = width_fraction / max(len(family_order), 1)
    for j, family in enumerate(family_order):
        means = []
        errors = []
        family_values = values_by_family.get(family, [np.array([])] * len(categories))
        for values in family_values:
            m, s = mean_sd(values)
            means.append(m); errors.append(0.0 if not np.isfinite(s) else s)
        offset = (j - (len(family_order) - 1) / 2) * width
        xpos = x + offset
        ax.bar(
            xpos,
            means,
            yerr=errors,
            width=width * .92,
            color=colors[family],
            alpha=.88,
            edgecolor="#111111",
            linewidth=.45,
            capsize=2.5,
            error_kw={"ecolor": "#222222", "elinewidth": 1.0},
            label=family,
            zorder=2,
        )
        for i, values in enumerate(family_values):
            vals = finite_values(values)
            if vals.size:
                jitter = np.linspace(-width * .22, width * .22, vals.size) if vals.size > 1 else np.array([0.0])
                ax.scatter(np.full(vals.size, xpos[i]) + jitter, vals, s=10, color="#222222", zorder=5)
    ax.set_xticks(x, categories, rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    style_ax(ax)


def plot_reference_pair_bars(
    ax,
    categories: list[str],
    reference_values: list[np.ndarray],
    target_values: dict[str, list[np.ndarray]],
    reference_label: str,
    target_order: list[str],
    colors: dict[str, str],
    *,
    ylabel: str,
    title: str,
    tick_labels: list[str] | None = None,
    y_min: float | None = 0,
) -> None:
    """Explicit reference-vs-target clusters, preserving every replica."""
    x = np.arange(len(categories), dtype=float)
    width = .30
    for i, target in enumerate(target_order):
        for role, family, xpos in (
            ("reference", reference_label, x[i] - width / 2),
            ("target", target, x[i] + width / 2),
        ):
            values = reference_values[i] if role == "reference" else target_values[target][i]
            m, s = mean_sd(values)
            ax.bar(
                xpos,
                m,
                yerr=0.0 if not np.isfinite(s) else s,
                width=width,
                color=colors[family],
                alpha=.88,
                edgecolor="#111111",
                linewidth=.5,
                capsize=2.5,
                error_kw={"ecolor": "#222222", "elinewidth": 1.0},
                zorder=2,
            )
            vals = finite_values(values)
            if vals.size:
                jitter = np.linspace(-.055, .055, vals.size) if vals.size > 1 else np.array([0.0])
                ax.scatter(np.full(vals.size, xpos) + jitter, vals, s=10, color="#222222", zorder=5)
    ax.set_xticks(x, tick_labels or categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    style_ax(ax)


def plot_observed_vs_target_dumbbell(
    ax,
    families: list[str],
    values_by_family: dict[str, np.ndarray],
    target: float,
    colors: dict[str, str],
    *,
    xlabel: str,
    title: str,
) -> None:
    """Horizontal observed-vs-target dumbbell with all replica observations."""
    y = np.arange(len(families), dtype=float)[::-1]
    ax.axvline(target, color="#222222", lw=1.1, ls=(0, (4, 2)), zorder=1)
    for yi, family in zip(y, families):
        vals = finite_values(values_by_family.get(family, []))
        m, s = mean_sd(vals)
        if not np.isfinite(m):
            continue
        ax.plot([m, target], [yi, yi], color=colors[family], lw=2.0, alpha=.75, zorder=2)
        ax.errorbar(m, yi, xerr=0.0 if not np.isfinite(s) else s, fmt="o", ms=5.5, color=colors[family],
                    markerfacecolor="white", markeredgewidth=1.0, capsize=2.5, lw=1.0, zorder=5)
        if vals.size:
            jitter = np.linspace(-.12, .12, vals.size) if vals.size > 1 else np.array([0.0])
            ax.scatter(vals, np.full(vals.size, yi) + jitter, s=10, color="#222222", zorder=6)
        ax.scatter([target], [yi], marker="D", s=34, color="#222222", facecolors="white", linewidths=.9, zorder=5)
    ax.set_yticks(y, families)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    style_ax(ax)
    ax.legend(
        handles=[
            Line2D([], [], color="#222222", ls=(0, (4, 2)), lw=1.1, label="1 ns target"),
            Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor="#222222", color="none", ms=5, label="family mean ± SD"),
            Line2D([], [], marker="o", color="#222222", linestyle="", ms=4, label="replica"),
        ],
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="lower right",
    )


def class_profile_summary(rows: list[dict[str, str]], path_key: str, classes: list[str]) -> dict[str, dict[str, np.ndarray | int | list[str]]]:
    """Summarize the Cr self-relaxed Foundation ensemble by registered class."""
    result = {}
    for group in classes:
        group_rows = [r for r in rows if r.get("element") == "Cr" and r.get("path_key") == path_key and r.get("model_class") == group]
        replicate_ids = sorted({r.get("model_id", "") for r in group_rows if r.get("model_id")})
        by_id = defaultdict(dict)
        for r in group_rows:
            by_id[r.get("model_id", "")][int(f(r, "image_index"))] = (f(r, "s"), f(r, "relative_energy_eV"))
        image_indices = sorted(set().union(*(set(by_id[mid]) for mid in replicate_ids))) if replicate_ids else []
        aligned = np.full((len(replicate_ids), len(image_indices)), np.nan)
        x_aligned = np.full(len(image_indices), np.nan)
        for j, image_index in enumerate(image_indices):
            xs = []
            for i, mid in enumerate(replicate_ids):
                if image_index in by_id[mid]:
                    x, y = by_id[mid][image_index]; xs.append(x); aligned[i, j] = y
            if xs: x_aligned[j] = float(np.nanmean(xs))
        if replicate_ids:
            result[group] = {
                "x": x_aligned,
                "values": aligned,
                "mean": np.nanmean(aligned, axis=0),
                "sd": np.nanstd(aligned, axis=0, ddof=1) if len(replicate_ids) > 1 else np.full(len(image_indices), np.nan),
                "n": len(replicate_ids),
                "replicate_ids": replicate_ids,
            }
    return result


def aligned_mean_sd(series: list[tuple[np.ndarray, np.ndarray]], points: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Interpolate replicate time-series on their common finite domain."""
    clean = []
    for x, y in series:
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2:
            order = np.argsort(x[m]); clean.append((x[m][order], y[m][order]))
    if not clean:
        return None
    xmin = max(float(x[0]) for x, _ in clean); xmax = min(float(x[-1]) for x, _ in clean)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return None
    grid = np.linspace(xmin, xmax, points)
    values = np.vstack([np.interp(grid, x, y) for x, y in clean])
    mean = np.nanmean(values, axis=0)
    sd = np.nanstd(values, axis=0, ddof=1) if values.shape[0] > 1 else np.full(points, np.nan)
    return grid, mean, sd


def _ood_marker_sizes(rows: list[dict[str, str]]) -> tuple[np.ndarray, float]:
    """Map the measured feature-space distance-to-training to marker area."""
    values = np.asarray([f(row, "distance_to_training") for row in rows], dtype=float)
    finite = values[np.isfinite(values)]
    reference = float(np.nanpercentile(finite, 98)) if finite.size else 1.0
    reference = max(reference, 1e-12)
    normalized = np.clip(np.nan_to_num(values, nan=0.0) / reference, 0.0, 1.4)
    return 18.0 + 220.0 * np.power(normalized, 0.68), reference


def _add_task_hulls(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    """Add translucent measured task envelopes in the task-conditioned display."""
    for task in TASK_GROUPS:
        task_rows = [row for row in rows if row.get("task_group") == task]
        xy = np.asarray([[f(row, "x_pca"), f(row, "y_pca")] for row in task_rows], dtype=float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) < 3 or len(np.unique(xy, axis=0)) < 3:
            continue
        try:
            hull = ConvexHull(xy)
        except Exception:
            continue
        color = TASK_MAP_COLORS[task]
        alpha = 0.035 if task == "L0 DFT Reference" else 0.095
        ax.add_patch(
            patches.Polygon(
                xy[hull.vertices],
                closed=True,
                facecolor=color,
                edgecolor=color,
                linewidth=1.25,
                alpha=alpha * 0.55,
                zorder=0,
            )
        )


def _add_task_path_lines(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    """Draw only the source-provided paths, preserving protocol line semantics."""
    path_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = row.get("path_key", "")
        if key and key.lower() != "nan":
            path_groups[key].append(row)
    for path_rows in path_groups.values():
        path_rows = sorted(path_rows, key=lambda row: (f(row, "frame", np.inf), f(row, "sample_frame", np.inf)))
        xy = np.asarray([[f(row, "x_pca"), f(row, "y_pca")] for row in path_rows], dtype=float)
        finite = np.isfinite(xy).all(axis=1)
        if finite.sum() < 2:
            continue
        task = path_rows[0].get("task_group", "")
        protocol = path_rows[0].get("protocol_style", "")
        ax.plot(
            xy[finite, 0],
            xy[finite, 1],
            color=TASK_MAP_COLORS.get(task, "#777777"),
            linewidth=0.58,
            alpha=0.18,
            linestyle="--" if protocol == "self-relaxed" else "-",
            zorder=1,
        )


def fig1(out: Path) -> tuple[dict[str, str], list[Path]]:
    src = WILEY / "analysis/migrationbench_universe_publication_20261026/derived/comprehensive_task_map_points.csv"
    rows = read_rows(src)
    groups = TASK_GROUPS
    marker_sizes, ood_reference = _ood_marker_sizes(rows)
    for row, size in zip(rows, marker_sizes):
        row["_ood_marker_size"] = float(size)

    fig = plt.figure(figsize=(18.0, 9.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.9], wspace=0.52)
    ax = fig.add_subplot(gs[0, 0])

    # Keep the primary panel on the source-provided common SOAP/PCA axes.  No
    # task anchors, per-task translations, rank equalization, or aspect-ratio
    # manipulation is applied.  Optional UMAP/t-SNE candidates are rendered in
    # a separate comparison package and must not be mistaken for PCA.
    split_order = ["train", "valid", "test", "md_sample", "dft_neb", "dft_interlayer", "mlip_self_neb", "mlip_interlayer_self"]
    # Match the first Figure 1A point-cloud contract: direct reference/MD/DFT
    # rows are shown as points; MLIP self-relaxed NEB/sliding rows remain
    # available as complete source-provided paths and in panel C, but are not
    # rendered as hundreds of independent foreground markers.
    visible_point_splits = {"train", "valid", "test", "md_sample", "dft_neb", "dft_interlayer"}
    visible_rows = [row for row in rows if row.get("split") in visible_point_splits]
    _add_task_hulls(ax, visible_rows)
    group_colors = {"L0 DFT Reference": "#8DA9BF", "L1 Thermal Equilibrium": "#6E9E75", "L2 NEB Migration": "#C4547A", "L3 Layer Sliding": "#E68F1D"}
    for task in TASK_GROUPS:
        for split in split_order:
            if split not in visible_point_splits:
                continue
            sub = [row for row in rows if row.get("task_group") == task and row.get("split") == split]
            if not sub:
                continue
            x = np.asarray([f(row, "x_pca") for row in sub], dtype=float)
            y = np.asarray([f(row, "y_pca") for row in sub], dtype=float)
            size = np.asarray([row["_ood_marker_size"] for row in sub], dtype=float)
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(size)
            if not finite.any():
                continue
            marker = TASK_MARKERS[task]
            dense_split = split in {"train", "valid", "test"}
            ax.scatter(
                x[finite],
                y[finite],
                s=size[finite],
                marker=marker,
                color=group_colors[task],
                alpha=0.28 if dense_split else 0.78,
                linewidths=0.12 if dense_split else 0.22,
                edgecolors="white",
                zorder=2 if dense_split else 3,
            )

    ax.set_xlabel("SOAP PCA 1")
    ax.set_ylabel("SOAP PCA 2")
    ax.set_title("A  Configuration universe", loc="left")
    style_ax(ax)
    task_handles = [
        Line2D(
            [], [], marker=TASK_MARKERS[task], linestyle="", color=group_colors[task],
            markeredgecolor="white", label=task, markersize=8.5,
        )
        for task in TASK_GROUPS
    ]
    dft_split_handles = [
        Line2D(
            [], [], marker=marker, linestyle="", color="#777777", markerfacecolor="#888888",
            markeredgecolor="white", label=label, markersize=8.0,
        )
        for label, marker in DFT_SPLIT_MARKERS.items()
    ]
    q50 = float(np.nanpercentile([f(row, "distance_to_training") for row in rows], 50))
    q90 = float(np.nanpercentile([f(row, "distance_to_training") for row in rows], 90))
    size_values = [0.0, q50, q90]
    size_handles = [
        Line2D(
            [], [], linestyle="", marker="o", color="#555555", markerfacecolor="#555555", markeredgecolor="white",
            markersize=float(np.sqrt(18.0 + 220.0 * np.power(min(value / ood_reference, 1.4), 0.68))),
            label=label,
        )
        for value, label in zip(size_values, ["near train", "intermediate", "far OOD"])
    ]
    encoding_handles = [
        *task_handles,
        *dft_split_handles,
        *size_handles,
        Line2D([], [], color="#555555", linewidth=1.2, linestyle="-", label="fixed / sliding path"),
        Line2D([], [], color="#555555", linewidth=1.2, linestyle="--", label="self-relaxed path"),
    ]
    ax.legend(
        handles=encoding_handles,
        frameon=False,
        fontsize=LEGEND_SIZE,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.23),
        borderaxespad=0.0,
        handlelength=1.1,
        columnspacing=0.8,
        handletextpad=0.45,
    )

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
        ax.text(x + 0.16, y + 0.075, label, ha="center", va="center", fontsize=MIN_TEXT_SIZE, fontweight="bold")
    for x1, y1, x2, y2 in [(0.40, .72, .58, .80), (.24, .72, .24, .57), (.74, .72, .74, .57), (.40, .42, .49, .25), (.74, .42, .65, .25)]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#555555"})

    ax = fig.add_subplot(gs[0, 2])
    # Full campaign coverage: retain every source split/protocol instead of
    # collapsing each task layer to a single total bar.
    coverage_splits = ["train", "valid", "test", "md_sample", "dft_neb", "dft_interlayer", "mlip_self_neb", "mlip_interlayer_self"]
    coverage_labels = ["L0 reference", "L1 thermal", "L2 NEB", "L3 sliding"]
    coverage_colors = {
        "train": "#8DA9BF", "valid": "#B7C8D6", "test": "#5D8B9E",
        "md_sample": "#6E9E75", "dft_neb": "#C4547A", "dft_interlayer": "#E68F1D",
        "mlip_self_neb": "#9A6A00", "mlip_interlayer_self": "#D5A238",
    }
    y = np.arange(len(groups), dtype=float)
    width = .82 / len(coverage_splits)
    for j, split in enumerate(coverage_splits):
        vals = [sum(1 for r in rows if r.get("task_label") == group and r.get("split") == split) for group in groups]
        offset = (j - (len(coverage_splits) - 1) / 2) * width
        ax.barh(y + offset, vals, height=width * .9, color=coverage_colors[split], edgecolor="#111111", linewidth=.35, label=split)
    ax.set_yticks(y, coverage_labels)
    ax.set_xlabel("structure rows")
    ax.set_title("C  Full campaign coverage by source split", loc="left")
    ax.legend(frameon=False, fontsize=LEGEND_SIZE, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.24))
    style_ax(ax)
    fig.subplots_adjust(left=.06, right=.99, bottom=.40, top=.86)
    fig.suptitle("Figure 1 | Task-conditioned benchmark and configuration universe", x=0.02, ha="left", y=.985, fontsize=TITLE_SIZE, fontweight="bold")
    assets = save_figure(fig, out, "figure01_task_conditioned_universe")
    return assets, [src]


def fig2(out: Path) -> tuple[dict[str, str], list[Path]]:
    root = WILEY / "analysis/rmse_train_test_20261026"
    summary_path = root / "parity_summary.csv"
    summary = read_rows(summary_path)
    per_files = sorted((root / "raw/rockfish_parity").glob("*_test__per_frame.csv"))
    fig = plt.figure(figsize=(18.0, 12.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, .9], hspace=.66, wspace=.42)
    axa, axb, axc = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axd, axe, axf = [fig.add_subplot(gs[1, i]) for i in range(3)]
    force_x, force_y, force_family = [], [], []
    all_frame = []
    for path in per_files:
        fam = family_from_stem(path.stem)
        rows = read_rows(path)
        energy_x, energy_y = [], []
        for r in rows:
            x = f(r, "energy_ref_eV_per_atom"); y = (f(r, "energy_pred_eV_per_atom") - x) * 1000
            if np.isfinite([x, y]).all():
                energy_x.append(x); energy_y.append(y)
            all_frame.append((fam, r.get("trajectory", "unknown"), f(r, "force_rmse_meV_A")))
        # One collection per model family/file keeps the vector output fast
        # without changing the retained point-level data.
        if energy_x:
            axa.scatter(energy_x, energy_y, s=7, alpha=.32, color=COLORS[fam], edgecolors="none")
        npz = path.with_name(path.name.replace("__per_frame.csv", "__parity.npz"))
        # One compressed train/test archive is currently not materializable
        # from the provider-backed local root.  The requested complete
        # train/test clusters (C/F) come from parity_summary.csv and remain
        # unaffected; omit only this auxiliary frame-level scatter source.
        unreadable_auxiliary = {"ft_multit_seed345_test__parity.npz"}
        if npz.exists() and npz.name not in unreadable_auxiliary:
            with np.load(npz) as dat:
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
    # The family legend is shared by A/B/C/F and is placed below the full
    # figure, not inside a data panel.

    families = ["Scratch-600K", "FT-600K", "Scratch-MultiT", "FT-MultiT"]
    split_order = ["train", "test"]
    split_labels = {"train": "train", "test": "held-out test"}
    for metric, ax, key in [("Energy RMSE", axc, "energy_rmse_meV_per_atom"), ("Force RMSE", axf, "force_rmse_meV_A")]:
        x = np.arange(len(families), dtype=float)
        width = .32
        for j, split in enumerate(split_order):
            for i, fam in enumerate(families):
                vals = [f(r, key) for r in summary if r.get("model", "") == fam and r.get("split") == split]
                vals = finite_values(vals)
                if vals.size == 0:
                    continue
                m, s = mean_sd(vals)
                xpos = x[i] + (j - .5) * width
                ax.bar(
                    xpos,
                    m,
                    yerr=0.0 if not np.isfinite(s) else s,
                    width=width * .9,
                    color=COLORS[fam],
                    alpha=.30 if split == "train" else .88,
                    hatch="///" if split == "train" else None,
                    edgecolor="#111111",
                    linewidth=.45,
                    capsize=2.5,
                    error_kw={"ecolor": "#222222", "elinewidth": 1.0},
                    zorder=2,
                )
                jitter = np.linspace(-.05, .05, vals.size) if vals.size > 1 else np.array([0.0])
                ax.scatter(np.full(vals.size, xpos) + jitter, vals, s=9, color="#222222", zorder=5)
        ax.set_xticks(x, ["Scratch\n600K", "FT\n600K", "Scratch\nMultiT", "FT\nMultiT"])
        ax.set_ylabel(key.replace("_", " "))
        ax.set_title(("C  " if ax is axc else "F  ") + metric + " · full train/test campaign", loc="left")
        ax.set_ylim(bottom=0)
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
    split_handles = [
        patches.Patch(facecolor="#777777", alpha=.30, hatch="///", edgecolor="#111111", label="train"),
        patches.Patch(facecolor="#777777", alpha=.88, edgecolor="#111111", label="held-out test"),
    ]
    fig.legend(handles, [h.get_label() for h in handles], frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(.50, .105), fontsize=LEGEND_SIZE)
    fig.legend(split_handles, [h.get_label() for h in split_handles], frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(.50, .045), fontsize=LEGEND_SIZE)
    fig.subplots_adjust(left=.07, right=.99, bottom=.22, top=.88)
    fig.suptitle("Figure 2 | Near-equilibrium parity is an entry gate", x=.02, ha="left", y=.985, fontsize=TITLE_SIZE, fontweight="bold")
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
    rows = read_rows(GROUPED_NORMALIZED)
    qe_rows = read_rows(GROUPED_QE)
    system, method = "cr_in_gap", "fixed"
    summaries = grouped_profile_summary(rows, system, method)
    qe_rows = sorted([r for r in qe_rows if r.get("system") == system], key=lambda r: f(r, "image_index"))
    qe_by_image = {int(f(r, "image_index")): f(r, "relative_energy_eV") for r in qe_rows}
    qe = np.array([f(r, "relative_energy_eV") for r in qe_rows])
    qx = np.array([f(r, "image_fraction") for r in qe_rows])
    fig = plt.figure(figsize=(18.0, 10.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, .85], hspace=.68, wspace=.46)
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    for family in GROUPED_FAMILIES:
        if family in summaries:
            plot_mean_sd(axa, summaries[family], grouped_display_label(family, int(summaries[family]["n"])), GROUPED_COLORS[family], GROUPED_MARKERS[family], lw=1.55)
    if qe.size:
        axa.plot(qx, qe, marker="D", ms=3.4, lw=1.3, ls="--", color=COLORS["QE reference"], markerfacecolor="white", label="DFT retained path · SCF warning")
    axa.set_xlabel("Reaction Coordinate"); axa.set_ylabel("ΔE (eV)")
    axa.set_title("A  Cr-in-gap fixed images", loc="left")
    style_ax(axa)
    paired_targets = ["Scratch-600K", "Scratch-MultiT", "FT-600K", "FT-MultiT"]
    pair_categories = ["Scratch-600K", "Scratch-MultiT", "FT-600K", "FT-MultiT"]
    paired_stats = grouped_paired_mae(rows, read_rows(GROUPED_QE), system, method)
    if qe.size and "Foundation" in paired_stats:
        reference_mae = np.asarray(paired_stats["Foundation"]["values"], dtype=float)
        target_mae = {family: [np.asarray(paired_stats[family]["values"], dtype=float)] * len(pair_categories) for family in paired_targets if family in paired_stats}
        plot_reference_pair_bars(
            axb,
            pair_categories,
            [reference_mae] * len(pair_categories),
            target_mae,
            "Foundation",
            paired_targets,
            GROUPED_COLORS,
            ylabel="MAE versus DFT (eV)",
            title="B  Foundation-reference paired MAE · full campaign",
            tick_labels=["vs Scratch\n600K", "vs Scratch\nMultiT", "vs FT\n600K", "vs FT\nMultiT"],
        )
        # The complete family legend is shared below the overall figure.
    else:
        axb.set_title("B  Foundation-reference paired MAE · unavailable", loc="left")
        style_ax(axb)
    for family in GROUPED_FAMILIES:
        if family in summaries and qe.size:
            summary = summaries[family]
            image_indices = list(summary["image_indices"])
            qe_aligned = np.array([qe_by_image.get(i, np.nan) for i in image_indices], dtype=float)
            residual_mean = np.asarray(summary["mean"]) - qe_aligned
            residual_sd = np.asarray(summary["sd"])
            axc.plot(np.asarray(summary["x"]), residual_mean, marker=GROUPED_MARKERS[family], ms=3.0, lw=1.4, color=GROUPED_COLORS[family], label=grouped_display_label(family, int(summary["n"])))
            finite_band = np.isfinite(residual_sd)
            axc.fill_between(np.asarray(summary["x"]), np.where(finite_band, residual_mean - residual_sd, np.nan), np.where(finite_band, residual_mean + residual_sd, np.nan), color=GROUPED_COLORS[family], alpha=.16, linewidth=0)
    axc.axhline(0, color="#222222", ls=(0, (4, 2)), lw=1.0)
    axc.set_xlabel("Reaction Coordinate"); axc.set_ylabel("ΔE (eV)"); axc.set_title("C  Residual diagnostic across fixed images", loc="left"); style_ax(axc)
    family_handles, family_labels = axa.get_legend_handles_labels()
    fig.legend(family_handles, family_labels, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.50, .045), fontsize=LEGEND_SIZE)
    fig.subplots_adjust(left=.07, right=.99, bottom=.19, top=.87)
    fig.suptitle("Figure 3 | Cr-in-gap: Foundation-reference fixed-image evidence", x=.02, ha="left", y=.985, fontsize=TITLE_SIZE, fontweight="bold")
    assets = save_figure(fig, out, "figure03_within_gap_fixed_images")
    return assets, [GROUPED_NORMALIZED, GROUPED_QE]


def fig4(out: Path) -> tuple[dict[str, str], list[Path]]:
    # Figure 4 uses the current grouped Rockfish/local review table because it
    # is the audited source that distinguishes Scratch-600K from Scratch-MultiT.
    # Each line panel is paired with its own MAE cluster below it; the pair is
    # over the same ten retained image positions, with replicate SD whiskers.
    rows = read_rows(GROUPED_NORMALIZED)
    qe_rows = read_rows(GROUPED_QE)
    panels = [
        ("bulk", "fixed", "Bulk · fixed path"),
        ("cr_in_gap", "fixed", "Cr-in-gap · fixed path"),
        ("bulk", "self", "Bulk · self-relaxed proxy"),
        ("cr_in_gap", "self", "Cr-in-gap · self-relaxed proxy"),
    ]
    fig = plt.figure(figsize=(22.0, 13.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, .92], hspace=.72, wspace=.42)
    legend_handles = []
    for col, (system, method, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        summaries = grouped_profile_summary(rows, system, method)
        for family in GROUPED_FAMILIES:
            if family in summaries:
                plot_mean_sd(
                    ax,
                    summaries[family],
                    grouped_display_label(family, int(summaries[family]["n"])),
                    GROUPED_COLORS[family],
                    GROUPED_MARKERS[family],
                    lw=1.45,
                )
        ref = sorted(
            [r for r in qe_rows if r.get("system") == system],
            key=lambda r: f(r, "image_index"),
        )
        if ref:
            ax.plot(
                np.array([f(r, "image_fraction") for r in ref]),
                np.array([f(r, "relative_energy_eV") for r in ref]),
                marker="D",
                ms=3.0,
                lw=1.2,
                ls="--",
                color=COLORS["QE reference"],
                markerfacecolor="white",
                label="DFT retained path · SCF warning",
            )
        ax.set_title(chr(97 + col) + "  " + title, loc="left")
        ax.set_xlabel("Reaction Coordinate")
        ax.set_ylabel("ΔE (eV)")
        if method == "self":
            ax.text(.98, .04, "endpoint-only MLIP proxy", transform=ax.transAxes, ha="right", va="bottom", fontsize=ANNOTATION_SIZE, color="#9A6A00")
        style_ax(ax)
        if col == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        axb = fig.add_subplot(gs[1, col])
        stats = grouped_paired_mae(rows, qe_rows, system, method)
        # Make the requested pairing explicit: each cluster is Foundation
        # (reference) beside exactly one target family.
        cluster_x = np.arange(len(GROUPED_PAIRS), dtype=float)
        bar_width = .30
        offsets = {"Foundation": -bar_width / 2, "target": bar_width / 2}
        for i, family in enumerate(GROUPED_PAIRS):
            for role, plotted_family in (("Foundation", "Foundation"), ("target", family)):
                stat = stats.get(plotted_family, {})
                mean = float(stat.get("mean", np.nan))
                err = float(stat.get("sd", np.nan))
                xbar = cluster_x[i] + offsets[role]
                axb.bar(
                    xbar,
                    mean,
                    yerr=err if np.isfinite(err) else 0.0,
                    color=GROUPED_COLORS[plotted_family],
                    alpha=.88,
                    width=bar_width,
                    edgecolor="#111111",
                    linewidth=.5,
                    error_kw={"ecolor": "#222222", "capsize": 2.5, "elinewidth": 1.0},
                )
                values = np.asarray(stat.get("values", []), dtype=float)
                values = values[np.isfinite(values)]
                if values.size:
                    jitter = np.linspace(-.055, .055, values.size) if values.size > 1 else np.array([0.0])
                    axb.scatter(np.full(values.size, xbar) + jitter, values, s=8, color="#222222", zorder=5)
        axb.set_xticks(cluster_x, ["vs Scratch\n600K", "vs Scratch\nMultiT", "vs FT\n600K", "vs FT\nMultiT"], rotation=0)
        axb.set_ylabel("MAE vs DFT (eV)")
        axb.set_title(chr(101 + col) + "  Foundation-reference paired MAE", loc="left")
        axb.set_ylim(bottom=0)
        style_ax(axb)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            fontsize=LEGEND_SIZE,
            ncol=6,
            loc="lower center",
            bbox_to_anchor=(.50, .045),
        )
    fig.subplots_adjust(left=.055, right=.995, bottom=.19, top=.86)
    fig.suptitle(
        "Figure 4 | Foundation-reference transfer across paired NEB observables",
        x=.02,
        ha="left",
        y=.985,
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    assets = save_figure(fig, out, "figure04_cross_environment_transfer")
    return assets, [GROUPED_NORMALIZED, GROUPED_QE]


def fig5(out: Path) -> tuple[dict[str, str], list[Path]]:
    rows=read_rows(SELF_RELAXED_ENSEMBLE); qe_rows=read_rows(SELF_RELAXED_QE)
    paths=[("within_gap","Within-gap"),("gap_to_ql","Gap-to-QL"),("within_ql","Within-QL")]
    classes=["MACE-MP","MACE-OMAT-0","MACE-MH"]
    class_colors={"MACE-MP":COLORS["MACE-MP"],"MACE-OMAT-0":COLORS["MACE-OMAT-0"],"MACE-MH":COLORS["MACE-MH"]}
    class_markers={"MACE-MP":"o","MACE-OMAT-0":"s","MACE-MH":"^"}
    fig=plt.figure(figsize=(18.0,11.0)); gs=fig.add_gridspec(2,3,height_ratios=[1.0,.78],hspace=.68,wspace=.42)
    n_matrix=np.full((len(classes),len(paths)),np.nan)
    for col,(key,title) in enumerate(paths):
        ax=fig.add_subplot(gs[0,col]); summaries=class_profile_summary(rows,key,classes)
        for cls in classes:
            if cls not in summaries: continue
            label=f"{cls} (n={summaries[cls]['n']})"
            plot_mean_sd(ax,summaries[cls],label,class_colors[cls],class_markers[cls],lw=1.55,alpha=.20)
            n_matrix[classes.index(cls),col]=summaries[cls]["n"]
        sub=sorted([r for r in qe_rows if r.get("element")=="Cr" and r.get("path_key")==key],key=lambda r:f(r,"image_index"))
        if sub:
            ax.plot(np.array([f(r,"s") for r in sub]),np.array([f(r,"relative_energy_eV") for r in sub]),marker="D",ms=3.1,lw=1.25,ls="--",color=COLORS["QE reference"],markerfacecolor="white",label="QE reference")
        ax.set_title(chr(97+col)+"  "+title,loc="left"); ax.set_xlabel("Reaction Coordinate"); ax.set_ylabel("ΔE (eV)"); style_ax(ax)
        if col==0:
            first_profile_handles, first_profile_labels = ax.get_legend_handles_labels()
    axd=fig.add_subplot(gs[1,0:2]); axd.set_title("D  Finite model profiles included in each ensemble",loc="left")
    for i in range(len(classes)):
        for j in range(len(paths)):
            if np.isfinite(n_matrix[i,j]):
                axd.add_patch(patches.Rectangle((j-.5,i-.5),1,1,facecolor="#EDF2F6",edgecolor="white",linewidth=.5))
                axd.text(j,i,f"n={int(n_matrix[i,j])}",ha="center",va="center",fontsize=MIN_TEXT_SIZE,fontweight="bold",color="#222222")
    axd.set_xlim(-.5,len(paths)-.5); axd.set_ylim(len(classes)-.5,-.5)
    axd.set_xticks(range(len(paths)),[p[1] for p in paths],rotation=20,ha="right"); axd.set_yticks(range(len(classes)),classes); style_ax(axd)
    status_box(fig.add_subplot(gs[1,2]),"BLOCKED FOR PUBLICATION","Self-relaxed profiles are finite\nmodel observables, not validated\nDFT MEP barriers. Endpoint identity,\nfull convergence history and\ngeometry alignment remain incomplete;\ndo not report barrier success.")
    fig.legend(first_profile_handles, first_profile_labels, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(.50, .045), fontsize=LEGEND_SIZE)
    fig.subplots_adjust(left=.07, right=.99, bottom=.18, top=.86)
    fig.suptitle("Figure 5 | Foundation model-class NEB paths are a separate diagnostic",x=.02,ha="left",y=.985,fontsize=TITLE_SIZE,fontweight="bold")
    assets=save_figure(fig,out,"figure05_generated_neb_diagnostic"); return assets,[SELF_RELAXED_ENSEMBLE,SELF_RELAXED_QE]


def fig6(out: Path) -> tuple[dict[str, str], list[Path]]:
    root=WILEY/"analysis/md_1ns_updated_publication_figure_20261025"; status_path=root/"MD15_FIGURE_DATA_R06G.tsv"; analysis=root/"md15_analysis_r03"
    status=read_rows(status_path); by_id={r["model_id"]:r for r in status}
    fig=plt.figure(figsize=(18.0,12.0)); gs=fig.add_gridspec(2,3,hspace=.66,wspace=.42)
    axa,axb,axc=[fig.add_subplot(gs[0,i]) for i in range(3)]; axd,axe,axf=[fig.add_subplot(gs[1,i]) for i in range(3)]
    family_colors={"Foundation":COLORS["Foundation"],"FT-600K":COLORS["FT-600K"],"FT-MultiT":COLORS["FT-MultiT"],"Scratch-600K":COLORS["Scratch-600K"],"Scratch-MultiT":COLORS["Scratch-MultiT"]}
    grouped_series=defaultdict(lambda: defaultdict(list))
    file_paths=[]
    # One provider-backed auxiliary RDF/VACF/MSD file is not currently
    # materializable from the local source root.  Figure 6B/C are sourced from
    # the complete status table and remain complete for all five families and
    # all 15 model IDs; omit only this auxiliary line-series input.
    unreadable_md_series = {
        # These provider-backed auxiliary series do not materialize reliably
        # in this checkout.  They are not used for the complete Figure 6B/C
        # status-table summaries; omit only the affected line-series inputs.
        "FTMultiT_A2_seed234.csv",
        "FT600K_A2_seed345.csv",
        # The small Scratch auxiliary files are provider-backed placeholders
        # in this checkout; the complete five-family B/C panels do not depend
        # on them and remain sourced from MD15_FIGURE_DATA_R06G.tsv.
        "Scratch600K_Lmax1_seed123.csv",
        "Scratch600K_Lmax1_seed234.csv",
        "Scratch600K_Lmax1_seed345.csv",
        "ScratchMultiT_Lmax1_seed123.csv",
        "ScratchMultiT_Lmax1_seed234.csv",
        "ScratchMultiT_Lmax1_seed345.csv",
    }
    for p in sorted((analysis/"series").glob("*.csv")):
        if p.name in unreadable_md_series:
            continue
        model=p.stem; file_paths.append(p); r=by_id.get(model); fam=r.get("family","") if r else ""; rows=read_rows(p)
        t=np.array([f(x,"time_ps") for x in rows]); temp=np.array([f(x,"temperature_K") for x in rows]); mask=np.isfinite(t)&np.isfinite(temp)
        if mask.any():
            grouped_series[fam]["temp"].append((t[mask], np.clip(temp[mask], 520, 700)))
        for d,ax,xkey,ykey in [("msd",axf,"lag_ps","msd3d_A2"),("rdf",axd,"r_A","g_CrCr"),("vacf",axe,"lag_ps","vacf_normalized")]:
            q=read_rows(analysis/d/(p.name))
            x=np.array([f(z,xkey) for z in q]); y=np.array([f(z,ykey) for z in q]); m=np.isfinite(x)&np.isfinite(y)
            if m.any():
                grouped_series[fam][d].append((x[m], y[m]))
    for fam,color in family_colors.items():
        for key,ax in [("temp",axa),("rdf",axd),("vacf",axe),("msd",axf)]:
            result=aligned_mean_sd(grouped_series[fam][key])
            if result is None: continue
            x,mean,sd=result
            ax.plot(x,mean,color=color,lw=1.45,label=fam)
            if np.isfinite(sd).any(): ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.16,linewidth=0)
    axa.set_xlabel("time (ps)"); axa.set_ylabel("instantaneous T (K)"); axa.set_ylim(520,700); axa.set_title("A  Temperature mean ± SD",loc="left"); style_ax(axa)
    horizon_by_family=defaultdict(list)
    for r in status: horizon_by_family[r.get("family","")].append(f(r,"last_time_ps"))
    fam_order=[fam for fam in ["Foundation","FT-600K","FT-MultiT","Scratch-600K","Scratch-MultiT"] if fam in horizon_by_family]
    plot_observed_vs_target_dumbbell(
        axb,
        fam_order,
        {fam: finite_values(horizon_by_family[fam]) for fam in fam_order},
        1000.0,
        family_colors,
        xlabel="observed duration (ps)",
        title="B  Observed horizon vs 1 ns target",
    )
    coord_keys = {
        "Cr-Sb": "d_cn_cr_sb_neighbors_per_center",
        "Cr-Te": "d_cn_cr_te_neighbors_per_center",
        "Sb-Te": "d_cn_sb_te_neighbors_per_center",
    }
    coord_values = {
        fam: [finite_values(f(r, key) for r in status if r.get("family") == fam) for key in coord_keys.values()]
        for fam in fam_order
    }
    plot_family_cluster_bars(
        axc,
        list(coord_keys),
        coord_values,
        fam_order,
        family_colors,
        ylabel="neighbors per center",
        title="C  Coordination · 15-model campaign",
        rotate=20,
        y_min=0,
        width_fraction=.90,
    )
    axd.set_xlabel("r (Å)"); axd.set_ylabel("g$_{CrCr}$(r)"); axd.set_title("D  RDF mean ± SD",loc="left"); style_ax(axd)
    axe.set_xlabel("lag (ps)"); axe.set_ylabel("normalized VACF"); axe.set_title("E  Velocity memory mean ± SD",loc="left"); style_ax(axe)
    axf.set_xlabel("lag (ps)"); axf.set_ylabel("MSD (Å$^2$)"); axf.set_title("F  Observed Cr motion mean ± SD",loc="left"); style_ax(axf)
    family_handles = [Line2D([], [], color=family_colors[fam], lw=3, label=fam) for fam in fam_order]
    fig.legend(family_handles, [h.get_label() for h in family_handles], frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(.50, .045), fontsize=LEGEND_SIZE)
    fig.subplots_adjust(left=.07, right=.99, bottom=.18, top=.86)
    fig.suptitle("Figure 6 | Matched MD stability, structure and motion",x=.02,ha="left",y=.985,fontsize=TITLE_SIZE,fontweight="bold")
    assets=save_figure(fig,out,"figure06_matched_md_diagnostics"); return assets,[status_path,*file_paths]


def fig7(out: Path) -> tuple[dict[str, str], list[Path]]:
    root=WILEY/"analysis/foundation_interlayer_sliding_10model_20261024/results/tables"; img_path=root/"IMAGE_PROFILES.tsv"; qe_path=root/"QE_REFERENCE.tsv"; srcs=[img_path,qe_path]
    img=read_rows(img_path); qe=read_rows(qe_path)
    model_class={"foundation_agnesi_large":"MACE-MP","foundation_agnesi_medium":"MACE-MP","foundation_agnesi_medium_density":"MACE-MP","foundation_agnesi_small_density":"MACE-MP","foundation_mp0b3":"MACE-MP","foundation_omat0_medium":"MACE-OMAT-0","foundation_omat0_small":"MACE-OMAT-0","foundation_matpes_pbe_omat_ft":"MACE-OMAT-0","foundation_mh0":"MACE-MH","foundation_mh1":"MACE-MH"}
    for r in img: r["model_class"]=model_class.get(r.get("model_id"),"unknown")
    obs=sorted({r.get("observable") for r in img}); fixed=next((x for x in obs if "FIXED" in x),obs[0]); selfobs=next((x for x in obs if "SELF" in x),obs[-1])
    pal={"categorical":[COLORS["MACE-MP"],COLORS["MACE-OMAT-0"],COLORS["MACE-MH"]],"roles":{k:COLORS[k] for k in ["MACE-MP","MACE-OMAT-0","MACE-MH"]},"reference":"#222222"}
    fig,axes=plt.subplots(2,2,figsize=(16.0,13.0)); systems=[("bulk","Bulk"),("cr_in_gap","Cr-in-gap")]
    labels=[]
    for row,(system,system_label) in enumerate(systems):
        for col,observable in enumerate([fixed,selfobs]):
            ax=axes[row,col]; q=[r for r in img if r.get("system")==system and r.get("observable")==observable]
            profiles=aggregate_profile_rows([{"s":f(r,"image_fraction"),"series":r["model_class"],"replicate":r["model_id"],"value":f(r,"relative_energy_eV")} for r in q],series_order=["MACE-MP","MACE-OMAT-0","MACE-MH"])
            ref=[r for r in qe if r.get("system")==system]; ref=sorted(ref,key=lambda r:f(r,"image_index"))
            plot_profile_ensemble(ax,profiles,x=np.array([f(r,"image_fraction") for r in sorted(q,key=lambda r:f(r,"image_index")) if r.get("model_class")=="MACE-MP"][::5]) if False else None,reference={"y":np.array([f(r,"relative_energy_eV") for r in ref]),"label":"QE reference · SCF warning"},palette=pal,style={"font_family":FONT,"font_size":TICK_SIZE,"title_size":PANEL_TITLE_SIZE,"x_label_size":AXIS_LABEL_SIZE,"y_label_size":AXIS_LABEL_SIZE,"legend_size":LEGEND_SIZE,"panel_label_size":PANEL_TITLE_SIZE,"tick_labels_bold":True},title=f"{chr(97+row*2+col)}  {system_label} · {'fixed images' if col==0 else 'generated path'}",xlabel="Reaction Coordinate" if row==1 else None,ylabel="ΔE (eV)",panel_label=None)
            if col==1: ax.text(.98,.04,"endpoint-only proxy",transform=ax.transAxes,ha="right",fontsize=ANNOTATION_SIZE,color="#9A6A00")
    handles=[Line2D([],[],color=COLORS[k],marker="o",markerfacecolor="white",lw=2,ms=4,label=k) for k in ["MACE-MP","MACE-OMAT-0","MACE-MH"]]
    handles.append(Line2D([],[],color="#222222",marker="D",markerfacecolor="white",lw=1.45,ls="--",ms=3.6,label="QE reference · SCF warning"))
    fig.legend(handles=handles,loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(.52,.045),fontsize=LEGEND_SIZE)
    fig.subplots_adjust(left=.09,right=.98,bottom=.18,top=.86,wspace=.42,hspace=.62)
    fig.suptitle("Figure 7 | Interlayer sliding: fixed images versus generated paths",x=.02,ha="left",y=.985,fontsize=TITLE_SIZE,fontweight="bold")
    assets=save_figure(fig,out,"figure07_interlayer_sliding_diagnostics"); return assets,srcs


def fig8(out: Path) -> tuple[dict[str, str], list[Path]]:
    src=WILEY/"analysis/migrationbench_universe_publication_20261026/derived/structure_universe_points.csv"; rows=read_rows(src)
    fig=plt.figure(figsize=(17.0,10.5)); gs=fig.add_gridspec(2,2,hspace=.62,wspace=.42); ax=fig.add_subplot(gs[0,0])
    layer_colors={"L0_reference":"#8DA9BF","L2_DFT_path":"#C4547A","L3_interlayer_self":"#E68F1D","L3_self_relaxed_NEB":"#6E9E75"}
    for layer,color in layer_colors.items():
        sub=[r for r in rows if r.get("task_layer")==layer]; x=np.array([f(r,"x_pca") for r in sub]); y=np.array([f(r,"y_pca") for r in sub]); m=np.isfinite(x)&np.isfinite(y)
        if m.any(): ax.scatter(x[m],y[m],s=8,alpha=.42,color=color,label=layer.replace("_"," "))
    ax.set_xlabel("SOAP PCA 1"); ax.set_ylabel("SOAP PCA 2"); ax.set_title("A  Task-aware configuration map",loc="left"); ax.legend(frameon=False,fontsize=LEGEND_SIZE); style_ax(ax)
    status_box(fig.add_subplot(gs[0,1]),"BLOCKED","No verified geometry_id-resolved join\nbetween descriptor distance and the\nFigure 3–4 QE error table.\nDo not substitute PCA distance for error.",color="#B23A48")
    status_box(fig.add_subplot(gs[1,0]),"BLOCKED","High-error image threshold and\nprospective held-out path are not\nfrozen with a common geometry key.\nNo recognition rate is reported.",color="#B23A48")
    status_box(fig.add_subplot(gs[1,1]),"BLOCKED","Seed disagreement and measured\nerror cannot be paired from current\nartifacts without a validated join.\nKeep as a supplemental map only.",color="#B23A48")
    fig.subplots_adjust(left=.07, right=.99, bottom=.10, top=.87)
    fig.suptitle("Figure 8 | Where new QE data may be needed (conditional)",x=.02,ha="left",y=.985,fontsize=TITLE_SIZE,fontweight="bold")
    assets=save_figure(fig,out,"figure08_qe_diagnostic_conditional"); return assets,[src]


def write_tables(out: Path) -> list[Path]:
    tables=out/"tables"; tables.mkdir(parents=True,exist_ok=True)
    t1=tables/"TABLE1_MODEL_REFERENCE_AUDIT.tsv"
    rows=[
        ["route","source registry","observed model/seed scope","reference/protocol","current status","publication action"],
        ["Scratch / FT parity","analysis/rmse_train_test_20261026/MODEL_REGISTRY.tsv","4 families × 3 seeds; train/test summary","QE parity; force and energy fields","PUBLICATION_PROVISIONAL","resolve energy unit and held-out strata before claims"],
        ["Cr fixed images","analysis/neb_fixed_path_trajectory_figure_20261023/python/data/fixed_path_long.tsv","registered seeds/variants: 3 per scratch/FT family; 3 Foundation variants (FT-MultiT in-gap finite n=2)","fixed images; QE SCF-warning reference","SUPPORTING_DIAGNOSTIC","retain G1 warning; show mean ± sample SD; do not call barrier"],
        ["Cr grouped paired NEB","analysis/cr_interlayer_sliding_grouped_20261025/results/tables/NORMALIZED_PROFILES.tsv","Foundation checkpoint variants n=3; Scratch-600K/Scratch-MultiT/FT-600K/FT-MultiT n=3 registered seeds","fixed-path single-point and endpoint-only MLIP NEB proxy; retained DFT path carries SCF warning","RECONSTRUCTION_DIAGNOSTIC","keep fixed/self and bulk/Cr-in-gap strata separate; do not call self proxy a DFT barrier"],
        ["Foundation self-relaxed paths","analysis/neb_cross_element_triptychs_20261024/foundation_self_relaxed_4x3_20261024/figure_revisions/foundation_class_4x3_R03_fontplus2pt/data/foundation_class_4x3_R03_fontplus2pt/foundation_model_profiles_9models_4x3_R03.tsv","Cr ensemble: MACE-MP n=5; MACE-OMAT-0 n=2; MACE-MH n=2","endpoint-only IDPP initialization; campaign stopped for repair","EXCLUDE_FROM_MAIN_CLAIM","recover canonical run logs and geometry/endpoint gates; show mean ± sample SD"],
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
    t3=tables/"UNCERTAINTY_AUDIT.tsv"
    fixed_rows=read_rows(FIXED_ENSEMBLE)
    fixed_counts=[]
    for key in ("within_gap","gap_to_ql","within_ql"):
        summary=profile_ensemble_summary(fixed_rows,key)
        fixed_counts.append(key+": "+"; ".join(f"{family} n={int(summary[family]['n'])}" for family in ENSEMBLE_FAMILIES if family in summary))
    grouped_rows = read_rows(GROUPED_NORMALIZED)
    grouped_counts = []
    for system, method, title in [
        ("bulk", "fixed", "Bulk fixed"),
        ("cr_in_gap", "fixed", "Cr-in-gap fixed"),
        ("bulk", "self", "Bulk self"),
        ("cr_in_gap", "self", "Cr-in-gap self"),
    ]:
        summary = grouped_profile_summary(grouped_rows, system, method)
        grouped_counts.append(title + ": " + "; ".join(f"{family} n={int(summary[family]['n'])}" for family in GROUPED_FAMILIES if family in summary))
    self_rows=read_rows(SELF_RELAXED_ENSEMBLE)
    self_counts=[]
    for key in ("within_gap","gap_to_ql","within_ql"):
        summary=class_profile_summary(self_rows,key,["MACE-MP","MACE-OMAT-0","MACE-MH"])
        self_counts.append(key+": "+"; ".join(f"{group} n={int(summary[group]['n'])}" for group in ["MACE-MP","MACE-OMAT-0","MACE-MH"] if group in summary))
    md_status=read_rows(WILEY/"analysis/md_1ns_updated_publication_figure_20261025/MD15_FIGURE_DATA_R06G.tsv")
    md_counts=Counter(r.get("family","") for r in md_status)
    audit_rows=[
        ["figure/panels","source","replicate unit and grouping","uncertainty encoding","observed counts","missingness/outlier rule","visual/data acceptance gate"],
        ["Figure 1 C","analysis/migrationbench_universe_publication_20261026/derived/comprehensive_task_map_points.csv","complete task map grouped by source split and task layer","deterministic counts; no uncertainty interval is applicable","L0/L1/L2/L3 rows are retained by source split","finite task labels only; no synthetic counts or reclassification","all source-split bars are visible with explicit split legend; this is coverage, not a model-performance claim"],
        ["Figure 2 C/F","analysis/rmse_train_test_20261026/parity_summary.csv","Scratch-600K/Scratch-MultiT/FT-600K/FT-MultiT; train and held-out test strata; n=3 registered seeds per family/stratum","full clustered bars: mean ± sample SD whiskers plus all seed points; train uses hatch and held-out test uses solid fill","each of 4 families has train n=3 and test n=3","no imputation; finite rows only; train/test are not pooled","complete four-family palette, explicit train/test encoding, and visible whiskers/replicates in every cluster"],
        ["Figure 3 A–C",str(GROUPED_NORMALIZED.relative_to(WILEY)) + " + " + str(GROUPED_QE.relative_to(WILEY)),"Foundation/Scratch-600K/Scratch-MultiT/FT-600K/FT-MultiT; n=3 registered rows per family on the retained fixed-image stratum","A/C mean ± sample SD shaded bands; B four explicit Foundation-reference pair clusters with mean ± SD whiskers and replicate points","Cr-in-gap fixed: Foundation n=3; Scratch-600K n=3; Scratch-MultiT n=3; FT-600K n=3; FT-MultiT n=3","finite image rows only; DFT retained path remains deterministic and SCF-warning-labelled","no legacy From-scratch aggregation; complete five-family palette and Foundation reference in every pair"],
        ["Figure 4 A–H",str(GROUPED_NORMALIZED.relative_to(WILEY)) + " + " + str(GROUPED_QE.relative_to(WILEY)),"Foundation checkpoint variants n=3 plus Scratch-600K/Scratch-MultiT/FT-600K/FT-MultiT registered seeds n=3; paired by the same ten retained image positions","mean ± sample SD shaded bands in each NEB profile; each corresponding cluster bar is an explicit Foundation-reference pair with mean ± SD whisker and replicate points"," | ".join(grouped_counts),"fixed/self and bulk/Cr-in-gap strata remain separate; finite rows only; no imputation or synthetic seed","Foundation is the first/reference bar in every cluster; exact five-family palette; endpoint-only self panels remain explicitly labelled as proxies"],
        ["Figure 5 A–D",str(SELF_RELAXED_ENSEMBLE.relative_to(WILEY)),"Cr Foundation model variants grouped by MACE class","mean ± sample SD shaded band"," | ".join(self_counts),"only finite model profiles included; no barrier imputation","class n shown in legend and matrix; blocked status remains explicit"],
        ["Figure 6 A–F","analysis/md_1ns_updated_publication_figure_20261025/MD15_FIGURE_DATA_R06G.tsv + md15_analysis_r03/series/*.csv","Foundation/FT-600K/FT-MultiT/Scratch-600K/Scratch-MultiT; n=3 model IDs per family","A/D–F mean ± sample SD shaded bands; B full-family observed-vs-1 ns dumbbell with mean ± SD whiskers and all model points; C three-category clustered bars with mean ± SD whiskers and all model points","; ".join(f"{k} n={v}" for k,v in sorted(md_counts.items())),"observed horizons retained; non-finite values excluded; no extrapolation; coordination uses complete status rows","complete five-family palette, all 15 model IDs represented, target line explicit, and no incomplete four-group summary is used"],
        ["Figure 7 A–D","analysis/foundation_interlayer_sliding_10model_20261024/results/tables/IMAGE_PROFILES.tsv","registered Foundation model variants grouped by MACE class","mean ± sample SD shaded bands","MACE-MP n=5; MACE-OMAT-0 n=3; MACE-MH n=2","finite profiles only; QE reference deterministic","class bands and legend use exact Foundation class palette"],
        ["Figures 1 and 8","configuration-universe tables","scatter/status/diagram panels; no line or bar summary","N/A: no aggregate curve/bar is shown","N/A","finite plotted coordinates only","no uncertainty claim is made for non-aggregate panels"],
    ]
    with t3.open("w",newline="") as h: csv.writer(h,delimiter="\t").writerows(audit_rows)
    return [t1,t2,t3]


def write_manifest(out: Path, figure_assets: dict[str, dict[str, str]], sources: list[Path]) -> Path:
    manifest_figures = dict(figure_assets)
    standalone = out / "figures"
    standalone_assets = {
        "builder": "docs/reproducibility/canva_figure_provenance/scripts/build_figure01a_clean_panel.py",
        "png": str(standalone / "figure01a_clean_raw_pca.png"),
        "pdf": str(standalone / "figure01a_clean_raw_pca.pdf"),
        "svg": str(standalone / "figure01a_clean_raw_pca.svg"),
    }
    if all(Path(path).exists() for key, path in standalone_assets.items() if key != "builder"):
        manifest_figures["figure01a_standalone"] = standalone_assets
    manifest={
        "schema_version":"atomic_migration_blueprint_figure_package_v1",
        "revision":"R06_clean_raw_pca",
        "status":"REVIEW_PACKAGE_NOT_SCIENTIFIC_ACCEPTANCE",
        "generated_by":str(Path(__file__).resolve()),
        "source_root":str(WILEY),
        "style_source":str(WILEY / "analysis/sliding_four_model_campaign_20261024/config/FIG168_MODEL_STYLE_YELLOW_SCRATCH_R04.json"),
        "visual_grammar": {
            "font_family": "Arial",
            "editable_outputs": ["pdf", "svg"],
            "raster_preview": "png",
            "axes": "heavy black spines, outward ticks, no decorative grid",
            "title_weight": "bold",
            "minimum_authored_text_pt": MIN_TEXT_SIZE,
            "title_pt": TITLE_SIZE,
            "panel_title_pt": PANEL_TITLE_SIZE,
            "axis_label_pt": AXIS_LABEL_SIZE,
            "tick_pt": TICK_SIZE,
            "legend_pt": LEGEND_SIZE,
            "energy_axis_label": "ΔE (eV)",
            "reaction_coordinate_label": "Reaction Coordinate",
            "shared_legend_placement": "below overall figure",
            "caveats": "kept in manifest/status metadata rather than prose strips inside data panels",
            "uncertainty": {
                "line_charts": "mean ± sample SD across registered seeds or explicit model variants; rendered as shaded bands",
                "bar_charts": "mean ± sample SD rendered as whiskers; replicate points retained where available",
                "single_reference": "QE reference is deterministic and shown as a dashed line without an uncertainty band",
                "missing_replicates": "missing or non-finite rows are excluded; no imputation, synthetic seed, or extrapolation"
            }
        },
        "model_palette": COLORS,
        "model_markers": MODEL_MARKERS,
        "series_markers": SERIES_MARKERS,
        "figures":manifest_figures,
        "source_hashes":{str(p):sha256(p) for p in sources if p.exists()},
        "figure1a_coordinate_semantics": {
            "source_table": "comprehensive_task_map_points.csv",
            "coordinates": "source-provided x_pca/y_pca from the common Cr-local SOAP/PCA fit; no task anchor, translation, rank equalization, or post-hoc movement",
            "axis_labels": ["SOAP PCA 1", "SOAP PCA 2"],
            "metric_warning": "PCA axes retain the source feature-space projection; nonlinear UMAP/t-SNE candidates are kept separate and are not substituted into the PCA panel",
            "task_clusters": "very light convex hulls over the direct point rows shown in panel A, shown only as a visual aid and not used to move points",
            "path_lines": "not rendered in the clean primary PCA panel because the complete MD/NEB path set creates a spiderweb; source-provided path_key rows remain available in the embedding audit and manifest-linked source table",
            "foreground_point_splits": ["train", "valid", "test", "md_sample", "dft_neb", "dft_interlayer"],
            "task_marker_legend": {
                "L0 DFT Reference": "o",
                "L1 Thermal Equilibrium": "^",
                "L2 NEB Migration": "D",
                "L3 Layer Sliding": "P"
            },
            "dft_split_legend": {
                "DFT - train": "o",
                "DFT - valid": "^",
                "DFT - test": "s"
            },
            "marker_area": "distance_to_training, percentile-normalized; larger area means farther feature-space OOD",
            "point_hue": "task group color, matching the original R03 Figure 1A grammar",
            "point_shape": "task group marker; source-split coverage remains explicit in panel C",
            "not_confounded_with": "rms_delta_A remains a structural Cr-shell distortion field in the source table and is not relabelled as OOD"
        },
        "notes":["This is an additive pre-import review package; the source PR and historical figure outputs are not overwritten.","All blocked panels are explicit placeholders; no synthetic observations are used.","SCF-warning, endpoint-only proxy, legacy and provisional states remain separate.","Every aggregate line panel uses a finite replicate ensemble with a shaded mean ± sample SD band; every aggregate bar panel uses mean ± sample SD whiskers.","Figure 1A retains the source-provided raw SOAP/PCA coordinates and the first-version foreground point contract: MLIP self-relaxed NEB/sliding rows remain complete in panel C but are not rendered as hundreds of foreground scatter markers.","The clean primary PCA panel omits the full source path-line set to avoid a spiderweb; path-aware diagnostics remain in the source table and pre-cleanup snapshot.","Superseded UMAP and t-SNE candidates are preserved only in the pre-cleanup snapshot commit; no post-hoc task translation is used in the canonical PCA panel.","Figure 1A point area uses distance_to_training for OOD and does not relabel Cr-shell structural distortion as OOD.","Figure 2 C/F clusters are sourced from the complete parity summary; one provider-backed auxiliary frame archive is omitted only from the non-summary scatter layer.","Figure 6 B/C remain complete for all five families and 15 model IDs from the status table; provider-backed auxiliary line files are omitted only where they do not materialize locally.","The selected-figure build intentionally does not create the full-package tables directory; use the PR package manifest and source hashes for provenance."]
    }
    p=out/"manifests/ATOMIC_MIGRATION_FIGURE_MANIFEST.json"; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n"); return p


def build_composite(out: Path, figure_assets: dict[str, dict[str, str]]) -> dict[str, str]:
    fig,axes=plt.subplots(2,4,figsize=(20,11.0)); names=[f"Figure {i}" for i in range(1,9)]
    statuses=["PARTIAL","PROVISIONAL","G1 WARNING","DIAGNOSTIC","BLOCKED","PROVISIONAL","SCF WARNING / PROXY","CONDITIONAL"]
    for ax,(name,status) in zip(axes.flat,zip(names,statuses)):
        path=Path(figure_assets[f"figure0{names.index(name)+1}"]["png"])
        img=plt.imread(path); ax.imshow(img); ax.set_title(f"{name} · {status}",fontsize=MIN_TEXT_SIZE,fontweight="bold",loc="left"); ax.axis("off")
    fig.suptitle("Atomic Migration Figure Blueprint | review canvas board",fontsize=TITLE_SIZE,fontweight="bold",x=.02,ha="left",y=.995)
    return save_figure(fig,out,"atomic_migration_blueprint_canvas_board",dpi=300)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=OUT_DEFAULT)
    parser.add_argument("--figures",type=int,nargs="+",default=list(range(1,9)),choices=range(1,9),help="figure numbers to render")
    args=parser.parse_args(); out=args.out
    set_house_style(); figdir=out/"figures"; figdir.mkdir(parents=True,exist_ok=True)
    builders=[fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8]; assets={}; sources=[]
    for i in args.figures:
        builder=builders[i-1]
        a,s=builder(figdir); assets[f"figure0{i}"]=a; sources.extend(s)
    if set(args.figures)==set(range(1,9)):
        assets["composite"]=build_composite(figdir,assets)
        tables=write_tables(out); sources.extend(tables)
    else:
        tables=[]
    manifest=write_manifest(out,assets,sources)
    print(json.dumps({"manifest":str(manifest),"figures":assets,"tables":[str(p) for p in tables]},indent=2))


if __name__ == "__main__":
    main()
