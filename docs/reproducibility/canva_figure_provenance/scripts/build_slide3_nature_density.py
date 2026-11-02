#!/usr/bin/env python3
"""Build a Nature-style density/parity treatment for Canva page 3.

Two products are rendered:

* ``reference_style_demo`` is a deterministic, synthetic 2x3 recreation of
  the supplied Nature-style figure.  It is a style proof only and is never
  mixed with MigrationBench evidence.
* ``slide3_parity_density`` uses the audited MigrationBench parity NPZ outputs.
  Hexbin counts use every finite parity observation; the visible
  train/test point layers use a deterministic vector sample so the SVG stays
  practical to edit in Canva.

The project plot keeps the actual ``train``/``test`` terminology.  The test
statistics are computed only from held-out test observations and are printed
to two decimals.  All text is set to at least 18 pt in the source figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


WORKTREE = Path(__file__).resolve().parents[4]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA = PACKAGE_ROOT / "data"
EXTERNAL_DATA_ROOT = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/rmse_train_test_20261026")
DATA_ROOT = LOCAL_DATA / "figure02" if (LOCAL_DATA / "figure02").is_dir() else EXTERNAL_DATA_ROOT
RAW_ROOT = LOCAL_DATA / "figure02/raw_rockfish_parity" if (LOCAL_DATA / "figure02/raw_rockfish_parity").is_dir() else DATA_ROOT / "raw/rockfish_parity"
SUMMARY_PATH = DATA_ROOT / "parity_summary.csv"
OUT_ROOT = WORKTREE / "docs/reproducibility/canva_figure_provenance/R04_slide3_nature_density"
FIG_ROOT = OUT_ROOT / "figures"

FONT = "Arial"
MIN_TEXT = 18
TITLE_SIZE = 22
PANEL_TITLE = 20
AXIS_LABEL = 20
TICK_SIZE = 18
ANNOTATION = 18
LEGEND_SIZE = 18
COLORBAR_SIZE = 18
FORCE_LIMIT = 5.25

# The existing R04 family colors are retained exactly.
COLORS = {
    "FT-600K": "#002E71",
    "FT-MultiT": "#67ACE5",
    "Scratch-600K": "#FF9F1B",
    "Scratch-MultiT": "#F4C95D",
}
FAMILIES = ["Scratch-600K", "FT-600K", "Scratch-MultiT", "FT-MultiT"]
SEEDS = ["123", "234", "345"]
SPLITS = ["train", "test"]
METRICS = {
    "energy": {
        "x": "energy_ref_eV_per_atom",
        "y": "energy_pred_eV_per_atom",
        "x_label": "DFT energy / atom (eV)",
        "y_label": "MLIP energy / atom (eV)",
    },
    "force": {
        "x": "force_ref_eV_A",
        "y": "force_pred_eV_A",
        "x_label": "DFT force component (eV Å$^{-1}$)",
        "y_label": "MLIP force component (eV Å$^{-1}$)",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT, "Helvetica", "DejaVu Sans"],
            "font.size": MIN_TEXT,
            "axes.titlesize": PANEL_TITLE,
            "axes.titleweight": "bold",
            "axes.labelsize": AXIS_LABEL,
            "axes.labelweight": "bold",
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.linewidth": 1.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(axis="both", which="major", direction="out", length=7, width=1.35, pad=6, labelsize=TICK_SIZE)
    ax.tick_params(axis="both", which="minor", direction="out", length=3, width=0.8)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.25)
        spine.set_color("#111111")
    for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick.set_fontfamily(FONT)
        tick.set_fontsize(TICK_SIZE)
        tick.set_fontweight("bold")


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {ext: FIG_ROOT / f"{stem}.{ext}" for ext in ("png", "pdf", "svg")}
    fig.savefig(paths["png"], dpi=420, bbox_inches="tight", pad_inches=0.18, facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.18, facecolor="white")
    fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.18, facecolor="white")
    plt.close(fig)
    return {key: str(value) for key, value in paths.items()}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rank_values(values: np.ndarray) -> np.ndarray:
    """Average-rank values without requiring SciPy."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rank_values(np.asarray(x, dtype=float)), rank_values(np.asarray(y, dtype=float)))


def finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def sample_points(x: np.ndarray, y: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic evenly spaced sample for editable vector point overlays."""
    if x.size <= limit:
        return x, y
    indices = np.linspace(0, x.size - 1, limit, dtype=int)
    return x[indices], y[indices]


def draw_hexbin(ax: plt.Axes, x: np.ndarray, y: np.ndarray, *, extent: tuple[float, float, float, float], norm: LogNorm, gridsize: int = 48):
    return ax.hexbin(
        x,
        y,
        gridsize=gridsize,
        extent=extent,
        mincnt=1,
        cmap="magma",
        norm=norm,
        linewidths=0.0,
        edgecolors="none",
        rasterized=False,
    )


def add_vector_colorbar(fig: plt.Figure, reference_ax: plt.Axes, norm: LogNorm, label: str) -> None:
    """Add a discrete, fully vector count colorbar beside one panel row."""
    bbox = reference_ax.get_position()
    cax = fig.add_axes([bbox.x1 + 0.014, bbox.y0, 0.018, bbox.height])
    cax.set_yscale("log")
    cax.set_xlim(0.0, 1.0)
    cax.set_ylim(norm.vmin, norm.vmax)
    cmap = mpl.colormaps["magma"]
    edges = np.geomspace(norm.vmin, norm.vmax, 72)
    for lower, upper in zip(edges[:-1], edges[1:]):
        midpoint = np.sqrt(lower * upper)
        cax.add_patch(
            Rectangle(
                (0.0, lower),
                1.0,
                upper - lower,
                facecolor=cmap(norm(midpoint)),
                edgecolor="none",
                linewidth=0.0,
                transform=cax.transData,
            )
        )
    cax.xaxis.set_visible(False)
    cax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
    cax.yaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
    cax.yaxis.tick_right()
    cax.tick_params(axis="y", which="major", labelsize=COLORBAR_SIZE, width=1.0, length=5, pad=5)
    cax.set_ylabel(label, fontsize=COLORBAR_SIZE, fontweight="bold", labelpad=13, rotation=270)
    cax.yaxis.set_label_position("right")
    for spine in cax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.15)
        spine.set_color("#111111")


def count_norm(panel_points: list[tuple[np.ndarray, np.ndarray]], extent: tuple[float, float, float, float], gridsize: int = 48) -> LogNorm:
    max_count = 1
    x0, x1, y0, y1 = extent
    for x, y in panel_points:
        counts, _, _ = np.histogram2d(x, y, bins=(gridsize, gridsize), range=((x0, x1), (y0, y1)))
        max_count = max(max_count, int(np.max(counts)))
    return LogNorm(vmin=1, vmax=max(2, max_count))


def count_norm_by_extent(
    panel_points: list[tuple[np.ndarray, np.ndarray]],
    extents: list[tuple[float, float, float, float]],
    gridsize: int = 48,
) -> LogNorm:
    """Return one shared count scale while allowing panel-specific limits."""
    max_count = 1
    for (x, y), extent in zip(panel_points, extents, strict=True):
        x0, x1, y0, y1 = extent
        counts, _, _ = np.histogram2d(x, y, bins=(gridsize, gridsize), range=((x0, x1), (y0, y1)))
        max_count = max(max_count, int(np.max(counts)))
    return LogNorm(vmin=1, vmax=max(2, max_count))


def reference_demo_data() -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Create deterministic style-only train/valid clouds matching the reference grammar."""
    rng = np.random.default_rng(20261031)
    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for row, (n_train, n_valid) in enumerate(((45, 5), (450, 50))):
        for col in range(3):
            scale = (1.0 + 0.28 * col) * (1.0 if row == 0 else 0.42)
            train_x = np.abs(rng.normal(0.9 + 0.22 * col, 0.23 * scale, n_train))
            train_y = np.abs(0.55 * train_x + rng.normal(0.55 + 0.1 * col, 0.18 * scale, n_train))
            valid_x = np.abs(rng.normal(1.25 + 0.34 * col, 0.30 * scale, n_valid))
            valid_y = np.abs(0.72 * valid_x + rng.normal(0.40 + 0.08 * col, 0.16 * scale, n_valid))
            out[(row, col)] = (train_x, train_y, valid_x, valid_y)
    return out


def render_reference_demo() -> dict[str, str]:
    demo = reference_demo_data()
    titles = ["Ensemble", "Distance", "Posterior"]
    fig, axes = plt.subplots(2, 3, figsize=(19.8, 12.0), sharex=False, sharey=False, squeeze=False)
    all_counts: list[int] = []
    row_norms: list[LogNorm] = []
    extents: list[tuple[float, float, float, float]] = []
    for row in range(2):
        row_points = []
        x_all: list[np.ndarray] = []
        y_all: list[np.ndarray] = []
        for col in range(3):
            tr_x, tr_y, va_x, va_y = demo[(row, col)]
            x_all.extend((tr_x, va_x))
            y_all.extend((tr_y, va_y))
        x_max = max(float(np.max(x)) for x in x_all) * 1.18
        y_max = max(float(np.max(y)) for y in y_all) * 1.18
        extent = (0.0, x_max, 0.0, y_max)
        extents.append(extent)
        for col in range(3):
            tr_x, tr_y, va_x, va_y = demo[(row, col)]
            row_points.append((np.concatenate((tr_x, va_x)), np.concatenate((tr_y, va_y))))
        row_norms.append(count_norm(row_points, extent, gridsize=42))

    for row in range(2):
        norm = row_norms[row]
        for col in range(3):
            ax = axes[row, col]
            tr_x, tr_y, va_x, va_y = demo[(row, col)]
            x = np.concatenate((tr_x, va_x))
            y = np.concatenate((tr_y, va_y))
            hb = draw_hexbin(ax, x, y, extent=extents[row], norm=norm, gridsize=42)
            ax.scatter(tr_x, tr_y, s=34, color="#8d8d8d", alpha=0.32, linewidths=0, zorder=3)
            ax.scatter(va_x, va_y, s=42, color="#111111", alpha=0.92, linewidths=0, zorder=4)
            lo, hi = extents[row][0], min(extents[row][1], extents[row][3])
            ax.plot([lo, hi], [lo, hi], linestyle=(0, (6, 4)), color="#cf2e2e", linewidth=2.4, zorder=2)
            ax.set_xlim(extents[row][0], extents[row][1])
            ax.set_ylim(extents[row][2], extents[row][3])
            if row == 0:
                ax.set_title(titles[col], fontsize=TITLE_SIZE, fontweight="bold", pad=12)
            ax.text(
                0.055,
                0.95,
                f"$N_{{{{\\mathrm{{{{train}}}}}}}}$ = {tr_x.size}, $N_{{{{\\mathrm{{{{valid}}}}}}}}$ = {va_x.size}\n"
                f"Spearman: {spearman(va_x, va_y):.2f}\nPearson: {pearson(va_x, va_y):.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=ANNOTATION,
                linespacing=1.15,
            )
            style_axis(ax)
            ax.tick_params(labelsize=TICK_SIZE)
            if row == 0:
                ax.tick_params(labelbottom=False)
            if col > 0:
                ax.tick_params(labelleft=False)
    fig.supxlabel("Maximal atom-based uncertainty in eV Å$^{-1}$", fontsize=AXIS_LABEL, fontweight="bold", y=0.035)
    fig.supylabel("Maximal atomic force RMSE in eV Å$^{-1}$", fontsize=AXIS_LABEL, fontweight="bold", x=0.018)
    handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor="#8d8d8d", markeredgecolor="none", alpha=0.32, markersize=10, label="train"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="#111111", markeredgecolor="none", alpha=0.92, markersize=10, label="valid"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=LEGEND_SIZE, handletextpad=0.55, columnspacing=1.6)
    fig.subplots_adjust(left=0.085, right=0.91, bottom=0.13, top=0.91, hspace=0.25, wspace=0.16)
    for row in range(2):
        add_vector_colorbar(fig, axes[row, -1], row_norms[row], "Count")
    return save_figure(fig, "reference_style_demo_nature_density")


def load_project_arrays() -> dict[tuple[str, str, str], dict[str, np.ndarray]]:
    """Load the complete parity matrix directly from the audited NPZ outputs.

    The JSON sidecars live in iCloud and may be metadata-only while the parity
    NPZ contents are already locally readable.  The NPZ eval_id is the exact
    output identifier, so family/seed/split are parsed from that identifier and
    the expected 4 x 3 x 2 matrix is checked before plotting.
    """
    arrays: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    family_by_prefix = {
        "ft_600k_": "FT-600K",
        "ft_multit_": "FT-MultiT",
        "scratch_600k_": "Scratch-600K",
        "scratch_multit_": "Scratch-MultiT",
    }
    for npz_path in sorted(RAW_ROOT.glob("*_parity.npz")):
        eval_id = npz_path.name.removesuffix("__parity.npz")
        prefix = next((prefix for prefix in family_by_prefix if eval_id.startswith(prefix)), None)
        match = re.search(r"_seed(123|234|345)_(train|test)$", eval_id)
        if prefix is None or match is None:
            continue
        key = (family_by_prefix[prefix], match.group(1), match.group(2))
        with np.load(npz_path, allow_pickle=False) as data:
            arrays[key] = {name: np.asarray(data[name], dtype=float) for name in data.files}
    expected = {(family, seed, split) for family in FAMILIES for seed in SEEDS for split in SPLITS}
    if set(arrays) != expected:
        raise RuntimeError(f"Incomplete parity matrix: missing={sorted(expected - set(arrays))}")
    return arrays


def project_xy(arrays: dict[tuple[str, str, str], dict[str, np.ndarray]], family: str, split: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    x_key = METRICS[metric]["x"]
    y_key = METRICS[metric]["y"]
    xs = []
    ys = []
    for seed in SEEDS:
        x, y = finite_xy(arrays[(family, seed, split)][x_key], arrays[(family, seed, split)][y_key])
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def project_limits(arrays: dict[tuple[str, str, str], dict[str, np.ndarray]], metric: str) -> tuple[float, float, float, float]:
    xs = []
    ys = []
    for family in FAMILIES:
        for split in SPLITS:
            x, y = project_xy(arrays, family, split, metric)
            xs.append(x)
            ys.append(y)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad, lo - pad, hi + pad


def focused_square_limits(
    x: np.ndarray,
    y: np.ndarray,
    *,
    quantile_low: float = 0.005,
    quantile_high: float = 0.995,
    padding: float = 0.08,
) -> tuple[float, float, float, float]:
    """Focus a panel on its central distribution while preserving a square aspect."""
    values = np.concatenate((x, y))
    lo, hi = np.quantile(values, [quantile_low, quantile_high])
    span = float(hi - lo)
    if not np.isfinite(span) or span <= 0:
        span = max(abs(float(hi)), 1.0)
    lo = float(lo) - padding * span
    hi = float(hi) + padding * span
    return lo, hi, lo, hi


def outside_fraction(x: np.ndarray, y: np.ndarray, extent: tuple[float, float, float, float]) -> float:
    """Fraction of finite observations outside a focused panel extent."""
    x0, x1, y0, y1 = extent
    if x.size == 0:
        return float("nan")
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    return float(1.0 - np.mean(inside))


def energy_inset_limits(panel: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    """Local square window around the densest held-out energy core."""
    x, y = finite_xy(panel["test_x"], panel["test_y"])
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        half_width = 0.12
        center = float(np.mean(np.concatenate((x, y))))
        return center - half_width, center + half_width, center - half_width, center + half_width
    counts, xedges, yedges = np.histogram2d(x, y, bins=80, range=((lo, hi), (lo, hi)))
    i, j = np.unravel_index(int(np.argmax(counts)), counts.shape)
    center_x = 0.5 * (xedges[i] + xedges[i + 1])
    center_y = 0.5 * (yedges[j] + yedges[j + 1])
    bin_width = max(float(xedges[i + 1] - xedges[i]), float(yedges[j + 1] - yedges[j]))
    # The previous local window left too much empty space around the energy
    # core.  Keep a small visual margin while making the distribution occupy
    # more of the inset, without clipping the dense test region.
    half_width = max(0.105, 5.2 * bin_width)
    center = 0.5 * (center_x + center_y)
    return center - half_width, center + half_width, center - half_width, center + half_width


def energy_zero_reference(panel: dict[str, np.ndarray], extent: tuple[float, float, float, float]) -> float:
    """Return a data-derived DFT anchor for a local, relative-energy inset.

    The anchor is the median DFT energy of the held-out observations that lie
    in the selected local core.  It is subtracted from both the DFT and MLIP
    axes, so the parity residual and all correlations are unchanged.  The
    absolute-energy panel remains visible as the provenance/context view.
    """
    x = np.asarray(panel["test_x"], dtype=float)
    x0, x1, _, _ = extent
    local = x[np.isfinite(x) & (x >= x0) & (x <= x1)]
    if local.size == 0:
        local = x[np.isfinite(x)]
    if local.size == 0:
        raise RuntimeError("Cannot define an energy zero reference from an empty panel")
    return float(np.median(local))


def add_energy_inset(ax: plt.Axes, panel: dict[str, np.ndarray], family: str) -> tuple[float, float, float, float]:
    """Add a boxed, hexbin, relative-energy train/test view.

    The local inset uses ``E_rel = E - E0`` on both axes.  ``E0`` is computed
    from the held-out DFT core for this family and applied identically to the
    DFT and MLIP values.  This centers the physically relevant energy core
    without changing any model error or rank statistic.
    """
    absolute_extent = energy_inset_limits(panel)
    energy_offset = energy_zero_reference(panel, absolute_extent)
    x0, x1, y0, y1 = (value - energy_offset for value in absolute_extent)
    extent = (x0, x1, y0, y1)
    inset = ax.inset_axes([0.50, 0.065, 0.46, 0.39], zorder=6)
    inset.set_facecolor("white")
    inset.patch.set_alpha(0.96)
    all_x = np.concatenate((panel["train_x"], panel["test_x"])) - energy_offset
    all_y = np.concatenate((panel["train_y"], panel["test_y"])) - energy_offset
    inset_norm = count_norm([(all_x, all_y)], extent, gridsize=18)
    draw_hexbin(inset, all_x, all_y, extent=extent, norm=inset_norm, gridsize=18).set_alpha(0.62)
    tr_x, tr_y = sample_points(panel["train_x"], panel["train_y"], 2400)
    te_x, te_y = sample_points(panel["test_x"], panel["test_y"], 2400)
    tr_x, tr_y = tr_x - energy_offset, tr_y - energy_offset
    te_x, te_y = te_x - energy_offset, te_y - energy_offset
    inset.scatter(tr_x, tr_y, s=22, color="#8d8d8d", alpha=0.34, linewidths=0, zorder=3)
    inset.scatter(te_x, te_y, s=26, color=COLORS[family], alpha=0.92, linewidths=0, zorder=4)
    inset.plot([x0, x1], [x0, x1], linestyle=(0, (5, 3)), color="#cf2e2e", linewidth=1.5, zorder=2)
    inset.set_xlim(x0, x1)
    inset.set_ylim(y0, y1)
    inset.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
    inset.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
    relative_formatter = mpl.ticker.FormatStrFormatter("%+.2f")
    inset.xaxis.set_major_formatter(relative_formatter)
    inset.yaxis.set_major_formatter(relative_formatter)
    inset.tick_params(axis="both", which="major", direction="out", length=4, width=0.9, pad=2, labelsize=TICK_SIZE)
    inset.minorticks_off()
    for tick in (*inset.get_xticklabels(), *inset.get_yticklabels()):
        tick.set_fontfamily(FONT)
        tick.set_fontsize(TICK_SIZE)
        tick.set_fontweight("bold")
    inset.text(
        0.97,
        0.95,
        rf"$E_0={energy_offset:.0f}\ \mathrm{{eV}}$",
        transform=inset.transAxes,
        ha="right",
        va="top",
        fontsize=MIN_TEXT,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.25)
        spine.set_color("#111111")
    return absolute_extent


def render_project(arrays: dict[tuple[str, str, str], dict[str, np.ndarray]]) -> tuple[dict[str, str], dict[str, object]]:
    panels: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for metric in METRICS:
        for family in FAMILIES:
            train_x, train_y = project_xy(arrays, family, "train", metric)
            test_x, test_y = project_xy(arrays, family, "test", metric)
            panels[(metric, family)] = {
                "train_x": train_x,
                "train_y": train_y,
                "test_x": test_x,
                "test_y": test_y,
            }

    energy_limit = project_limits(arrays, "energy")
    force_limit = (-FORCE_LIMIT, FORCE_LIMIT, -FORCE_LIMIT, FORCE_LIMIT)
    force_limits = {family: force_limit for family in FAMILIES}
    limits = {
        "energy": {family: energy_limit for family in FAMILIES},
        "force": force_limits,
    }
    norms = {}
    for metric in METRICS:
        points = []
        extents = []
        for family in FAMILIES:
            p = panels[(metric, family)]
            points.append((np.concatenate((p["train_x"], p["test_x"])), np.concatenate((p["train_y"], p["test_y"]))))
            extents.append(limits[metric][family])
        norms[metric] = count_norm_by_extent(points, extents, gridsize=48)

    fig, axes = plt.subplots(2, 4, figsize=(25.5, 14.4), sharex=False, sharey=False, squeeze=False)
    for row, metric in enumerate(("energy", "force")):
        norm = norms[metric]
        for col, family in enumerate(FAMILIES):
            ax = axes[row, col]
            p = panels[(metric, family)]
            x0, x1, y0, y1 = limits[metric][family]
            all_x = np.concatenate((p["train_x"], p["test_x"]))
            all_y = np.concatenate((p["train_y"], p["test_y"]))
            draw_hexbin(ax, all_x, all_y, extent=limits[metric][family], norm=norm, gridsize=48)
            tr_x, tr_y = sample_points(p["train_x"], p["train_y"], 2200)
            te_x, te_y = sample_points(p["test_x"], p["test_y"], 2200)
            ax.scatter(tr_x, tr_y, s=11, color="#8d8d8d", alpha=0.28, linewidths=0, zorder=3)
            ax.scatter(te_x, te_y, s=13, color=COLORS[family], alpha=0.92, linewidths=0, zorder=4)
            ax.plot([x0, x1], [x0, x1], linestyle=(0, (6, 4)), color="#cf2e2e", linewidth=2.2, zorder=2)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            if row == 0:
                ax.set_title(f"{chr(65 + col)}  {family}", loc="left", color=COLORS[family], fontsize=PANEL_TITLE, fontweight="bold", pad=12)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_title(f"{chr(69 + col)}  {family}", loc="left", color=COLORS[family], fontsize=PANEL_TITLE, fontweight="bold", pad=12)
            if metric == "energy" and col > 0:
                ax.tick_params(labelleft=False)
            annotation = f"Spearman: {spearman(p['test_x'], p['test_y']):.2f}\nPearson: {pearson(p['test_x'], p['test_y']):.2f}"
            ax.text(
                0.045,
                0.95,
                annotation,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=ANNOTATION,
                linespacing=1.15,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 3.0},
            )
            style_axis(ax)
            if metric == "energy":
                add_energy_inset(ax, p, family)

        # One shared y title per row, and one shared x title per row.  This
        # preserves the distinct energy/force units without repeating labels.
        fig.text(0.5, 0.515 if row == 0 else 0.045, METRICS[metric]["x_label"], ha="center", va="center", fontsize=AXIS_LABEL, fontweight="bold")
        fig.text(0.018, 0.72 if row == 0 else 0.285, METRICS[metric]["y_label"], ha="center", va="center", rotation=90, fontsize=AXIS_LABEL, fontweight="bold")

    handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor="#8d8d8d", markeredgecolor="none", alpha=0.28, markersize=10, label="train"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="#333333", markeredgecolor="none", alpha=0.92, markersize=10, label="held-out test"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, frameon=False, fontsize=LEGEND_SIZE, handletextpad=0.55, columnspacing=1.6)
    fig.text(0.5, 0.985, "DFT reference vs MLIP prediction | train context, held-out test, and count density", ha="center", va="top", fontsize=TITLE_SIZE, fontweight="bold")
    fig.subplots_adjust(left=0.075, right=0.925, bottom=0.12, top=0.91, hspace=0.30, wspace=0.12)
    for row, metric in enumerate(("energy", "force")):
        add_vector_colorbar(fig, axes[row, -1], norms[metric], "Count per hexbin")
    outputs = save_figure(fig, "slide3_parity_density_train_test")
    metadata = {
        "data_root": str(DATA_ROOT),
        "summary_sha256": sha256(SUMMARY_PATH),
        "input_matrix": "24 parity NPZ outputs: 4 families x 3 seeds x 2 splits; family/seed/split parsed from exact eval_id filenames",
        "families": FAMILIES,
        "splits": SPLITS,
        "metric_rows": ["energy", "force"],
        "correlation_basis": "all finite held-out test parity observations pooled across seeds per family",
        "hexbin_basis": "all finite train plus held-out test parity observations pooled across seeds per family",
        "point_overlay": "deterministic evenly spaced vector sample, max 2200 train and max 2200 held-out test points per panel",
        "energy_inset": "per-family local inset centered on the densest held-out test energy core; both axes use hexbin count density and deterministic vector train/test overlays",
        "energy_normalization": {
            "formula": "E_rel = E - E0",
            "reference": "E0 is the median held-out DFT energy within the selected local core for each family",
            "applied_to": "DFT and MLIP axes identically inside the inset",
            "absolute_context": "the main energy panels retain absolute DFT/MLIP energy values",
            "effect_on_error": "translation only; parity residuals, RMSE, Spearman, and Pearson are unchanged",
            "no_synthetic_points": True,
        },
        "force_focus": {
            "method": "shared square limits of -5.25 to +5.25 eV/A for the full force row, retaining the central distribution with light visual whitespace",
            "outside_fraction": {
                family: {
                    split: outside_fraction(
                        panels[("force", family)][f"{split}_x"],
                        panels[("force", family)][f"{split}_y"],
                        limits["force"][family],
                    )
                    for split in SPLITS
                }
                for family in FAMILIES
            },
        },
        "min_text_size_pt": MIN_TEXT,
        "palette": COLORS,
        "outputs": outputs,
    }
    return outputs, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--project-only", action="store_true")
    args = parser.parse_args()
    setup_style()
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "script": str(Path(__file__).resolve()),
        "font": FONT,
        "min_text_size_pt": MIN_TEXT,
        "reference_style_demo": {"data_status": "synthetic deterministic style proof only"},
    }
    if not args.project_only:
        manifest["reference_style_demo"]["outputs"] = render_reference_demo()
    if not args.reference_only:
        arrays = load_project_arrays()
        outputs, metadata = render_project(arrays)
        manifest["slide3_parity_density"] = metadata
    (OUT_ROOT / "slide3_nature_density_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
