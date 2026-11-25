#!/usr/bin/env python3
"""Render an evidence-aware 1 ns MACE-MD benchmark figure.

This is a one-off manuscript figure built from the audited 1 ns analysis
exports.  It deliberately does not manufacture HFACF/thermal-conductivity
curves: the four-model 1 ns archive contains no heat-flux time series.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

PLOT_LIBRARY_SRC = Path("/Users/alina/Documents/Codex/plot_library/src")
if str(PLOT_LIBRARY_SRC) not in sys.path:
    sys.path.insert(0, str(PLOT_LIBRARY_SRC))
from plot_atlas.palettes import get_palette  # noqa: E402


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = get_palette("nature_steered_dynamics")
INK = PALETTE["ink"]
MUTED = "#65717C"
TARGET = "#5E6572"
FAIL = "#B44731"
FAIL_2 = "#D29A2C"
STABLE_1 = PALETTE["roles"]["ml_global"]
STABLE_2 = PALETTE["roles"]["ml_r06"]
NEUTRAL = "#D9DDE1"
PALE_FAIL = "#F8ECE8"
PALE_BLUE = "#EAF2F8"

MODELS = {
    "foundation-omat": {"label": "Foundation-OMAT", "color": FAIL, "kind": "failure"},
    "from-scratch": {"label": "From-scratch", "color": FAIL_2, "kind": "failure"},
    "2-naive-fine-tuning": {"label": "FT-600K", "color": STABLE_1, "kind": "stable"},
    "3-multi_T-fine-tuning": {"label": "FT-MultiT", "color": STABLE_2, "kind": "stable"},
}


def read_csv(model: str, name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / model / name)


def read_json(model: str, name: str) -> dict:
    return json.loads((DATA / model / name).read_text())


def style_ax(ax, *, labelsize: float = 7.0) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(0.8)
    ax.tick_params(direction="out", length=3.2, width=0.8, colors=INK, labelsize=labelsize)
    for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick.set_fontweight("bold")
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def panel_letter(ax, letter: str) -> None:
    ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left", color=INK)


def rounded_box(ax, x: float, y: float, w: float, h: float, text: str,
                face: str, edge: str, text_color: str = INK, fs: float = 7.0) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=face, edgecolor=edge, linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=text_color, fontweight="bold", linespacing=1.15)


def panel_a(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.01, 0.98, "Unified 1 ns evaluation protocol", va="top", fontsize=8.7,
            fontweight="bold", color=INK)
    ax.text(0.99, 0.98, "dt = 1 fs  ·  2050 atoms  ·  600 K  ·  NVT Langevin",
            va="top", ha="right", fontsize=6.3, color=MUTED)

    xs = [0.025, 0.215, 0.405, 0.595]
    for x, (model, spec) in zip(xs, MODELS.items()):
        face = PALE_FAIL if spec["kind"] == "failure" else PALE_BLUE
        rounded_box(ax, x, 0.66, 0.16, 0.14, spec["label"], face, spec["color"], fs=6.3)
        ax.plot([x + 0.08, x + 0.08], [0.66, 0.52], color=spec["color"], lw=0.9)

    rounded_box(ax, 0.28, 0.34, 0.44, 0.18, "1 ns molecular dynamics\ncheckpointed trajectory",
                "#FFFFFF", INK, fs=8.0)
    for x in [0.105, 0.295, 0.485, 0.675]:
        ax.add_patch(FancyArrowPatch((x, 0.52), (0.5, 0.52), arrowstyle="-",
                                     lw=0.8, color=MUTED))
    ax.plot([0.50, 0.50], [0.34, 0.23], color=INK, lw=0.9)
    ax.plot([0.14, 0.86], [0.23, 0.23], color=INK, lw=0.9)
    outcomes = [
        (0.14, "trajectory\nstability", "finite T/F/geometry", STABLE_1),
        (0.38, "local structure", "minimum distance", "#7A6F58"),
        (0.62, "diffusion", "MSD → Dxy", STABLE_2),
        (0.86, "dynamics", "VACF", "#6C5A91"),
    ]
    for x, title, sub, color in outcomes:
        ax.plot([x, x], [0.23, 0.15], color=color, lw=0.9)
        ax.text(x, 0.105, title, ha="center", va="center", fontsize=6.8,
                fontweight="bold", color=INK)
        ax.text(x, 0.045, sub, ha="center", va="center", fontsize=5.8, color=MUTED)
    ax.text(0.995, 0.22, "HFACF/κ requires an archived heat-flux stream",
            ha="right", va="bottom", fontsize=5.8, color=FAIL, style="italic")
    panel_letter(ax, "a")


def plot_stream(ax, model: str, df: pd.DataFrame, column: str, **kwargs) -> None:
    spec = MODELS[model]
    ax.plot(df["time_ps"], df[column], color=spec["color"], label=spec["label"], **kwargs)


def panel_b(fig, spec) -> None:
    inner = spec.subgridspec(2, 1, height_ratios=[1.02, 1.0], hspace=0.40)
    ax_t = fig.add_subplot(inner[0])
    ax_f = fig.add_subplot(inner[1], sharex=ax_t)
    panel_letter(ax_t, "b")
    ax_t.set_title("Thermal and force stability", loc="left", fontsize=8.7,
                   fontweight="bold", pad=5, color=INK)
    ax_t.set_yscale("log")
    ax_t.set_ylim(2.0e2, 2.0e16)
    ax_t.set_xlim(0, 1000)
    for model in MODELS:
        if model == "from-scratch":
            df = read_csv("1-from-scratch", "failure_scan.csv")
        else:
            df = read_csv(model, "timeseries.csv")
        plot_stream(ax_t, model, df, "temperature_K", lw=0.85 if MODELS[model]["kind"] == "failure" else 1.0,
                    alpha=0.72)
    ax_t.axhline(600, color=TARGET, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax_t.axvline(8.8, color=FAIL, lw=0.8, ls=(0, (3, 2)))
    ax_t.axvline(462.2, color=FAIL_2, lw=0.8, ls=(0, (3, 2)))
    ax_t.text(8.8, 1.2e16, "8.8 ps", color=FAIL, fontsize=6.1, ha="left", va="top")
    ax_t.text(462.2, 1.2e16, "462.2 ps", color=FAIL_2, fontsize=6.1, ha="center", va="top")
    ax_t.set_ylabel("Temperature (K)", fontsize=7.3)
    ax_t.set_xticklabels([])
    style_ax(ax_t)

    inset = inset_axes(ax_t, width="39%", height="38%", loc="lower right", borderpad=1.0)
    for model in MODELS:
        if model == "from-scratch":
            df = read_csv("1-from-scratch", "failure_scan.csv")
        else:
            df = read_csv(model, "timeseries.csv")
        short = df[df.time_ps <= 1000]
        inset.plot(short.time_ps, short.temperature_K, color=MODELS[model]["color"], lw=0.55, alpha=0.8)
    inset.axhline(600, color=TARGET, lw=0.55, ls=(0, (2, 2)))
    inset.set_xlim(0, 1000); inset.set_ylim(240, 720)
    inset.set_xticks([0, 500, 1000]); inset.set_yticks([300, 600])
    inset.tick_params(labelsize=4.8, length=2, width=0.6)
    for s in inset.spines.values(): s.set_linewidth(0.6)
    inset.set_title("stable range", fontsize=5.1, color=MUTED, pad=1.5)

    ax_f.set_yscale("log")
    ax_f.set_ylim(0.25, 2.0e6)
    for model in MODELS:
        if model == "from-scratch":
            df = read_csv("1-from-scratch", "failure_scan.csv")
        else:
            df = read_csv(model, "timeseries.csv")
        plot_stream(ax_f, model, df, "max_force_eV_A", lw=0.85 if MODELS[model]["kind"] == "failure" else 1.0,
                    alpha=0.72)
    ax_f.axhline(100, color=FAIL, lw=0.8, ls=(0, (3, 2)))
    ax_f.axvline(8.8, color=FAIL, lw=0.8, ls=(0, (3, 2)))
    ax_f.axvline(462.2, color=FAIL_2, lw=0.8, ls=(0, (3, 2)))
    ax_f.set_ylabel("Max force (eV Å$^{-1}$)", fontsize=7.3)
    ax_f.set_xlabel("Time (ps)", fontsize=7.3)
    ax_f.text(0.99, 0.05, "stable endpoint d$_{min}$: 2.29 Å / 1.63 Å",
              transform=ax_f.transAxes, ha="right", va="bottom", fontsize=5.6, color=MUTED)
    style_ax(ax_f)
    handles = [Line2D([], [], color=MODELS[m]["color"], lw=1.3, label=MODELS[m]["label"]) for m in MODELS]
    ax_t.legend(handles=handles, frameon=False, fontsize=5.8, ncol=2, loc="upper left",
                handlelength=1.5, columnspacing=0.9, handletextpad=0.35)


def fit_window(df: pd.DataFrame, lo: float, hi: float) -> tuple[float, float]:
    q = df[(df.lag_ps >= lo) & (df.lag_ps <= hi)].dropna()
    slope, intercept = np.polyfit(q.lag_ps.to_numpy(), q.msd_A2.to_numpy(), 1)
    return float(slope), float(intercept)


def panel_c(ax) -> None:
    panel_letter(ax, "c")
    ax.set_title("Cr mobility over the 1 ns trajectories", loc="left", fontsize=8.7,
                 fontweight="bold", pad=5, color=INK)
    ax.set_xlim(0, 1000)
    ax.set_ylim(-0.5, 48)
    ax.axvspan(200, 500, color="#E8EFF5", alpha=0.75, lw=0)
    ax.axvspan(500, 900, color="#F2F4F5", alpha=0.9, lw=0)
    for model in ["2-naive-fine-tuning", "3-multi_T-fine-tuning"]:
        df = read_csv(model, "msd.csv")
        spec = MODELS[model]
        ax.plot(df.lag_ps, df.msd_A2, color=spec["color"], lw=0.45, alpha=0.18)
        smooth = df.msd_A2.rolling(41, center=True, min_periods=1).mean()
        ax.plot(df.lag_ps, smooth, color=spec["color"], lw=1.55, label=spec["label"])
        for lo, hi in [(200, 500), (500, 900)]:
            slope, intercept = fit_window(df, lo, hi)
            xx = np.array([lo, hi])
            ax.plot(xx, slope * xx + intercept, color=spec["color"], lw=0.85, ls=(0, (3, 2)))
    ax.text(350, 45.2, "fit window", ha="center", va="top", fontsize=5.8, color=MUTED)
    ax.text(700, 45.2, "fit window", ha="center", va="top", fontsize=5.8, color=MUTED)
    ax.set_xlabel("Lag time (ps)", fontsize=7.3)
    ax.set_ylabel("Cr MSD (Å$^2$)", fontsize=7.3)
    ax.legend(frameon=False, fontsize=6.2, loc="upper left", handlelength=1.5)
    ax.text(0.02, 0.04, "Foundation-OMAT / From-scratch: unstable — no D$_{xy}$ reported",
            transform=ax.transAxes, fontsize=5.8, color=FAIL, ha="left", va="bottom")
    style_ax(ax)


def panel_d(ax) -> None:
    panel_letter(ax, "d")
    ax.set_title("Short-time velocity dynamics", loc="left", fontsize=8.7,
                 fontweight="bold", pad=5, color=INK)
    ax.set_xlim(0, 50)
    ax.set_ylim(-0.08, 1.02)
    for model in ["2-naive-fine-tuning", "3-multi_T-fine-tuning"]:
        df = read_csv(model, "vacf.csv")
        q = df[df.lag_ps <= 50]
        spec = MODELS[model]
        ax.plot(q.lag_ps, q.vacf_normalized, color=spec["color"], lw=0.45, alpha=0.20)
        smooth = q.vacf_normalized.rolling(21, center=True, min_periods=1).mean()
        ax.plot(q.lag_ps, smooth, color=spec["color"], lw=1.45, label=spec["label"])
    ax.axhline(0, color=TARGET, lw=0.75, ls=(0, (3, 2)))
    ax.set_xlabel("Lag time (ps)", fontsize=7.3)
    ax.set_ylabel("Normalized VACF", fontsize=7.3)
    ax.legend(frameon=False, fontsize=6.2, loc="upper right", handlelength=1.5)
    ax.text(0.02, 0.87, "Only scientifically stable trajectories are shown",
            transform=ax.transAxes, fontsize=5.8, color=MUTED)
    style_ax(ax)


def panel_e(ax) -> None:
    panel_letter(ax, "e")
    ax.set_title("Diffusion coefficient and block diagnostics", loc="left", fontsize=8.7,
                 fontweight="bold", pad=5, color=INK)
    ax.set_xscale("symlog", linthresh=0.01, linscale=0.9)
    ax.set_xlim(-0.08, 1.65)
    ax.axvline(0, color=TARGET, lw=0.7, ls=(0, (3, 2)))
    rows = [("2-naive-fine-tuning", 1.0), ("3-multi_T-fine-tuning", 0.0)]
    windows = [("200–500 ps", "200_500_ps", "o", 0.0), ("500–900 ps", "500_900_ps", "s", 0.20)]
    for model, y in rows:
        metrics = read_json(model, "metrics.json")
        block = metrics["block_slopes"]
        for win_label, key, marker, dy in windows:
            full = metrics["Dxy"][key]["D_cm2_s"] * 1e6
            finite = [r["D_cm2_s"] * 1e6 for r in block if r["window"] == key and r.get("D_cm2_s") is not None and r.get("n", 0) >= 500]
            yy = y + (dy - 0.10)
            if finite:
                ax.scatter(finite, [yy] * len(finite), s=14, marker=marker,
                           facecolor="white", edgecolor=MODELS[model]["color"], linewidth=0.75,
                           alpha=0.9, zorder=3)
            ax.scatter([full], [y + dy], s=50, marker="D", color=MODELS[model]["color"],
                       edgecolor="white", linewidth=0.65, zorder=4)
    ax.set_yticks([0.10, 1.10], ["FT-600K", "FT-MultiT"])
    ax.set_xlabel(r"$D_{xy}$ ($10^{-6}$ cm$^2$ s$^{-1}$)", fontsize=7.3)
    ax.set_ylim(-0.35, 1.55)
    ax.text(0.02, 0.02, "Foundation-OMAT / From-scratch: unstable — no D$_{xy}$ reported",
            transform=ax.transAxes, fontsize=5.8, color=FAIL)
    ax.text(0.02, -0.19, "HFACF and κ: not reported; no heat-flux time series in the archived 1 ns outputs",
            transform=ax.transAxes, fontsize=5.7, color=MUTED, style="italic")
    legend_handles = [
        Line2D([], [], marker="D", color="none", markerfacecolor=INK, markeredgecolor="white",
               markersize=5.5, label="full-window fit"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=INK,
               markersize=4.0, label="block fits"),
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=5.8, loc="upper left",
              handletextpad=0.35, labelspacing=0.25)
    style_ax(ax)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_artifact_manifest() -> None:
    files = {}
    for p in sorted(DATA.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(HERE))] = sha256(p)
    payload = {
        "figure": "mace_1ns_evidence_aware_benchmark",
        "source_root": "/scratch16/pclancy3/yi/mace_four_model_MD_analysis_20261003",
        "protocol": {"dt_fs": 1.0, "temperature_K": 600.0, "natoms": 2050, "target_step": 1000000},
        "scientific_status": {
            "stable_for_diffusion": ["FT-600K", "FT-MultiT"],
            "failure_mode": {
                "Foundation-OMAT": "first thermal threshold crossing at 8.8 ps",
                "From-scratch": "first instability at 462.2 ps",
            },
            "hfacf_status": "not available from archived four-model 1 ns outputs",
        },
        "input_sha256_local_exports": files,
        "source_audits": {
            model: read_json(model, "audit.json") for model in [
                "foundation-omat", "1-from-scratch", "2-naive-fine-tuning", "3-multi_T-fine-tuning"
            ]
        },
    }
    (HERE / "figure_manifest.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def write_caption() -> None:
    caption = r"""# Figure | Evidence-aware 1 ns benchmarking of four MACE models

**a, Unified protocol.** Four candidate models drove the same 2050-atom Cr--Sb$_2$Te$_3$ system at 600 K with a 1 fs Langevin trajectory to 1,000,000 steps. The archived trajectories were audited for finite energies, forces, temperatures, periodic cells, and geometry before downstream analysis. **b, Stability and integrity.** FT-600K and FT-MultiT remain near the 600 K target with bounded forces across 1 ns. Foundation-OMAT crosses the thermal failure threshold at 8.8 ps, whereas From-scratch remains near the target until its first reproducible instability at 462.2 ps. The failure trajectories are retained as formal failure-mode benchmarks. **c, Diffusion.** The Cr mean-squared displacement and linear-fit windows are shown only for the scientifically stable trajectories. Foundation-OMAT and From-scratch are explicitly withheld from diffusion reporting. **d, Velocity dynamics.** The normalized VACF is reported for the stable trajectories over the first 50 ps. **e, D$_{xy}$ and uncertainty.** Diamonds are full-window estimates; open symbols are block fits retained for uncertainty diagnostics. **HFACF/κ are not reported** because the four-model 1 ns archive does not contain a heat-flux time series; those quantities require a separate, explicitly instrumented production run.

The figure is therefore a direct 1 ns update of the original conceptual figure, but it does not imply that a completed wall-clock trajectory is scientifically stable or that an unavailable observable has been reconstructed.
"""
    (HERE / "figure_caption.md").write_text(caption)


def main() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(7.45, 9.2), facecolor="white")
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.05, 1.75, 1.15], width_ratios=[1, 1],
        left=0.085, right=0.975, bottom=0.105, top=0.95, wspace=0.34, hspace=0.56,
    )
    a = fig.add_subplot(gs[0, :])
    panel_a(a)
    panel_b(fig, gs[1, 0])
    c = fig.add_subplot(gs[1, 1])
    panel_c(c)
    d = fig.add_subplot(gs[2, 0])
    panel_d(d)
    e = fig.add_subplot(gs[2, 1])
    panel_e(e)
    fig.suptitle("A 1 ns benchmark separates dynamical stability from model completion",
                 x=0.085, y=0.988, ha="left", fontsize=11.2, fontweight="bold", color=INK)
    fig.text(0.085, 0.968, "Audited MigrationBench MACE-MD trajectories; failure modes are retained rather than hidden",
             ha="left", va="top", fontsize=7.2, color=MUTED)
    for ext, kwargs in [("pdf", {}), ("svg", {}), ("png", {"dpi": 500})]:
        fig.savefig(OUT / f"mace_1ns_evidence_aware_benchmark.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    write_artifact_manifest()
    write_caption()


if __name__ == "__main__":
    main()
