#!/usr/bin/env python3
"""Re-render the current Canva MD six-panel figure with the semantic palette.

The source/data contract is the local MD15 R06G renderer. This additive renderer
keeps the screenshot's six-panel content while changing only the visual grammar:
strategy = color, temperature = marker fill, seed = marker shape, and pair type
in the coordination bars = hatch.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


BUNDLE = Path(__file__).resolve().parents[1]
SOURCE_DIR = BUNDLE / "data/source_archive/analysis/md_1ns_updated_publication_figure_20261025"
SCRIPT = SOURCE_DIR / "render_md15_figure_R02.py"
CAMPAIGN = BUNDLE
OUT = CAMPAIGN / "figures/CANVA-P05"
OUT.mkdir(parents=True, exist_ok=True)

SPEC = importlib.util.spec_from_file_location("md15_source", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(SCRIPT)
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)

COLORS = {
    "Foundation": "#8FC7AB",
    "Fine-tuning": "#29A7B6",
    "Scratch": "#327FB1",
    "DFT": "#303840",
}
SEED_MARKERS = {123: "o", 234: "^", 345: "s"}
PAIR_HATCHES = {"Cr-Sb": "", "Cr-Te": "///", "Sb-Te": "..."}
PAIR_LABELS = {"Cr-Sb": "Cr–Sb", "Cr-Te": "Cr–Te", "Sb-Te": "Sb–Te"}

MODEL_IDS = [model_id for model_id, *_ in SOURCE.MODELS]
MODEL_INFO = {row[0]: row for row in SOURCE.MODELS}
STRATEGY_GROUPS = {
    "Foundation": ["Foundation_MPA0", "Foundation_MP0b3", "Foundation_OMAT0"],
    "Fine-tuning": [
        "FT600K_A2_seed123", "FT600K_A2_seed234", "FT600K_A2_seed345",
        "FTMultiT_A2_seed123", "FTMultiT_A2_seed234", "FTMultiT_A2_seed345",
    ],
    "Scratch": [
        "Scratch600K_Lmax1_seed123", "Scratch600K_Lmax1_seed234", "Scratch600K_Lmax1_seed345",
        "ScratchMultiT_Lmax1_seed123", "ScratchMultiT_Lmax1_seed234", "ScratchMultiT_Lmax1_seed345",
    ],
}
BAR_GROUPS = {
    "Foundation": STRATEGY_GROUPS["Foundation"],
    "FT-600K": ["FT600K_A2_seed123", "FT600K_A2_seed234", "FT600K_A2_seed345"],
    "FT-MultiT": ["FTMultiT_A2_seed123", "FTMultiT_A2_seed234", "FTMultiT_A2_seed345"],
    "Scratch-600K": ["Scratch600K_Lmax1_seed123", "Scratch600K_Lmax1_seed234", "Scratch600K_Lmax1_seed345"],
    "Scratch-MultiT": ["ScratchMultiT_Lmax1_seed123", "ScratchMultiT_Lmax1_seed234", "ScratchMultiT_Lmax1_seed345"],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strategy(model_id: str) -> str:
    family = MODEL_INFO[model_id][1]
    if family == "Foundation":
        return "Foundation"
    if family.startswith("FT"):
        return "Fine-tuning"
    return "Scratch"


def temperature(model_id: str) -> str | None:
    family = MODEL_INFO[model_id][1]
    if "600K" in family:
        return "600 K"
    if "MultiT" in family:
        return "Multi-T"
    return None


def marker(model_id: str) -> str:
    family = MODEL_INFO[model_id][1]
    if family == "Foundation":
        return "D"
    return SEED_MARKERS[int(MODEL_INFO[model_id][3])]


def facecolor(model_id: str) -> str:
    family = MODEL_INFO[model_id][1]
    if family == "Foundation":
        return COLORS["Foundation"]
    return "white" if temperature(model_id) == "600 K" else COLORS[strategy(model_id)]


def edgecolor(model_id: str) -> str:
    return COLORS[strategy(model_id)]


def draw_model_line(ax, model_id: str, x, y, *, alpha: float = 0.82, lw: float = 0.9, clip=None, markers: bool = True) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if not len(x):
        return
    if clip is not None:
        y_draw = np.clip(y, clip[0], clip[1])
    else:
        y_draw = y
    kwargs = {
        "color": edgecolor(model_id),
        "lw": lw,
        "alpha": alpha,
        "solid_capstyle": "round",
        "zorder": 2,
    }
    if markers:
        kwargs.update({
            "marker": marker(model_id),
            "markevery": max(1, len(x) // 9),
            "ms": 3.0,
            "mfc": facecolor(model_id),
            "mec": edgecolor(model_id),
            "mew": 0.7,
        })
    ax.plot(x, y_draw, **kwargs)


def style_axes(ax, title, letter, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=13.5, fontweight="bold", pad=10)
    # Keep panel letters above the top tick-label band.  At the previous
    # position the A/B/C glyphs occupied the same vertical band as the top
    # y-tick numbers (800, 2, and 80), which is especially visible after the
    # figure is rasterized for Canva.
    ax.text(-0.115, 1.15, letter, transform=ax.transAxes, fontsize=19, fontweight="bold", va="top", clip_on=False)
    ax.tick_params(which="major", labelsize=10.3, width=1.05, length=4.3, direction="out")
    ax.tick_params(which="minor", width=.75, length=2.4, direction="out")
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.05)
        spine.set_color(COLORS["DFT"])
    ax.grid(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11.6, fontweight="bold", labelpad=5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11.6, fontweight="bold", labelpad=5)


def add_inset(ax, panel: str) -> None:
    position = [0.56, 0.08, 0.38, 0.34] if panel in {"A", "B", "C"} else [0.56, 0.56, 0.38, 0.34]
    inset = ax.inset_axes(position, facecolor="white", zorder=10)
    if panel == "A":
        getter = lambda mid: (SOURCE.load_series(mid)["time_ps"], SOURCE.load_series(mid)["temperature_K"])
        xlim, ylim = (190, 270), (450, 720)
    elif panel == "B":
        getter = lambda mid: (SOURCE.load_series(mid)["time_ps"], SOURCE.load_series(mid)["pressure_GPa"])
        xlim, ylim = (190, 270), (-2.0, 2.0)
    elif panel == "C":
        getter = SOURCE.load_msd
        xlim, ylim = (0, 80), (0, 20)
    else:
        getter = SOURCE.load_vacf
        xlim, ylim = (0, 8), (-0.2, 1.05)
    for model_id in MODEL_IDS:
        x, y = getter(model_id)
        mask = np.isfinite(x) & np.isfinite(y) & (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
        if np.any(mask):
            draw_model_line(inset, model_id, x[mask], y[mask], alpha=.85, lw=.65, markers=True)
    if panel in {"B", "F"}:
        inset.axhline(0, color=COLORS["DFT"], ls=(0, (2, 2)), lw=.55, zorder=0)
    inset.set_xlim(*xlim); inset.set_ylim(*ylim)
    inset.tick_params(which="major", labelsize=7.4, width=.65, length=2.0, direction="out", pad=1.0)
    for text in inset.get_xticklabels() + inset.get_yticklabels():
        text.set_fontweight("bold")
    for spine in inset.spines.values():
        spine.set_linewidth(.75); spine.set_color(COLORS["DFT"])
    inset.grid(False)


def bar_face(group: str) -> str:
    if group in {"FT-600K", "Scratch-600K"}:
        return "white"
    if group == "Foundation":
        return COLORS["Foundation"]
    return COLORS["Fine-tuning"] if group == "FT-MultiT" else COLORS["Scratch"]


def bar_edge(group: str) -> str:
    if group == "Foundation":
        return COLORS["Foundation"]
    return COLORS["Fine-tuning"] if group.startswith("FT") else COLORS["Scratch"]


def render() -> dict[str, object]:
    plt.rcParams.update({
        "font.family": "Arial", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig, aa = plt.subplots(2, 3, figsize=(14.2, 8.25), dpi=260)
    aa = aa.ravel()

    for model_id in MODEL_IDS:
        s = SOURCE.load_series(model_id)
        draw_model_line(aa[0], model_id, s["time_ps"], s["temperature_K"], clip=(250, 820))
    aa[0].axhline(600, color=COLORS["DFT"], ls=(0, (3, 2)), lw=.85)
    aa[0].set_xlim(190, 1010); aa[0].set_ylim(250, 820)
    style_axes(aa[0], "Temperature stability", "A", "Time (ps)", "Temperature (K)"); add_inset(aa[0], "A")

    for model_id in MODEL_IDS:
        s = SOURCE.load_series(model_id)
        draw_model_line(aa[1], model_id, s["time_ps"], s["pressure_GPa"], clip=(-2.0, 2.0))
    aa[1].axhline(0, color=COLORS["DFT"], ls=(0, (3, 2)), lw=.85)
    aa[1].set_xlim(190, 1010); aa[1].set_ylim(-2.0, 2.0)
    style_axes(aa[1], "Pressure stability", "B", "Time (ps)", "Hydrostatic pressure (GPa)"); add_inset(aa[1], "B")

    for model_id in MODEL_IDS:
        x, y = SOURCE.load_msd(model_id)
        draw_model_line(aa[2], model_id, x, y, clip=(0, 80))
    aa[2].set_xlim(0, 800); aa[2].set_ylim(0, 80)
    style_axes(aa[2], "Mean-square displacement", "C", "Lag time (ps)", r"MSD$_{xy}$ ($\AA^2$)"); add_inset(aa[2], "C")

    pair_cutoffs = SOURCE.CN_PAIR_CUTOFFS
    positions = np.arange(len(BAR_GROUPS), dtype=float)
    pair_offsets = np.linspace(-0.22, 0.22, len(pair_cutoffs))
    for offset, (pair, hatch) in zip(pair_offsets, PAIR_HATCHES.items()):
        for idx, (group, model_ids) in enumerate(BAR_GROUPS.items()):
            values = np.asarray([SOURCE.load_partial_cn(mid, pair, pair_cutoffs[pair]) for mid in model_ids], dtype=float)
            values = values[np.isfinite(values)]
            mean = float(np.mean(values)) if len(values) else np.nan
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            aa[3].bar(
                positions[idx] + offset, mean, width=.19, yerr=std, capsize=2.0,
                color=bar_face(group), edgecolor=COLORS["DFT"], hatch=hatch,
                linewidth=.75, error_kw={"ecolor": COLORS["DFT"], "elinewidth": .8, "capthick": .8}, zorder=3,
            )
    aa[3].set_xticks(positions, ["Foundation", "FT-600K", "FT-MultiT", "Scratch-600K", "Scratch-MultiT"], rotation=18, ha="right")
    aa[3].set_xlim(-.55, len(BAR_GROUPS) - .45); aa[3].set_ylim(0, 7.0)
    style_axes(aa[3], "Local coordination number", "D", "Model class", "CN (neighbors/center atom)")
    aa[3].legend(
        handles=[Patch(facecolor="white", edgecolor=COLORS["DFT"], hatch=hatch, label=PAIR_LABELS[pair]) for pair, hatch in PAIR_HATCHES.items()],
        loc="upper left", ncol=3, frameon=False, fontsize=11.6, handlelength=1.25,
        handletextpad=.35, columnspacing=.65, borderaxespad=.2, labelspacing=.2,
    )

    global_curves = {}
    for model_id in MODEL_IDS:
        x, y = SOURCE.load_global_rdf(model_id)
        global_curves[model_id] = (x, y)
        draw_model_line(aa[4], model_id, x, y, alpha=.22, lw=.55, markers=True)
    x0 = next(iter(global_curves.values()))[0]
    for group, model_ids in STRATEGY_GROUPS.items():
        curves = np.asarray([global_curves[mid][1] for mid in model_ids], dtype=float)
        mean = SOURCE.smooth_curve(np.nanmean(curves, axis=0), sigma_bins=1.15)
        std = SOURCE.smooth_curve(np.nanstd(curves, axis=0, ddof=1), sigma_bins=1.15) if curves.shape[0] > 1 else np.zeros_like(mean)
        lower = np.clip(mean - std, 0.0, None); upper = np.clip(mean + std, 0.0, None)
        aa[4].fill_between(x0, lower, upper, color=COLORS[group], alpha=.13, lw=0, zorder=2)
        aa[4].plot(x0, mean, color=COLORS[group], lw=1.65, zorder=3)
    aa[4].axhline(1, color=COLORS["DFT"], ls=(0, (3, 2)), lw=.85, zorder=0)
    aa[4].set_xlim(0, 10); aa[4].set_ylim(0, 8)
    style_axes(aa[4], "All-atom global radial distribution", "E", r"r ($\AA$)", r"$g_{\mathrm{all}}(r)$")

    for model_id in MODEL_IDS:
        x, y = SOURCE.load_vacf(model_id)
        draw_model_line(aa[5], model_id, x, y, clip=(-.35, 1.05))
    aa[5].axhline(0, color=COLORS["DFT"], ls=(0, (3, 2)), lw=.85)
    aa[5].set_xlim(0, 800); aa[5].set_ylim(-.35, 1.05)
    style_axes(aa[5], "Velocity autocorrelation", "F", "Lag time (ps)", "Normalized VACF"); add_inset(aa[5], "F")

    strategy_handles = [Line2D([], [], color=COLORS[name], lw=2.4, label=name) for name in ("Foundation", "Fine-tuning", "Scratch")]
    temp_handles = [
        Line2D([], [], marker="o", linestyle="None", color=COLORS["DFT"], markerfacecolor="white", label="600 K"),
        Line2D([], [], marker="o", linestyle="None", color=COLORS["DFT"], markerfacecolor=COLORS["DFT"], label="Multi-T"),
    ]
    seed_handles = [Line2D([], [], marker=mk, linestyle="None", color=COLORS["DFT"], markerfacecolor="white", label=f"seed {seed}") for seed, mk in SEED_MARKERS.items()]
    foundation_handle = Line2D([], [], marker="D", linestyle="None", color=COLORS["Foundation"], markerfacecolor=COLORS["Foundation"], label="Foundation checkpoint")
    legend_size = 11.6  # match the x-axis title size for final-size readability
    fig.legend(strategy_handles, [h.get_label() for h in strategy_handles], loc="lower center", bbox_to_anchor=(.22, .045), ncol=3, frameon=False, fontsize=legend_size, handlelength=1.9, handletextpad=.45, columnspacing=.85)
    fig.legend(temp_handles, [h.get_label() for h in temp_handles], loc="lower center", bbox_to_anchor=(.52, .045), ncol=2, frameon=False, fontsize=legend_size, handlelength=1.4, handletextpad=.45, columnspacing=.85)
    fig.legend(seed_handles + [foundation_handle], [h.get_label() for h in seed_handles + [foundation_handle]], loc="lower center", bbox_to_anchor=(.82, .045), ncol=4, frameon=False, fontsize=legend_size, handlelength=1.4, handletextpad=.45, columnspacing=.75)
    fig.subplots_adjust(left=.075, right=.99, top=.965, bottom=.285, wspace=.34, hspace=.44)

    stem = OUT / "md_six_panel_palette_updated"
    outputs = {}
    for ext in ("pdf", "png", "svg"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=320 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        outputs[ext] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    plt.close(fig)
    manifest = {
        "figure_id": "CANVA-P05",
        "status": "REPRODUCED_FROM_LOCAL_MD15_R06G_WITH_PALETTE_REVISION",
        "content_match": "Current Canva PDF page 7 / supplied six-panel screenshot",
        "source_script": str(SCRIPT),
        "source_script_sha256": sha256(SCRIPT),
        "source_pdf": str(SOURCE_DIR / "results/fig_md_1ns_updated_A_to_F_15models_R06G.pdf"),
        "source_pdf_sha256": sha256(SOURCE_DIR / "results/fig_md_1ns_updated_A_to_F_15models_R06G.pdf"),
        "panel_order": ["Temperature stability", "Pressure stability", "Mean-square displacement", "Local coordination number", "All-atom global radial distribution", "Velocity autocorrelation"],
        "visual_encoding": {
            "strategy_color": COLORS,
            "temperature_fill": {"600 K": "open", "Multi-T": "filled"},
            "seed_marker": {"123": "circle", "234": "triangle", "345": "square"},
            "foundation_checkpoint": "diamond",
            "pair_category_in_bar_panel": "hatch",
            "reference_lines": COLORS["DFT"],
        },
        "outputs": outputs,
        "boundary": "Raw trajectories and evidence status remain those of the local MD15 R06G package; this revision changes visual encoding and separates Scratch temperature clusters in the coordination bars.",
    }
    manifest_path = CAMPAIGN / "MD_SIX_PANEL_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(render(), indent=2, ensure_ascii=False))
