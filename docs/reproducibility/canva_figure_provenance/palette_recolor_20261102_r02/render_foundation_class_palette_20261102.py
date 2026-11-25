#!/usr/bin/env python3
"""Re-render Foundation class ensembles with one green family and semantic styles."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


CAMPAIGN = Path(__file__).resolve().parent
OUT = CAMPAIGN / "figures"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {"MACE-MP": "#2F6B4F", "MACE-OMAT-0": "#5F9E73", "MACE-MH": "#8FC7AB", "DFT": "#303840"}
MARKERS = {"MACE-MP": "o", "MACE-OMAT-0": "^", "MACE-MH": "s"}
LINESTYLES = {"MACE-MP": "-", "MACE-OMAT-0": "--", "MACE-MH": "-."}


def load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F3 = load_module("foundation_3x3_source", Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/foundation_self_relaxed_4x3_20261024/archive/legacy_figure_products_20261024/build_foundation_class_3x3_r01.py"))
F4 = load_module("foundation_4x3_source", Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/foundation_self_relaxed_4x3_20261024/archive/legacy_figure_products_20261024/build_foundation_class_4x3_r03.py"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_tsv(path: Path):
    import csv
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        for key in ("image_index", "n_models"):
            if key in row and row[key] != "":
                row[key] = int(row[key])
        for key in ("s", "mean_relative_energy_eV", "sd_relative_energy_eV", "relative_energy_eV"):
            if key in row and row[key] not in ("", "None", None):
                row[key] = float(row[key])
    return rows


def axes_style(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(COLORS["DFT"])
        spine.set_linewidth(1.05)
    ax.tick_params(direction="out", width=1.0, length=4.0)
    ax.minorticks_on()


def class_rows(source, element, path_key, grouped):
    result = {}
    for model_class in source.CLASS_ORDER:
        values = [r for r in grouped if r["element"] == element and r["path_key"] == path_key and r["model_class"] == model_class]
        values.sort(key=lambda row: row["image_index"])
        result[model_class] = values
    return result


def draw_panel(ax, source, element, path_key, grouped, qe_rows, panel_label):
    axes_style(ax)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1], ["0", ".25", ".5", ".75", "1"])
    ax.text(-.16, 1.06, panel_label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
    bands = []
    for model_class in source.CLASS_ORDER:
        values = class_rows(source, element, path_key, grouped)[model_class]
        x = np.asarray([float(r["s"]) for r in values]); y = np.asarray([float(r["mean_relative_energy_eV"]) for r in values])
        sd = np.asarray([0.0 if r.get("sd_relative_energy_eV") is None else float(r["sd_relative_energy_eV"]) for r in values])
        c = COLORS[model_class]
        ax.fill_between(x, y - sd, y + sd, color=c, alpha=.20, linewidth=0, zorder=2)
        ax.plot(x, y, color=c, lw=2.0, ls=LINESTYLES[model_class], marker=MARKERS[model_class], markersize=4.5, markerfacecolor="white", markeredgecolor=c, markeredgewidth=.95, zorder=5)
        bands.extend([y - sd, y + sd])
    qe = sorted([r for r in qe_rows if r["element"] == element and r["path_key"] == path_key], key=lambda row: row["image_index"])
    if qe:
        x = np.asarray([float(r["s"]) for r in qe]); y = np.asarray([float(r["relative_energy_eV"]) for r in qe])
        ax.plot(x, y, color=COLORS["DFT"], lw=1.55, ls=(0, (4, 2)), marker="D", markersize=3.8, markerfacecolor="white", markeredgecolor=COLORS["DFT"], markeredgewidth=.85, zorder=6)
        bands.append(y)
    all_values = np.concatenate(bands)
    lo, hi = float(np.nanmin(all_values)), float(np.nanmax(all_values)); pad = max(.06 * (hi - lo), .04)
    ax.set_ylim(lo - pad, hi + pad)


def render(source, model_rows, grouped, qe_rows, stem_name):
    elements = source.ELEMENTS
    paths = source.PATHS
    fig, axs = plt.subplots(len(elements), len(paths), figsize=(10.8, 2.95 * len(elements)), squeeze=False)
    labels = iter("abcdefghijklmnopqrstuvwxyz")
    for i, element in enumerate(elements):
        for j, path_info in enumerate(paths):
            path_key, display = path_info[:2]
            ax = axs[i, j]
            draw_panel(ax, source, element, path_key, grouped, qe_rows, next(labels))
            if i == 0:
                ax.set_title(display, fontsize=12, fontweight="bold", pad=8)
            if j == 0:
                ax.set_ylabel(f"{element}\nΔE (eV)", fontsize=11, fontweight="bold", labelpad=8)
            if i == len(elements) - 1:
                ax.set_xlabel("normalized image coordinate, $s$", fontsize=10.8, labelpad=7)
    handles = [Line2D([], [], color=COLORS[c], lw=2.0, ls=LINESTYLES[c], marker=MARKERS[c], markerfacecolor="white", markeredgecolor=COLORS[c], markeredgewidth=.95, label=f"{c} (n=5 / 5)" if c == "MACE-MP" else f"{c} (n=2 / 2)") for c in source.CLASS_ORDER]
    handles.append(Line2D([], [], color=COLORS["DFT"], lw=1.55, ls=(0, (4, 2)), marker="D", markersize=3.8, markerfacecolor="white", markeredgecolor=COLORS["DFT"], label="DFT-NEB"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .985), ncol=4, frameon=False, handlelength=2.2, columnspacing=1.2, handletextpad=.45)
    fig.subplots_adjust(left=.095, right=.985, bottom=.075, top=.88 if len(elements) == 3 else .90, wspace=.34, hspace=.43)
    stem = OUT / stem_name
    for ext in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=350 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem


def data_3x3():
    base = Path(F3.__file__).parent / "data" / "foundation_class_3x3_R03"
    model_rows = read_tsv(base / "foundation_model_profiles_9models_3x3_R03.tsv")
    grouped = read_tsv(base / "foundation_model_class_profiles_3x3_R03.tsv")
    qe_rows = read_tsv(base / "foundation_model_qe_reference_profiles_3x3_R03.tsv")
    return F3, model_rows, grouped, qe_rows, Path(F3.__file__)


def data_4x3():
    r01 = F4.R02.R01
    base = Path(F4.__file__).parent / "data" / "foundation_class_4x3_R03"
    model_rows = read_tsv(base / "foundation_model_profiles_9models_4x3_R03.tsv")
    grouped = read_tsv(base / "foundation_model_class_profiles_4x3_R03.tsv")
    qe_rows = read_tsv(base / "foundation_model_qe_reference_profiles_4x3_R03.tsv")
    return r01, model_rows, grouped, qe_rows, Path(F4.__file__)


def main():
    source3, rows3, grouped3, qe3, script3 = data_3x3()
    source4, rows4, grouped4, qe4, script4 = data_4x3()
    stems = {"CANVA-P07": render(source3, rows3, grouped3, qe3, "CANVA-P07_foundation_class_3x3_palette_updated"), "CANVA-P08": render(source4, rows4, grouped4, qe4, "CANVA-P08_foundation_class_4x3_palette_updated")}
    manifest = {
        "status": "REPRODUCED_FROM_LOCAL_FOUNDATION_CLASS_TABLES_WITH_PALETTE_REVISION",
        "visual_encoding": {"foundation_class_colors": COLORS, "foundation_class_marker": {"MACE-MP": "circle", "MACE-OMAT-0": "triangle", "MACE-MH": "square"}, "foundation_class_line": {"MACE-MP": "solid", "MACE-OMAT-0": "dashed", "MACE-MH": "dash-dot"}, "reference": COLORS["DFT"]},
        "sources": {"CANVA-P07": {"script": str(script3), "sha256": sha256(script3), "rows": len(rows3), "class_rows": len(grouped3), "qe_rows": len(qe3)}, "CANVA-P08": {"script": str(script4), "sha256": sha256(script4), "rows": len(rows4), "class_rows": len(grouped4), "qe_rows": len(qe4)}},
        "outputs": {key: {ext: {"path": str(stem.with_suffix(f".{ext}")), "sha256": sha256(stem.with_suffix(f".{ext}")), "bytes": stem.with_suffix(f".{ext}").stat().st_size} for ext in ("pdf", "svg", "png")} for key, stem in stems.items()},
    }
    (CAMPAIGN / "FOUNDATION_CLASS_PALETTE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
