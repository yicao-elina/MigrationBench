#!/usr/bin/env python3
"""Re-render the local 2×4 parity and frame-RMSE figures with temperature fill."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


BUNDLE = Path(__file__).resolve().parents[1]
ROOT = BUNDLE / "data/source_archive/analysis/rmse_train_test_20261026"
SCRIPT = ROOT / "make_figures.py"
CAMPAIGN = BUNDLE
OUT = CAMPAIGN / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SPEC = importlib.util.spec_from_file_location("rmse_source", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(SCRIPT)
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)

MODEL_ORDER = ["Scratch-600K", "FT-600K", "Scratch-MultiT", "FT-MultiT"]
MODEL_COLOR = {"Scratch-600K": "#327FB1", "Scratch-MultiT": "#327FB1", "FT-600K": "#29A7B6", "FT-MultiT": "#29A7B6"}
SEED_MARKERS = {"123": "o", "234": "^", "345": "s"}
SEEDS = ["123", "234", "345"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def filled(model: str) -> bool:
    return "MultiT" in model


def marker_face(model: str) -> str:
    return MODEL_COLOR[model] if filled(model) else "white"


def style_axes(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#303840"); spine.set_linewidth(1.25)
    ax.tick_params(direction="out", width=1.2, length=4.0)


def common_handles():
    strategy = [
        Line2D([], [], marker="o", linestyle="None", color="#327FB1", markerfacecolor="white", label="Scratch"),
        Line2D([], [], marker="o", linestyle="None", color="#29A7B6", markerfacecolor="white", label="Fine-tuning"),
    ]
    temperature = [
        Line2D([], [], marker="o", linestyle="None", color="#303840", markerfacecolor="white", label="600 K"),
        Line2D([], [], marker="o", linestyle="None", color="#303840", markerfacecolor="#303840", label="Multi-T"),
    ]
    seed = [Line2D([], [], marker=mk, linestyle="None", color="#303840", markerfacecolor="white", label=f"seed {s}") for s, mk in SEED_MARKERS.items()]
    split = [
        Line2D([], [], marker="o", linestyle="None", color="#303840", alpha=.25, label="Train", markersize=6),
        Line2D([], [], marker="o", linestyle="None", color="#303840", alpha=.75, label="Test", markersize=6),
    ]
    return strategy, temperature, seed, split


def parity() -> Path:
    rows, arrays = SOURCE.load()
    by = {(r["model"], r["seed"], r["split"]): r for r in rows}

    def xy(model, seed, split, metric):
        d = arrays[(model, seed, split)]
        if metric == "energy":
            return d["energy_ref_eV_per_atom"], d["energy_pred_eV_per_atom"]
        return d["force_ref_eV_A"], d["force_pred_eV_A"]

    limits = {}
    for metric in ("energy", "force"):
        vals = [v for model in MODEL_ORDER for seed in SEEDS for split in ("train", "test") for v in xy(model, seed, split, metric)]
        lo, hi = float(np.nanmin(np.concatenate(vals))), float(np.nanmax(np.concatenate(vals)))
        pad = .04 * (hi - lo if hi > lo else 1.0)
        limits[metric] = (lo - pad, hi + pad)

    fig, axs = plt.subplots(2, 4, figsize=(18.5, 8.8), squeeze=False)
    fig.subplots_adjust(left=.065, right=.995, bottom=.14, top=.88, hspace=.30, wspace=.25)
    for i, model in enumerate(MODEL_ORDER):
        color = MODEL_COLOR[model]
        temp = "Multi-T" if filled(model) else "600 K"
        for j, (metric, xlabel, ylabel) in enumerate((("energy", "DFT energy / atom (eV)", "MLIP energy / atom (eV)"), ("force", "DFT force component (eV Å$^{-1}$)", "MLIP force component (eV Å$^{-1}$)"))):
            ax = axs[j, i]; style_axes(ax)
            lo, hi = limits[metric]
            for seed in SEEDS:
                for split, alpha, size in (("train", .18, 7.0), ("test", .68, 9.0)):
                    x, y = xy(model, seed, split, metric)
                    x, y = SOURCE.sample_for_plot(x, y, limit=12000)
                    ax.scatter(x, y, s=size, marker=SEED_MARKERS[seed], facecolors=marker_face(model), edgecolors=color, linewidths=.42, alpha=alpha, rasterized=False, zorder=2 if split == "train" else 3)
            ax.plot([lo, hi], [lo, hi], "--", color="#303840", lw=1.25, zorder=1)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(xlabel, fontweight="bold"); ax.set_ylabel(ylabel, fontweight="bold")
            if j == 0:
                ax.set_title(f"{chr(97+i)}) {model} · {temp}", loc="left", color=color, pad=8)
            k = "energy_r2" if metric == "energy" else "force_r2"
            te = [by[(model, s, "test")][k] for s in SEEDS]
            annotation = f"Test R² {np.mean(te):.3f}±{np.std(te, ddof=1):.3f}"
            if metric == "energy":
                bias = [by[(model, s, "test")]["energy_bias_meV_per_atom"] for s in SEEDS]
                annotation += f"\nBias {np.mean(bias):+.1f} meV atom$^{{-1}}$"
            ax.text(.04, .96, annotation, transform=ax.transAxes, va="top", ha="left", fontsize=10.4, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .82, "pad": 2.8})
    strategy, temperature, seed, split = common_handles()
    fig.legend(strategy + temperature + seed + split, [h.get_label() for h in strategy + temperature + seed + split], loc="upper center", bbox_to_anchor=(.5, .955), frameon=False, ncol=6, fontsize=11.5, handletextpad=.45, columnspacing=1.0)
    fig.text(.5, .99, "DFT reference vs MLIP prediction: train context and test performance", ha="center", va="top", fontsize=15, fontweight="bold")
    path = OUT / "CANVA-P03_parity_2x4_palette_updated"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=320 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def frame_distributions() -> Path:
    rows = SOURCE.load_summaries()
    by = {(r["model"], r["seed"], r["split"]): r for r in rows}
    frame_values = {}
    for model in SOURCE.MODELS:
        for seed in SOURCE.SEEDS:
            for split in ("train", "test"):
                p = SOURCE.RAW / f"{by[(model, seed, split)]['eval_id']}__per_frame.csv"
                with p.open() as handle:
                    records = list(__import__("csv").DictReader(handle))
                frame_values[(model, seed, split, "energy_abs_error_meV_per_atom")] = np.asarray([float(r["energy_abs_error_meV_per_atom"]) for r in records], dtype=float)
                frame_values[(model, seed, split, "force_rmse_meV_A")] = np.asarray([float(r["force_rmse_meV_A"]) for r in records], dtype=float)
    metric_limits = {}
    for metric in ("energy_abs_error_meV_per_atom", "force_rmse_meV_A"):
        values = np.concatenate([v for k, v in frame_values.items() if k[-1] == metric])
        positive = values[np.isfinite(values) & (values > 0)]
        metric_limits[metric] = (float(np.min(positive)) * .65, float(np.max(positive)) * 1.55)

    fig = plt.figure(figsize=(13.0, 9.8))
    outer = fig.add_gridspec(2, 2, left=.10, right=.985, bottom=.075, top=.79, wspace=.24, hspace=.42)
    for i, model in enumerate(SOURCE.MODELS):
        sub = outer[divmod(i, 2)].subgridspec(2, 1, hspace=.17)
        color = MODEL_COLOR[model]
        for j, (metric, ylabel, summary_key) in enumerate((("energy_abs_error_meV_per_atom", "|ΔE| / atom\n(meV)", "energy_rmse_meV_per_atom"), ("force_rmse_meV_A", "Frame force RMSE\n(meV Å$^{-1}$)", "force_rmse_meV_A"))):
            ax = fig.add_subplot(sub[j]); style_axes(ax)
            for xpos, split in enumerate(("train", "test")):
                vals = np.concatenate([frame_values[(model, seed, split, metric)] for seed in SOURCE.SEEDS])
                ax.boxplot(vals, positions=[xpos], widths=.25, showfliers=False, patch_artist=True, boxprops={"facecolor": marker_face(model), "edgecolor": color, "linewidth": 1.2}, medianprops={"color": "#303840", "linewidth": 1.3}, whiskerprops={"color": color}, capprops={"color": color})
                for seed in SOURCE.SEEDS:
                    values = frame_values[(model, seed, split, metric)]
                    jitter = np.linspace(-.12, .12, len(values)) if len(values) else np.array([])
                    ax.scatter(np.full(len(values), xpos) + jitter, values, s=3.5, marker=SEED_MARKERS[seed], facecolors=marker_face(model), edgecolors=color, alpha=.26, linewidths=.15, zorder=2, rasterized=False)
                rmses = [by[(model, seed, split)][summary_key] for seed in SOURCE.SEEDS]
                ax.scatter([xpos], [float(np.mean(rmses))], marker="D", s=35, facecolor="white", edgecolor="#303840", linewidth=1.1, zorder=4)
            ax.set_xticks([0, 1], ["Train", "Test"]); ax.set_ylabel(ylabel, fontweight="bold"); ax.set_yscale("log"); ax.set_ylim(*metric_limits[metric])
            if j == 0:
                ax.set_title(f"{chr(97+i)}) {model} · {'Multi-T' if filled(model) else '600 K'}", loc="left", color=color, pad=8); ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Split", fontweight="bold")
    strategy, temperature, seed, _split = common_handles()
    mean_handle = Line2D([], [], marker="D", linestyle="None", markerfacecolor="white", markeredgecolor="#303840", markersize=7, label="mean global RMSE")
    fig.legend(strategy + temperature + seed + [mean_handle], [h.get_label() for h in strategy + temperature + seed + [mean_handle]], loc="upper center", bbox_to_anchor=(.5, .905), frameon=False, ncol=6, fontsize=10.5, handletextpad=.45, columnspacing=1.0)
    fig.suptitle("Frame-level error distributions with all train/test AIMD frames", y=.985, fontsize=14.5, fontweight="bold")
    path = OUT / "CANVA-P04_frame_rmse_palette_updated"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=320 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main():
    SOURCE.style()
    p03 = parity(); p04 = frame_distributions()
    manifest = {
        "source_script": str(SCRIPT), "source_script_sha256": sha256(SCRIPT),
        "status": "REPRODUCED_FROM_LOCAL_RMSE_PACKAGE_WITH_PALETTE_REVISION",
        "visual_encoding": {"strategy_color": {"Foundation": "#8FC7AB", "Fine-tuning": "#29A7B6", "Scratch": "#327FB1"}, "temperature_fill": {"600 K": "open", "Multi-T": "filled"}, "seed_marker": {"123": "circle", "234": "triangle", "345": "square"}, "reference": "#303840"},
        "outputs": {},
    }
    for label, stem in (("CANVA-P03_parity_2x4", p03), ("CANVA-P04_frame_rmse", p04)):
        manifest["outputs"][label] = {ext: {"path": str(stem.with_suffix(f".{ext}")), "sha256": sha256(stem.with_suffix(f".{ext}")), "bytes": stem.with_suffix(f".{ext}").stat().st_size} for ext in ("pdf", "svg", "png")}
    (CAMPAIGN / "RMSE_PALETTE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
