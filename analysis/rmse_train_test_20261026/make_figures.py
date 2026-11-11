#!/usr/bin/env python3
"""Build parity/R2 panels and frame-level RMSE distribution panels."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw" / "rockfish_parity"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
MODELS = ["Scratch-600K", "Scratch-MultiT", "FT-600K", "FT-MultiT"]
COLORS = {"Scratch-600K": "#ff9414", "Scratch-MultiT": "#f4c04a", "FT-600K": "#0b3a82", "FT-MultiT": "#6aa9e8"}
SEEDS = ["123", "234", "345"]


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 12.5, "axes.labelsize": 13, "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.linewidth": 1.5, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
        "xtick.major.width": 1.4, "ytick.major.width": 1.4, "xtick.major.size": 4.5, "ytick.major.size": 4.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def axis_style(ax) -> None:
    ax.grid(False)
    for s in ax.spines.values():
        s.set_color("black")
        s.set_linewidth(1.5)
    ax.tick_params(direction="out", width=1.4, length=4.5)


def load() -> tuple[list[dict], dict[tuple[str, str, str], dict]]:
    summaries, arrays = [], {}
    for p in sorted(RAW.glob("*.json")):
        d = json.loads(p.read_text())
        if d.get("status") != "PARITY_EVAL_OK" or not d.get("model_sha_ok") or not d.get("data_sha_ok"):
            raise RuntimeError(f"Parity matrix contains a failed or hash-unchecked result: {p.name}")
        key = (d["model_family"], str(d["training_seed"]), d["split"])
        n = np.load(RAW / f"{d['eval_id']}__parity.npz")
        summaries.append({
            "model": d["model_family"], "seed": str(d["training_seed"]), "split": d["split"],
            "eval_id": d["eval_id"], "model_sha256": d["model_sha256"], "data_sha256": d["data_sha256"],
            **d["summary"],
        })
        arrays[key] = {k: n[k] for k in n.files}
    expected = {(m, s, split) for m in MODELS for s in SEEDS for split in ("train", "test")}
    if set(arrays) != expected:
        raise RuntimeError(f"Incomplete parity matrix: missing={sorted(expected - set(arrays))}")
    return summaries, arrays


def load_summaries() -> list[dict]:
    """Load only JSON summaries for frame-distribution-only re-rendering."""
    summaries = []
    for p in sorted(RAW.glob("*.json")):
        d = json.loads(p.read_text())
        if d.get("status") != "PARITY_EVAL_OK" or not d.get("model_sha_ok") or not d.get("data_sha_ok"):
            raise RuntimeError(f"Parity matrix contains a failed or hash-unchecked result: {p.name}")
        summaries.append({
            "model": d["model_family"], "seed": str(d["training_seed"]), "split": d["split"],
            "eval_id": d["eval_id"], "model_sha256": d["model_sha256"], "data_sha256": d["data_sha256"],
            **d["summary"],
        })
    return summaries


def write_summary(rows: list[dict]) -> None:
    fields = list(rows[0])
    with (ROOT / "parity_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def sample_for_plot(x: np.ndarray, y: np.ndarray, limit: int = 9000) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= limit:
        return x, y
    idx = np.linspace(0, len(x) - 1, limit, dtype=int)
    return x[idx], y[idx]


def parity_figure(rows: list[dict], arrays: dict) -> Path:
    """Render train/test parity with common 600K and MultiT axes.

    Training points are deliberately low-saturation gray and drawn first;
    colored test points are drawn over them.  Axes are shared within the two
    model groups so the first two and last two columns can be compared directly.
    Energy panels retain a test-focused inset because the full training energy
    range is much wider than the test range.
    """
    by = {(r["model"], r["seed"], r["split"]): r for r in rows}
    parity_models = ["Scratch-600K", "FT-600K", "Scratch-MultiT", "FT-MultiT"]
    groups = {"600K": parity_models[:2], "MultiT": parity_models[2:]}

    def xy(model: str, split: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
        xx, yy = [], []
        for seed in SEEDS:
            d = arrays[(model, seed, split)]
            if metric == "energy":
                xx.append(d["energy_ref_eV_per_atom"]); yy.append(d["energy_pred_eV_per_atom"])
            else:
                xx.append(d["force_ref_eV_A"]); yy.append(d["force_pred_eV_A"])
        return np.concatenate(xx), np.concatenate(yy)

    limits = {}
    test_limits = {}
    # Force panels use one common range across all four model families and
    # both splits, so the force-error clouds are directly comparable.
    force_full = [v for model in parity_models for split in ("train", "test") for v in xy(model, split, "force")]
    force_lo = float(np.nanmin(np.concatenate(force_full)))
    force_hi = float(np.nanmax(np.concatenate(force_full)))
    force_pad = 0.04 * (force_hi - force_lo if force_hi > force_lo else 1.0)
    limits[("force", "all")] = (force_lo - force_pad, force_hi + force_pad)
    for metric in ("energy", "force"):
        for group, models in groups.items():
            if metric != "force":
                full = [v for model in models for split in ("train", "test") for v in xy(model, split, metric)]
                lo = float(np.nanmin(np.concatenate(full))); hi = float(np.nanmax(np.concatenate(full)))
                pad = 0.04 * (hi - lo if hi > lo else 1.0)
                limits[(metric, group)] = (lo - pad, hi + pad)
            test = [v for model in models for v in xy(model, "test", metric)]
            zlo = float(np.nanmin(np.concatenate(test))); zhi = float(np.nanmax(np.concatenate(test)))
            zpad = 0.08 * (zhi - zlo if zhi > zlo else 1.0)
            test_limits[(metric, group)] = (zlo - zpad, zhi + zpad)

    fig, axs = plt.subplots(2, 4, figsize=(18.5, 8.8), squeeze=False)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.105, top=0.88, hspace=0.30, wspace=0.25)
    for i, model in enumerate(parity_models):
        c = COLORS[model]
        group = "600K" if i < 2 else "MultiT"
        for j, (metric, xlabel, ylabel) in enumerate((
            ("energy", "DFT energy / atom (eV)", "MLIP energy / atom (eV)"),
            ("force", "DFT force component (eV Å$^{-1}$)", "MLIP force component (eV Å$^{-1}$)"),
        )):
            ax = axs[j, i]; axis_style(ax)
            train_x, train_y = xy(model, "train", metric)
            test_x, test_y = xy(model, "test", metric)
            tr_xp, tr_yp = sample_for_plot(train_x, train_y, limit=12000)
            te_xp, te_yp = sample_for_plot(test_x, test_y, limit=12000)
            if metric == "energy":
                ax.scatter(tr_xp, tr_yp, s=8, color="#b8b8b8", alpha=0.16, linewidths=0, rasterized=True, zorder=2)
                ax.scatter(te_xp, te_yp, s=13, color=c, alpha=0.72, linewidths=0, rasterized=True, zorder=3)
            else:
                ax.scatter(tr_xp, tr_yp, s=6, color="#b8b8b8", alpha=0.14, marker="o", linewidths=0, rasterized=True, zorder=2)
                ax.scatter(te_xp, te_yp, s=8, color=c, alpha=0.60, marker="x", linewidths=0.55, rasterized=True, zorder=3)
            lo, hi = limits[(metric, "all" if metric == "force" else group)]
            ax.plot([lo, hi], [lo, hi], "--", color="#222222", lw=1.25, zorder=1)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel(xlabel, fontweight="bold"); ax.set_ylabel(ylabel, fontweight="bold")
            ax.set_aspect("equal", adjustable="box")
            if j == 0:
                ax.set_title(f"{chr(97+i)}) {model}", loc="left", color=c, pad=8)
            if metric == "energy":
                # A local test zoom makes the offset visible without discarding
                # the full train-range context in the main axes.
                zlo, zhi = test_limits[(metric, group)]
                ins = ax.inset_axes([0.53, 0.08, 0.42, 0.34])
                axis_style(ins)
                # Keep the train context visible in the local zoom as well;
                # the gray layer is intentionally more opaque here because
                # the colored test points can otherwise cover it completely.
                ins.scatter(tr_xp, tr_yp, s=5.0, color="#9a9a9a", alpha=0.34, linewidths=0, rasterized=True, zorder=2)
                ins.scatter(te_xp, te_yp, s=5.0, color=c, alpha=0.72, linewidths=0, rasterized=True)
                ins.plot([zlo, zhi], [zlo, zhi], "--", color="#222222", lw=0.8)
                ins.set_xlim(zlo, zhi); ins.set_ylim(zlo, zhi)
                ins.tick_params(labelsize=7.5, width=0.8, length=2.5)
            # R2 varies slightly across seed; report test mean/SD and energy bias.
            k = "energy_r2" if metric == "energy" else "force_r2"
            te = [by[(model, s, "test")][k] for s in SEEDS]
            annotation = f"Test R² {np.mean(te):.3f}±{np.std(te, ddof=1):.3f}"
            if metric == "energy":
                bias = [by[(model, s, "test")]["energy_bias_meV_per_atom"] for s in SEEDS]
                annotation += f"\nBias {np.mean(bias):+.1f} meV atom$^{{-1}}$"
            ax.text(0.04, 0.96, annotation, transform=ax.transAxes, va="top", ha="left", fontsize=11.0, bbox={"facecolor":"white", "edgecolor":"none", "alpha":0.82, "pad":2.8})
    fig.legend(handles=[mpl.lines.Line2D([], [], marker="o", linestyle="None", color="#b8b8b8", alpha=0.75, markersize=7, label="Train"), mpl.lines.Line2D([], [], marker="x", linestyle="None", color="#444444", alpha=0.90, markersize=7, label="Test")], loc="upper center", bbox_to_anchor=(0.5, 0.955), frameon=False, ncol=2, fontsize=15, handletextpad=0.5, columnspacing=1.4)
    fig.text(0.5, 0.99, "DFT reference vs MLIP prediction: train context and test performance", ha="center", va="top", fontsize=15, fontweight="bold")
    path = OUT / "fig_parity_train_test_common_2x4"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig); return path.with_suffix(".pdf")


def frame_distribution_figure(rows: list[dict]) -> Path:
    # Energy uses absolute per-frame error; force uses per-frame component RMSE.
    fig = plt.figure(figsize=(13.0, 9.8))
    outer = fig.add_gridspec(2, 2, left=0.10, right=0.985, bottom=0.075, top=0.79, wspace=0.24, hspace=0.42)
    by = {(r["model"], r["seed"], r["split"]): r for r in rows}
    frame_values = {}
    for model in MODELS:
        for seed in SEEDS:
            for split in ("train", "test"):
                p = RAW / f"{by[(model, seed, split)]['eval_id']}__per_frame.csv"
                with p.open() as f:
                    rr = list(csv.DictReader(f))
                frame_values[(model, seed, split, "energy_abs_error_meV_per_atom")] = np.asarray(
                    [float(r["energy_abs_error_meV_per_atom"]) for r in rr], dtype=float
                )
                frame_values[(model, seed, split, "force_rmse_meV_A")] = np.asarray(
                    [float(r["force_rmse_meV_A"]) for r in rr], dtype=float
                )
    metric_limits = {}
    for metric in ("energy_abs_error_meV_per_atom", "force_rmse_meV_A"):
        all_values = np.concatenate([v for k, v in frame_values.items() if k[-1] == metric])
        positive = all_values[np.isfinite(all_values) & (all_values > 0)]
        metric_limits[metric] = (float(np.min(positive)) * 0.65, float(np.max(positive)) * 1.55)
    for i, model in enumerate(MODELS):
        sub = outer[divmod(i, 2)].subgridspec(2, 1, hspace=0.17)
        c = COLORS[model]
        for j, (metric, ylabel, summary_key) in enumerate((
            ("energy_abs_error_meV_per_atom", "|ΔE| / atom\n(meV)", "energy_rmse_meV_per_atom"),
            ("force_rmse_meV_A", "Frame force RMSE\n(meV Å$^{-1}$)", "force_rmse_meV_A"),
        )):
            ax = fig.add_subplot(sub[j]); axis_style(ax)
            for xpos, split in enumerate(("train", "test")):
                vals = []
                for seed in SEEDS:
                    vals.extend(frame_values[(model, seed, split, metric)])
                vals = np.asarray(vals, dtype=float)
                # The box is only a compact distribution guide; every frame remains visible as a point.
                ax.boxplot(vals, positions=[xpos], widths=0.25, showfliers=False, patch_artist=True, boxprops={"facecolor":"white", "edgecolor":c, "linewidth":1.2}, medianprops={"color":"black", "linewidth":1.3}, whiskerprops={"color":c}, capprops={"color":c})
                jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) else np.array([])
                ax.scatter(np.full(len(vals), xpos) + jitter, vals, s=3.5, color=c, alpha=0.24, linewidths=0, zorder=2)
                # Dataset RMSE is a weighted/global metric and is intentionally marked separately.
                rmses = [by[(model, seed, split)][summary_key] for seed in SEEDS]
                ax.scatter([xpos], [float(np.mean(rmses))], marker="D", s=35, facecolor="white", edgecolor="black", linewidth=1.1, zorder=4)
            ax.set_xticks([0, 1], ["Train", "Test"])
            ax.set_ylabel(ylabel, fontweight="bold")
            ax.set_yscale("log")
            # Use one quantitative y-range for all energy panels and one for
            # all force panels so cross-model distributions are directly comparable.
            ax.set_ylim(*metric_limits[metric])
            if j == 0:
                ax.set_title(f"{chr(97+i)}) {model}", loc="left", color=c, pad=8)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Split", fontweight="bold")
    fig.legend(handles=[mpl.lines.Line2D([], [], marker="o", linestyle="None", color="#555555", alpha=0.45, markersize=5.5, label="individual AIMD frames"), mpl.lines.Line2D([], [], marker="D", linestyle="None", markerfacecolor="white", markeredgecolor="black", markersize=7, label="mean global RMSE")], loc="upper center", bbox_to_anchor=(0.5, 0.905), frameon=False, ncol=2, fontsize=13.5, handletextpad=0.5, columnspacing=1.5)
    fig.suptitle("Frame-level error distributions with all train/test AIMD frames", y=0.985, fontsize=14.5, fontweight="bold")
    path = OUT / "fig_rmse_frame_distributions"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig); return path.with_suffix(".pdf")


def main() -> None:
    style()
    if "--frame-only" in sys.argv:
        rows = load_summaries()
        frame_distribution_figure(rows)
        print(f"Rendered frame distributions from {len(rows)} JSON summaries")
        return
    rows, arrays = load()
    write_summary(rows)
    parity_figure(rows, arrays)
    if "--parity-only" in sys.argv:
        print(f"Rendered parity figure from {len(rows)} evaluations")
        return
    frame_distribution_figure(rows)
    print(f"Rendered {len(rows)} parity evaluations")


if __name__ == "__main__":
    main()
