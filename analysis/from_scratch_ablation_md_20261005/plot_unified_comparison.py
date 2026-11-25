#!/usr/bin/env python3
"""Plot Atlas comparison figure for new MD products and paper references.

This is a one-off composite that reuses the Plot Atlas publication style and
the registered molecular-dynamics line-panel primitive.  It never treats the
paper/legacy traces as same-protocol replicas; those traces are intentionally
drawn dashed and labeled as references.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_atlas import apply_publication_style, bold_tick_labels, get_palette, panel_label, save_figure
from plot_atlas.molecular_dynamics import steered_dynamics_panel


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "", "nan", "NaN") else float("nan")
    except ValueError:
        return float("nan")


def line_color(palette: dict, index: int) -> str:
    colors = palette["categorical"]
    return colors[index % len(colors)]


def load_new_runs(root: Path, max_new: int) -> list[dict]:
    runs = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("scientific_gate") not in {"screening_or_production_complete", "production_complete"}:
            continue
        run_dir = metrics_path.parent
        timeseries = read_csv(run_dir / "timeseries.csv")
        msd = read_csv(run_dir / "msd.csv")
        vacf = read_csv(run_dir / "vacf.csv")
        hfacf = read_csv(run_dir / "hfacf.csv")
        kappa = read_csv(run_dir / "kappa.csv")
        manifest_value = metrics.get("manifest")
        manifest_obj = manifest_value if isinstance(manifest_value, dict) else {}
        family = manifest_obj.get("family", "")
        # The production postprocessor stores the manifest path as a string;
        # recover family/seed from its sibling run manifest when needed.
        if not family:
            manifest_path = Path(metrics.get("manifest", ""))
            if not manifest_path.exists():
                candidate = run_dir.parent.parent / "runs" / run_dir.name / "run_manifest.json"
                manifest_path = candidate
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                family = manifest.get("family", "")
                seed = manifest.get("training_seed", "")
            else:
                seed = ""
        else:
            seed = manifest_obj.get("training_seed", "")
        if not family or seed == "":
            parts = run_dir.name.split("__")
            family = family or parts[0]
            seed = seed or next((p.removeprefix("seed") for p in parts if p.startswith("seed")), "")
        runs.append({
            "name": run_dir.name,
            "label": f"{family.replace('ablation_', '')} seed{seed}",
            "family": family,
            "seed": seed,
            "timeseries": timeseries,
            "msd": msd,
            "vacf": vacf,
            "hfacf": hfacf,
            "kappa": kappa,
            "metrics": metrics,
        })
    return runs[:max_new]


def plot_series(ax, rows, xkey, ykey, label, color, *, linestyle="-", alpha=0.9, linewidth=1.6):
    x = np.asarray([f(r.get(xkey)) for r in rows])
    y = np.asarray([f(r.get(ykey)) for r in rows])
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.any():
        ax.plot(x[mask], y[mask], color=color, lw=linewidth, alpha=alpha, ls=linestyle, label=label)


def style_axis(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(width=1.0, length=5, pad=3)
    bold_tick_labels(ax)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")


def load_reference(reference_dir: Path) -> dict[str, list[dict[str, str]]]:
    return {
        name: read_csv(reference_dir / f"{name}.csv")
        for name in ("thermodynamics_data", "hfacf_data", "kappa_data", "msd_data", "vacf_data")
        if (reference_dir / f"{name}.csv").exists()
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-root", type=Path, required=True, help="directory containing per-run metrics.json files")
    ap.add_argument("--reference-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-new", type=int, default=6)
    args = ap.parse_args()

    new_runs = load_new_runs(args.new_root, args.max_new)
    if not new_runs:
        raise SystemExit("no complete finite new runs found")
    reference = load_reference(args.reference_dir)
    palette = get_palette("nature_steered_dynamics")
    apply_publication_style(9.5)
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.1), constrained_layout=True)
    axes = axes.ravel()

    labels = {}
    series = {}
    colors = {}
    for i, run in enumerate(new_runs):
        labels[run["name"]] = run["label"]
        colors[run["name"]] = line_color(palette, i)
        series[run["name"]] = np.asarray([f(r.get("msd_xy_A2")) for r in run["msd"]])
    msd_x = np.asarray([f(r.get("lag_ps")) for r in new_runs[0]["msd"]])
    msd_ymax = max(0.01, float(np.nanmax(np.concatenate([v[np.isfinite(v)] for v in series.values()])) * 1.08))
    # Reuse the registered Atlas dynamics line grammar for the MSD panel.
    steered_dynamics_panel(
        axes[2],
        msd_x,
        series,
        labels=labels,
        palette={"categorical": [colors[k] for k in series]},
        xlim=(0, float(np.nanmax(msd_x))),
        ylim=(0, msd_ymax),
        xticks=tuple(np.linspace(0, float(np.nanmax(msd_x)), 5)),
        yticks=tuple(np.linspace(0, msd_ymax, 5)),
        xlabel="Lag time (ps)",
        ylabel="Cr MSD$_{xy}$ (Å$^2$)",
        legend=False,
    )

    for i, run in enumerate(new_runs):
        c = colors[run["name"]]
        plot_series(axes[0], run["timeseries"], "time_ps", "temperature_K", run["label"], c)
        plot_series(axes[1], run["timeseries"], "time_ps", "pressure_GPa", run["label"], c)
        plot_series(axes[3], run["vacf"], "lag_ps", "vacf_normalized", run["label"], c)
        plot_series(axes[4], run["hfacf"], "lag_ps", "hfacf_normalized", run["label"], c)
        plot_series(axes[5], run["kappa"], "time_ps", "kappa_W_mK", run["label"], c)

    # Paper traces are visible for context but use a neutral dashed grammar.
    ref_colors = {"FT - 600K": "#777777", "FT - Multi-T": "#999999", "Foundation": "#AAAAAA", "Scratch": "#555555"}
    thermo = reference.get("thermodynamics_data", [])
    for label in sorted({r.get("label", "") for r in thermo if r.get("label")}):
        rows = [r for r in thermo if r.get("label") == label]
        plot_series(axes[0], rows, "time_ps", "temperature_K", f"paper {label}", ref_colors.get(label, "#888888"), linestyle="--", alpha=0.5, linewidth=1.0)
        plot_series(axes[1], rows, "time_ps", "pressure_GPa", f"paper {label}", ref_colors.get(label, "#888888"), linestyle="--", alpha=0.5, linewidth=1.0)
    for index, key_prefix in ((3, "vacf_"), (4, "hfacf_"), (5, "kappa_")):
        rows = reference.get({3: "vacf_data", 4: "hfacf_data", 5: "kappa_data"}[index], [])
        if not rows:
            continue
        xkey = "time_ps"
        for key in rows[0]:
            if not key.startswith(key_prefix):
                continue
            label = key.removeprefix(key_prefix)
            plot_series(axes[index], rows, xkey, key, f"paper {label}", ref_colors.get(label, "#888888"), linestyle="--", alpha=0.5, linewidth=1.0)

    style_axis(axes[0], "Time (ps)", "Temperature (K)")
    style_axis(axes[1], "Time (ps)", "Pressure (GPa)")
    style_axis(axes[3], "Lag time (ps)", "Normalized VACF")
    style_axis(axes[4], "Lag time (ps)", "Normalized HFACF")
    style_axis(axes[5], "Integration time (ps)", "Running κ (W m$^{-1}$ K$^{-1}$)")
    axes[0].axhline(600.0, color="#333333", lw=0.9, ls=":")
    axes[3].axhline(0.0, color="#333333", lw=0.8, ls=":")
    axes[4].axhline(0.0, color="#333333", lw=0.8, ls=":")
    axes[5].axhline(0.0, color="#333333", lw=0.8, ls=":")
    axes[0].legend(frameon=True, fontsize=7, ncol=2, loc="best")
    panel_titles = ["Thermodynamic stability", "Pressure diagnostic", "Cr migration", "Cr dynamics", "Heat-flux diagnostic", "Thermal transport diagnostic"]
    for ax, title, letter in zip(axes, panel_titles, "abcdef"):
        ax.set_title(title, fontsize=10, fontweight="normal")
        panel_label(ax, letter)
    axes[4].legend(frameon=False, fontsize=6, ncol=2, loc="upper right")
    axes[5].legend(frameon=False, fontsize=6, ncol=2, loc="best")

    args.output.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        save_figure(fig, args.output / f"md_unified_comparison.{ext}")
    plt.close(fig)
    manifest = {
        "figure": "md_unified_comparison",
        "new_root": str(args.new_root),
        "reference_dir": str(args.reference_dir),
        "new_runs": [r["name"] for r in new_runs],
        "new_run_gate": "complete finite runs only",
        "reference_style": "dashed and labeled paper reference; not same-protocol replicas",
        "heat_flux_note": "repository simplified stress-based heat-flux convention; κ is diagnostic until convergence audit",
        "atlas_palette": "nature_steered_dynamics",
    }
    (args.output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
