#!/usr/bin/env python3
"""Overleaf-ready B--G panels plus an A-panel composite placeholder.

Visual grammar follows the supplied reference.  All model legend entries are
shared at figure level; scientific qualifications live in the LaTeX caption.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

PLOT_LIBRARY_SRC = Path("/Users/alina/Documents/Codex/plot_library/src")
if str(PLOT_LIBRARY_SRC) not in sys.path:
    sys.path.insert(0, str(PLOT_LIBRARY_SRC))
from plot_atlas.palettes import get_palette  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
THERMO = DATA / "thermo"
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
get_palette("isotope_sites")

# User-approved JHU/reference colors; do not alter these roles.
COLORS = {
    "2-naive-fine-tuning": "#234B87",
    "3-multi_T-fine-tuning": "#69A9E7",
    "foundation-omat": "#D34F2D",
    "1-from-scratch": "#FF9817",
}
LABELS = {
    "2-naive-fine-tuning": "FT - 600K",
    "3-multi_T-fine-tuning": "FT - Multi-T",
    "foundation-omat": "Foundation-OMAT",
    "1-from-scratch": "From-scratch",
}
ORDER = ["2-naive-fine-tuning", "3-multi_T-fine-tuning", "foundation-omat", "1-from-scratch"]
EV_A3_TO_GPA = 160.21766208
AXIS_TITLE_SIZE = 17
LEGEND_SIZE = 16


def style_ax(ax) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#3A3A3A")
        spine.set_linewidth(1.0)
    ax.tick_params(direction="out", length=4.5, width=1.25, labelsize=13, colors="black")
    for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(AXIS_TITLE_SIZE)
    ax.yaxis.label.set_fontsize(AXIS_TITLE_SIZE)


def panel_letter(ax, letter: str) -> None:
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=22,
            fontweight="bold", va="bottom", ha="left", color="black")


def stream(model: str) -> pd.DataFrame:
    directory = DATA / ("1-from-scratch" if model == "1-from-scratch" else model)
    filename = "failure_scan.csv" if model == "1-from-scratch" else "timeseries.csv"
    return pd.read_csv(directory / filename)


def pressure(model: str) -> pd.DataFrame:
    rows = []
    # The continuation headers start at 8, 80, or 200 ps.  Add the archived
    # original trajectory headers so pressure is plotted from t=0 rather than
    # appearing artificially blank during the original-run interval.
    original_name = "0-foundation-omat" if model == "foundation-omat" else model
    sources = [
        (THERMO / f"{original_name}.original.headers", 0.0),
        (THERMO / f"{model}.headers", None),
    ]
    for path, original_offset in sources:
        if not path.exists():
            continue
        for index, line in enumerate(path.read_text().splitlines()):
            st = re.search(r"\bstep=([0-9.+-]+)", line)
            tm = re.search(r"\btime_fs=([0-9.+-]+)", line)
            sm = re.search(r"stress=\"([^\"]+)\"", line)
            if not sm:
                continue
            sigma = np.fromstring(sm.group(1), sep=" ")
            if sigma.size != 9:
                continue
            if tm:
                time_ps = float(tm.group(1)) / 1000.0
                step = int(float(st.group(1))) if st else round(time_ps * 1000)
            else:
                time_ps = float(index) * 0.2 + float(original_offset or 0.0)
                step = round(time_ps * 1000)
            rows.append({
                "step": step,
                "time_ps": time_ps,
                "pressure_GPa": -float(np.trace(sigma.reshape(3, 3)) / 3) * EV_A3_TO_GPA,
            })
    df = pd.DataFrame(rows)
    df = df.sort_values("step").drop_duplicates("step", keep="last").reset_index(drop=True)
    df.to_csv(THERMO / f"{model}.pressure_derived.csv", index=False)
    return df


def panel_temperature(ax) -> None:
    panel_letter(ax, "B")
    for model in ORDER:
        df = stream(model)
        ax.plot(df.time_ps, df.temperature_K, color=COLORS[model], lw=1.05)
    ax.set(xlim=(0, 1000), ylim=(250, 700), xlabel="Time (ps)", ylabel="Temperature (K)")
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    style_ax(ax)


def panel_force(ax) -> None:
    panel_letter(ax, "C")
    ax.set_yscale("log")
    for model in ORDER:
        df = stream(model)
        ax.plot(df.time_ps, df.max_force_eV_A, color=COLORS[model], lw=1.0)
    ax.set(xlim=(0, 1000), ylim=(0.25, 2e6), xlabel="Time (ps)", ylabel="Max force (eV Å$^{-1}$)")
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    style_ax(ax)


def panel_msd(ax) -> None:
    panel_letter(ax, "D")
    for model in ["2-naive-fine-tuning", "3-multi_T-fine-tuning", "foundation-omat"]:
        path = DATA / model / "msd_full.csv"
        if not path.exists():
            path = DATA / model / "msd.csv"
        if path.exists():
            df = pd.read_csv(path)
            ax.plot(df.lag_ps, df.msd_A2, color=COLORS[model], lw=1.05)
    ax.set(xlim=(0, 1000), ylim=(0, 50), xlabel="Time (ps)", ylabel="MSD (Å$^2$)")
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    style_ax(ax)


def panel_pressure(ax) -> None:
    panel_letter(ax, "E")
    for model in ORDER:
        df = pressure(model).copy()
        df.loc[(df.pressure_GPa < -0.8) | (df.pressure_GPa > 1.1), "pressure_GPa"] = np.nan
        ax.plot(df.time_ps, df.pressure_GPa, color=COLORS[model], lw=1.05)
    ax.set(xlim=(0, 1000), ylim=(-0.8, 1.1), xlabel="Time (ps)", ylabel="Pressure (GPa)")
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    style_ax(ax)


def panel_dxy(ax) -> None:
    panel_letter(ax, "F")
    ax.set_xscale("symlog", linthresh=0.01, linscale=0.9)
    ax.axvline(0, color="#777777", lw=0.8, ls=(0, (3, 2)))
    positions = {"2-naive-fine-tuning": 1.0, "3-multi_T-fine-tuning": 0.0}
    windows = [("200_500_ps", "o", 0.0), ("500_900_ps", "s", 0.18)]
    for model, y in positions.items():
        metrics = json.loads((DATA / model / "metrics.json").read_text())
        for window, marker, shift in windows:
            full = metrics["Dxy"][window]["D_cm2_s"] * 1e6
            blocks = [r["D_cm2_s"] * 1e6 for r in metrics["block_slopes"]
                      if r["window"] == window and r.get("D_cm2_s") is not None and r.get("n", 0) >= 500]
            if blocks:
                ax.scatter(blocks, [y + shift - 0.10] * len(blocks), s=18, marker=marker,
                           facecolor="white", edgecolor=COLORS[model], linewidth=0.8)
            ax.scatter([full], [y + shift], s=58, marker="D", color=COLORS[model],
                       edgecolor="white", linewidth=0.7, zorder=4)
    ax.set(xlim=(-0.08, 1.65), ylim=(-0.35, 1.55),
           xlabel=r"$D_{xy}$ ($10^{-6}$ cm$^2$ s$^{-1}$)")
    ax.set_yticks([0.1, 1.1], ["FT-600K", "FT-MultiT"])
    style_ax(ax)


def panel_vacf(ax) -> None:
    panel_letter(ax, "G")
    models = ["2-naive-fine-tuning", "3-multi_T-fine-tuning", "foundation-omat"]
    curves = {}
    for model in models:
        path = DATA / model / "vacf_full.csv"
        if not path.exists():
            path = DATA / model / "vacf.csv"
        if path.exists():
            curves[model] = pd.read_csv(path)
            ax.plot(curves[model].lag_ps, curves[model].vacf_normalized,
                    color=COLORS[model], lw=1.05)
    ax.axhline(0, color="#666666", lw=0.9, ls=(0, (3, 2)))
    ax.set(xlim=(0, 1000), ylim=(-0.40, 1.05), xlabel="Time (ps)", ylabel="Normalized VACF")
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    style_ax(ax)

    # The full 1 ns relaxation is the primary panel; retain the short-time
    # vibrational structure as a clean, legend-free inset.
    if curves:
        inset = ax.inset_axes([0.56, 0.50, 0.39, 0.39])
        for model, df in curves.items():
            q = df[df.lag_ps <= 5]
            inset.plot(q.lag_ps, q.vacf_normalized, color=COLORS[model], lw=0.9)
        inset.axhline(0, color="#666666", lw=0.7, ls=(0, (3, 2)))
        inset.set(xlim=(0, 5), ylim=(-0.40, 1.05))
        inset.set_xticks([0, 2, 4])
        inset.set_yticks([0, 0.5, 1.0])
        inset.tick_params(direction="out", length=2.5, width=0.8, labelsize=8)
        for tick in [*inset.get_xticklabels(), *inset.get_yticklabels()]:
            tick.set_fontweight("bold")
        for spine in inset.spines.values():
            spine.set_color("#3A3A3A")
            spine.set_linewidth(0.8)


def shared_legend(fig) -> None:
    handles = [Line2D([], [], color=COLORS[m], lw=1.7, label=LABELS[m]) for m in ORDER]
    legend_prop = FontProperties(family="Arial", size=LEGEND_SIZE, weight="bold")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.025),
               ncol=4, frameon=False, prop=legend_prop, handlelength=2.7,
               columnspacing=1.8, handletextpad=0.7)


def placeholder(ax) -> None:
    ax.set_axis_off()
    ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                           facecolor="#FAFAFA", edgecolor="#B9B9B9", linewidth=1.2,
                           linestyle=(0, (4, 3))))
    ax.text(0.5, 0.5, "A\nplaceholder for schematic", transform=ax.transAxes,
            ha="center", va="center", fontsize=18, color="#808080", fontweight="bold")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_caption_and_manifest() -> None:
    tex = r"""\caption{\textbf{Diffusion-focused benchmarking of four MACE models in 1~ns molecular dynamics.} (B) Temperature trajectory at 600~K. (C) Maximum force along the trajectory, shown on a logarithmic scale to retain both bounded and runaway regimes. (D) Cr mean-squared displacement. (E) Pressure derived from the archived periodic-cell stress metadata. (F) Diffusion coefficients $D_{xy}$ from the 200--500~ps and 500--900~ps windows; diamonds denote full-window fits and open markers denote block fits. (G) Normalized velocity autocorrelation function. FT-600K and FT-MultiT are the stable trajectories used for diffusion reporting. Foundation-OMAT becomes unstable at 8.8~ps and From-scratch at 462.2~ps; these trajectories are retained as failure-mode benchmarks and are not assigned a $D_{xy}$. Heat-flux autocorrelation and thermal conductivity are intentionally not claimed because the original simulations did not record the per-frame heat current required for a Green--Kubo estimate.}
\label{fig:mace_1ns_diffusion_benchmark}"""
    (HERE / "final_benchmark_caption.tex").write_text(tex + "\n")
    files = {str(p.relative_to(HERE)): digest(p) for p in sorted(DATA.rglob("*")) if p.is_file()}
    manifest = {
        "figure": "mace_1ns_overleaf_B_to_G",
        "source_root": "/scratch16/pclancy3/yi/mace_four_model_MD_analysis_20261003",
        "panel_letters": ["B", "C", "D", "E", "F", "G"],
        "panel_a": "placeholder_only_in_composite_template",
        "shared_legend": True,
        "in_plot_annotations": False,
        "full_lag_observables": {"MSD_max_lag_ps": 1000.2, "VACF_max_lag_ps": 1000.2,
                                  "VACF_inset_lag_ps": 5.0},
        "caption_file": "final_benchmark_caption.tex",
        "protocol": {"dt_fs": 1.0, "temperature_K": 600.0, "natoms": 2050, "target_step": 1000000},
        "removed_claims": ["HFACF", "thermal conductivity kappa"],
        "local_input_sha256": files,
    }
    (HERE / "overleaf_panels_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def save(fig, stem: str) -> None:
    for ext, kwargs in [("pdf", {}), ("svg", {}), ("png", {"dpi": 500})]:
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    mpl.rcParams.update({"font.family": "Arial", "font.size": 13, "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    # Quantitative figure: B--G, six panels, one shared legend at the bottom.
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.3), facecolor="white")
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.21, top=0.98, wspace=0.34, hspace=0.47)
    panel_temperature(axes[0, 0]); panel_force(axes[0, 1]); panel_msd(axes[0, 2])
    panel_pressure(axes[1, 0]); panel_dxy(axes[1, 1]); panel_vacf(axes[1, 2])
    shared_legend(fig)
    save(fig, "mace_1ns_overleaf_B_to_G")

    # Composite template: A placeholder plus the same B--G panels and legend.
    fig = plt.figure(figsize=(18.0, 8.3), facecolor="white")
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.0],
                          left=0.045, right=0.99, bottom=0.21, top=0.98,
                          wspace=0.33, hspace=0.47)
    a = fig.add_subplot(gs[:, 0]); placeholder(a)
    q = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]),
         fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])]
    panel_temperature(q[0]); panel_force(q[1]); panel_msd(q[2])
    panel_pressure(q[3]); panel_dxy(q[4]); panel_vacf(q[5])
    shared_legend(fig)
    save(fig, "mace_1ns_overleaf_A_to_G_template")
    write_caption_and_manifest()


if __name__ == "__main__":
    main()
