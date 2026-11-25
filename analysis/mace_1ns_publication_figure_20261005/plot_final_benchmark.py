#!/usr/bin/env python3
"""Final six-panel MACE-MD benchmark after removing unsupported HFACF/kappa."""
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

PLOT_LIBRARY_SRC = Path("/Users/alina/Documents/Codex/plot_library/src")
if str(PLOT_LIBRARY_SRC) not in sys.path:
    sys.path.insert(0, str(PLOT_LIBRARY_SRC))
from plot_atlas.palettes import get_palette  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
THERMO = DATA / "thermo"
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
get_palette("isotope_sites")  # registered Atlas palette; reference colors below are explicit overrides

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
STABLE = ["2-naive-fine-tuning", "3-multi_T-fine-tuning"]
FAILURE_TIME = {"foundation-omat": 8.8, "1-from-scratch": 462.2}
EV_A3_TO_GPA = 160.21766208


def style_ax(ax) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#3A3A3A")
        spine.set_linewidth(0.95)
    ax.tick_params(direction="out", length=4.0, width=1.15, labelsize=10, colors="black")
    for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")


def label(ax, letter: str) -> None:
    ax.text(-0.12, 1.07, f"{letter})", transform=ax.transAxes, fontsize=18,
            fontweight="bold", va="bottom", ha="left", color="black")


def legend(ax, loc="upper right", models=ORDER) -> None:
    handles = [Line2D([], [], color=COLORS[m], lw=1.45, label=LABELS[m]) for m in models]
    ax.legend(handles=handles, loc=loc, frameon=True, facecolor="white", edgecolor="#C5C5C5",
              framealpha=0.86, fontsize=9.0, handlelength=2.0, borderpad=0.35,
              labelspacing=0.30, handletextpad=0.55)


def stream(model: str) -> pd.DataFrame:
    path = DATA / ("1-from-scratch" if model == "1-from-scratch" else model)
    return pd.read_csv(path / ("failure_scan.csv" if model == "1-from-scratch" else "timeseries.csv"))


def pressure(model: str) -> pd.DataFrame:
    rows = []
    for line in (THERMO / f"{model}.headers").read_text().splitlines():
        st = re.search(r"\bstep=([0-9.+-]+)", line)
        tm = re.search(r"\btime_fs=([0-9.+-]+)", line)
        sm = re.search(r"stress=\"([^\"]+)\"", line)
        if not (st and tm and sm):
            continue
        sigma = np.fromstring(sm.group(1), sep=" ")
        if sigma.size != 9:
            continue
        rows.append({"step": int(float(st.group(1))), "time_ps": float(tm.group(1)) / 1000,
                     "pressure_GPa": -float(np.trace(sigma.reshape(3, 3)) / 3) * EV_A3_TO_GPA})
    df = pd.DataFrame(rows)
    df.to_csv(THERMO / f"{model}.pressure_derived.csv", index=False)
    return df


def panel_temperature(ax) -> None:
    label(ax, "a")
    for model in ORDER:
        df = stream(model)
        ax.plot(df.time_ps, df.temperature_K, color=COLORS[model], lw=1.05)
    ax.set(xlim=(0, 1000), ylim=(250, 700), ylabel="Temperature (K)")
    ax.set_xticks([0, 200, 400, 600, 800, 1000]); ax.set_xticklabels([])
    legend(ax, "lower right"); style_ax(ax)
    for model, t in FAILURE_TIME.items():
        ax.axvline(t, color=COLORS[model], lw=0.8, ls=(0, (3, 2)), alpha=0.65)
    ax.text(0.02, 0.93, "failure onset: 8.8 ps / 462.2 ps", transform=ax.transAxes,
            fontsize=7, color="#6B3C32", va="top")


def panel_pressure(ax) -> None:
    label(ax, "d")
    for model in ORDER:
        df = pressure(model).copy()
        df.loc[(df.pressure_GPa < -0.8) | (df.pressure_GPa > 1.1), "pressure_GPa"] = np.nan
        ax.plot(df.time_ps, df.pressure_GPa, color=COLORS[model], lw=1.05)
    ax.set(xlim=(0, 1000), ylim=(-0.8, 1.1), ylabel="Pressure (GPa)", xlabel="Time (ps)")
    ax.set_xticks([0, 200, 400, 600, 800, 1000]); legend(ax); style_ax(ax)
    for model, t in FAILURE_TIME.items():
        ax.axvline(t, color=COLORS[model], lw=0.8, ls=(0, (3, 2)), alpha=0.65)


def panel_force(ax) -> None:
    label(ax, "b")
    ax.set_yscale("log")
    for model in ORDER:
        df = stream(model)
        ax.plot(df.time_ps, df.max_force_eV_A, color=COLORS[model], lw=1.0)
    ax.axhline(100, color="#777777", lw=0.8, ls=(0, (3, 2)))
    ax.set(xlim=(0, 1000), ylim=(0.25, 2e6), ylabel="Max force (eV Å$^{-1}$)", xlabel="Time (ps)")
    ax.set_xticks([0, 200, 400, 600, 800, 1000]); legend(ax); style_ax(ax)
    ax.text(0.98, 0.05, "100 eV Å$^{-1}$ acceptance threshold", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.7, color="#666666")


def panel_msd(ax) -> None:
    label(ax, "c")
    for model in ["2-naive-fine-tuning", "3-multi_T-fine-tuning", "foundation-omat"]:
        path = DATA / model / "msd.csv"
        if path.exists():
            df = pd.read_csv(path)
            ax.plot(df.lag_ps, df.msd_A2, color=COLORS[model], lw=1.05)
    ax.set(xlim=(0, 1000), ylim=(0, 50), ylabel="MSD (Å²)", xlabel="Time (ps)")
    ax.set_xticks([0, 200, 400, 600, 800, 1000]); legend(ax, "upper left"); style_ax(ax)
    ax.text(0.98, 0.06, "failure-mode trajectories: no D$_{xy}$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="#6B3C32")


def panel_dxy(ax) -> None:
    label(ax, "e")
    ax.set_xscale("symlog", linthresh=0.01, linscale=0.9)
    ax.axvline(0, color="#777777", lw=0.8, ls=(0, (3, 2)))
    positions = {"2-naive-fine-tuning": 1.0, "3-multi_T-fine-tuning": 0.0}
    windows = [("200_500_ps", "o", 0.0), ("500_900_ps", "s", 0.18)]
    for model, y in positions.items():
        metrics = json.loads((DATA / model / "metrics.json").read_text())
        for win, marker, shift in windows:
            full = metrics["Dxy"][win]["D_cm2_s"] * 1e6
            blocks = [r["D_cm2_s"] * 1e6 for r in metrics["block_slopes"]
                      if r["window"] == win and r.get("D_cm2_s") is not None and r.get("n", 0) >= 500]
            if blocks:
                ax.scatter(blocks, [y + shift - 0.10] * len(blocks), s=16, marker=marker,
                           facecolor="white", edgecolor=COLORS[model], linewidth=0.8)
            ax.scatter([full], [y + shift], s=54, marker="D", color=COLORS[model],
                       edgecolor="white", linewidth=0.7, zorder=4)
    ax.set(xlim=(-0.08, 1.65), ylim=(-0.35, 1.55), xlabel=r"$D_{xy}$ ($10^{-6}$ cm$^2$ s$^{-1}$)")
    ax.set_yticks([0.1, 1.1], ["FT-600K", "FT-MultiT"])
    style_ax(ax)
    handles = [Line2D([], [], marker="D", color="black", linestyle="None", markersize=5.5, label="full-window fit"),
               Line2D([], [], marker="o", color="black", markerfacecolor="white", linestyle="None", markersize=4, label="block fits")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7)
    ax.text(0.02, 0.03, "Foundation / From-scratch: unstable — no D$_{xy}$", transform=ax.transAxes,
            fontsize=7, color="#6B3C32")


def panel_vacf(ax) -> None:
    label(ax, "f")
    for model in ["2-naive-fine-tuning", "3-multi_T-fine-tuning", "foundation-omat"]:
        path = DATA / model / "vacf.csv"
        if path.exists():
            df = pd.read_csv(path)
            q = df[df.lag_ps <= 5]
            ax.plot(q.lag_ps, q.vacf_normalized, color=COLORS[model], lw=1.05)
    ax.axhline(0, color="#666666", lw=0.9, ls=(0, (3, 2)))
    ax.set(xlim=(0, 5), ylim=(-0.40, 1.05), ylabel="Normalized VACF", xlabel="Time (ps)")
    ax.set_xticks([0, 1, 2, 3, 4, 5]); legend(ax); style_ax(ax)
    ax.text(0.98, 0.07, "From-scratch: unavailable after failure", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="#6B3C32")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_outputs() -> None:
    files = {str(p.relative_to(HERE)): digest(p) for p in sorted(DATA.rglob("*")) if p.is_file()}
    manifest = {
        "figure": "mace_1ns_diffusion_stability_benchmark",
        "source_root": "/scratch16/pclancy3/yi/mace_four_model_MD_analysis_20261003",
        "panels": ["temperature", "pressure", "max_force_geometry_stability", "MSD", "Dxy_block_uncertainty", "VACF"],
        "removed_from_final": ["HFACF", "kappa"],
        "protocol": {"dt_fs": 1.0, "temperature_K": 600.0, "natoms": 2050, "target_step": 1000000},
        "scientific_status": {
            "stable_for_diffusion": ["FT-600K", "FT-MultiT"],
            "failure_mode": {"Foundation-OMAT": "8.8 ps", "From-scratch": "462.2 ps"},
            "thermal_transport": "not claimed; heat current was not recorded",
        },
        "local_input_sha256": files,
    }
    (HERE / "final_benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (HERE / "final_benchmark_caption.md").write_text(
        """# Final six-panel MACE-MD benchmark\n\n"
        "The six panels reproduce the original visual grammar while restricting the quantitative content to observables present in the audited 1 ns archive: temperature, pressure, maximum force/geometry stability, Cr MSD, block-wise Dxy diagnostics, and VACF. Foundation-OMAT becomes unstable at 8.8 ps and From-scratch at 462.2 ps; both remain visible as failure-mode benchmarks and are excluded from Dxy reporting. FT-600K and FT-MultiT are the stable diffusion-bearing models. HFACF and thermal conductivity are removed from the final claim because the heat current was not recorded during the original simulations.\n"""
    )


def main() -> None:
    mpl.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.0), facecolor="white")
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.12, top=0.98, wspace=0.32, hspace=0.45)
    panel_temperature(axes[0, 0])
    panel_force(axes[0, 1])
    panel_msd(axes[0, 2])
    panel_pressure(axes[1, 0])
    panel_dxy(axes[1, 1])
    panel_vacf(axes[1, 2])
    for ext, kwargs in [("pdf", {}), ("svg", {}), ("png", {"dpi": 500})]:
        fig.savefig(OUT / f"mace_1ns_final_diffusion_benchmark.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    write_outputs()


if __name__ == "__main__":
    main()
