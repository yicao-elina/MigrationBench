#!/usr/bin/env python3
"""Strict six-axis replica of the supplied original MACE-MD figure.

The layout and visual grammar intentionally follow the reference image:
temperature / pressure, HFACF / kappa, MSD, and VACF.  The current 1 ns
archive does not contain heat-flux data, so those two axes are retained but
fail closed with an explicit data-availability annotation.
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

PLOT_LIBRARY_SRC = Path("/Users/alina/Documents/Codex/plot_library/src")
if str(PLOT_LIBRARY_SRC) not in sys.path:
    sys.path.insert(0, str(PLOT_LIBRARY_SRC))
from plot_atlas.palettes import get_palette  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
THERMO = DATA / "thermo"
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# The supplied reference explicitly fixes these semantic color roles.  The
# Atlas palette is loaded above; these reference colors are a deliberate
# user-specified style override for this reconstruction.
_atlas = get_palette("isotope_sites")
COLORS = {
    "2-naive-fine-tuning": "#234B87",   # FT-600K, dark blue
    "3-multi_T-fine-tuning": "#69A9E7", # FT-MultiT, light blue
    "foundation-omat": "#D34F2D",      # Foundation, red-orange
    "1-from-scratch": "#FF9817",       # Scratch, orange
}
LABELS = {
    "2-naive-fine-tuning": "FT - 600K",
    "3-multi_T-fine-tuning": "FT - Multi-T",
    "foundation-omat": "Foundation-OMAT",
    "1-from-scratch": "From-scratch",
}
ORDER = ["2-naive-fine-tuning", "3-multi_T-fine-tuning", "foundation-omat", "1-from-scratch"]
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


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.07, f"{label})", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=18, fontweight="bold", color="black")


def read_stream(model: str) -> pd.DataFrame:
    if model == "1-from-scratch":
        return pd.read_csv(DATA / "1-from-scratch" / "failure_scan.csv")
    return pd.read_csv(DATA / model / "timeseries.csv")


def plot_legend(ax, *, loc: str) -> None:
    handles = [Line2D([], [], color=COLORS[m], lw=1.45, label=LABELS[m]) for m in ORDER]
    ax.legend(handles=handles, loc=loc, frameon=True, facecolor="white", edgecolor="#C5C5C5",
              framealpha=0.86, fontsize=9.0, handlelength=2.0, borderpad=0.35,
              labelspacing=0.30, handletextpad=0.55)


def parse_headers(model: str) -> pd.DataFrame:
    """Parse stress metadata exported from each current 1 ns extxyz segment."""
    rows = []
    header_file = THERMO / f"{model}.headers"
    for line in header_file.read_text().splitlines():
        step = re.search(r"\bstep=([0-9.+-]+)", line)
        time_fs = re.search(r"\btime_fs=([0-9.+-]+)", line)
        stress = re.search(r"stress=\"([^\"]+)\"", line)
        if not (step and time_fs and stress):
            continue
        sigma = np.fromstring(stress.group(1), sep=" ")
        if sigma.size != 9:
            continue
        # ASE's virial stress convention is pressure = -Tr(stress)/3.
        pressure = -float(np.trace(sigma.reshape(3, 3)) / 3.0) * EV_A3_TO_GPA
        rows.append({
            "step": int(float(step.group(1))),
            "time_ps": float(time_fs.group(1)) / 1000.0,
            "pressure_GPa": pressure,
        })
    out = pd.DataFrame(rows)
    out.to_csv(THERMO / f"{model}.pressure_derived.csv", index=False)
    return out


def panel_temperature(ax) -> None:
    panel_label(ax, "b")
    for model in ORDER:
        df = read_stream(model)
        ax.plot(df.time_ps, df.temperature_K, color=COLORS[model], lw=1.05, label=LABELS[model])
    ax.set_xlim(0, 1000)
    ax.set_ylim(250, 700)
    ax.set_ylabel("Temperature (K)", fontsize=14)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_xticklabels([])
    style_ax(ax)
    plot_legend(ax, loc="lower right")
    for model, t in FAILURE_TIME.items():
        ax.axvline(t, color=COLORS[model], lw=0.8, ls=(0, (3, 2)), alpha=0.65)
    ax.text(0.02, 0.93, "unstable: 8.8 ps / 462.2 ps", transform=ax.transAxes,
            fontsize=7.0, color="#6B3C32", ha="left", va="top")


def panel_pressure(ax) -> None:
    for model in ORDER:
        df = parse_headers(model)
        # Keep the reference axis range without drawing artificial vertical
        # clipping strokes when an unstable trajectory leaves that range.
        q = df.copy()
        q.loc[(q.pressure_GPa < -0.8) | (q.pressure_GPa > 1.1), "pressure_GPa"] = np.nan
        ax.plot(q.time_ps, q.pressure_GPa, color=COLORS[model], lw=1.05, label=LABELS[model])
    ax.set_xlim(0, 1000)
    ax.set_ylim(-0.8, 1.1)
    ax.set_ylabel("Pressure (GPa)", fontsize=14)
    ax.set_xlabel("Time (ps)", fontsize=14)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    style_ax(ax)
    plot_legend(ax, loc="upper right")
    for model, t in FAILURE_TIME.items():
        ax.axvline(t, color=COLORS[model], lw=0.8, ls=(0, (3, 2)), alpha=0.65)


def unavailable_panel(ax, ylabel: str, label: str | None) -> None:
    if label is not None:
        panel_label(ax, label)
    ax.set_xlim(0, 1000)
    ax.set_ylim(-0.05 if ylabel == "Normalized HFACF" else -0.02,
                1.05 if ylabel == "Normalized HFACF" else 0.12)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_xticklabels([] if ylabel == "Normalized HFACF" else ["0", "200", "400", "600", "800", "1000"])
    if ylabel != "Normalized HFACF":
        ax.set_xlabel("Time (ps)", fontsize=14)
    style_ax(ax)
    plot_legend(ax, loc="upper right")
    ax.axhline(0, color="#666666", lw=0.8, ls=(0, (3, 2)))
    ax.text(0.5, 0.50, "NOT AVAILABLE\nheat-flux time series not archived",
            transform=ax.transAxes, ha="center", va="center", fontsize=10.0,
            fontweight="bold", color="#777777", linespacing=1.3)


def panel_msd(ax) -> None:
    panel_label(ax, "d")
    for model in ["2-naive-fine-tuning", "3-multi_T-fine-tuning", "foundation-omat"]:
        path = DATA / model / "msd.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        q = df[df.lag_ps <= 1000]
        ax.plot(q.lag_ps, q.msd_A2, color=COLORS[model], lw=1.1, label=LABELS[model])
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 50)
    ax.set_ylabel("MSD (Å²)", fontsize=14)
    ax.set_xlabel("Time (ps)", fontsize=14)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    style_ax(ax)
    handles = [Line2D([], [], color=COLORS[m], lw=1.45, label=LABELS[m]) for m in ORDER]
    ax.legend(handles=handles, loc="upper left", frameon=True, facecolor="white", edgecolor="#C5C5C5",
              framealpha=0.86, fontsize=9.0, handlelength=2.0, borderpad=0.35,
              labelspacing=0.30, handletextpad=0.55)
    ax.text(0.98, 0.06, "Foundation / From-scratch: failure-mode\n(no Dxy reported)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0, color="#6B3C32")


def panel_vacf(ax) -> None:
    panel_label(ax, "e")
    for model in ["2-naive-fine-tuning", "3-multi_T-fine-tuning", "foundation-omat"]:
        path = DATA / model / "vacf.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        q = df[df.lag_ps <= 5]
        ax.plot(q.lag_ps, q.vacf_normalized, color=COLORS[model], lw=1.1, label=LABELS[model])
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.40, 1.05)
    ax.set_ylabel("Normalized VACF", fontsize=14)
    ax.set_xlabel("Time (ps)", fontsize=14)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.axhline(0, color="#666666", lw=0.9, ls=(0, (3, 2)))
    style_ax(ax)
    handles = [Line2D([], [], color=COLORS[m], lw=1.45, label=LABELS[m]) for m in ORDER]
    ax.legend(handles=handles, loc="upper right", frameon=True, facecolor="white", edgecolor="#C5C5C5",
              framealpha=0.86, fontsize=9.0, handlelength=2.0, borderpad=0.35,
              labelspacing=0.30, handletextpad=0.55)
    ax.text(0.98, 0.07, "From-scratch: unavailable after failure", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.0, color="#6B3C32")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest() -> None:
    files = {str(p.relative_to(HERE)): sha256(p) for p in sorted(DATA.rglob("*")) if p.is_file()}
    payload = {
        "figure": "strict_six_axis_replica_of_original_MACE_MD_figure",
        "source_root": "/scratch16/pclancy3/yi/mace_four_model_MD_analysis_20261003",
        "layout": ["temperature", "pressure", "HFACF", "kappa", "MSD", "VACF"],
        "reference_style": "user_supplied_figure_b8c50a69",
        "protocol": {"dt_fs": 1.0, "temperature_K": 600.0, "natoms": 2050, "target_step": 1000000},
        "data_status": {
            "temperature": "available",
            "pressure": "derived from extxyz stress metadata using -Tr(stress)/3",
            "HFACF": "not available; no heat-flux time series in current four-model 1 ns archive",
            "kappa": "not available; no HFACF/integral in current four-model 1 ns archive",
            "MSD": "available for Foundation-OMAT and two stable fine-tuned models; From-scratch retained as failure-mode",
            "VACF": "available for Foundation-OMAT and two stable fine-tuned models; From-scratch retained as failure-mode",
        },
        "source_audit_files": {m: str(DATA / m / "audit.json") for m in [
            "foundation-omat", "1-from-scratch", "2-naive-fine-tuning", "3-multi_T-fine-tuning"
        ]},
        "local_input_sha256": files,
    }
    (HERE / "strict_replica_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    (HERE / "strict_replica_caption.md").write_text(
        """# Strict six-axis replica of the original MACE-MD figure\n\n"
        "Panels preserve the supplied reference layout: temperature, pressure, normalized HFACF, thermal-conductivity integral, MSD, and normalized VACF. Temperature and pressure use the four current 1 ns trajectories; pressure is derived from the extxyz stress metadata as -Tr(stress)/3. The current four-model 1 ns archive contains no heat-flux time series, so HFACF and κ are shown as unavailable rather than reconstructed from unrelated data. MSD/VACF retain the available current-trajectory curves and explicitly mark failure-mode models. Foundation-OMAT first becomes unstable at 8.8 ps; From-scratch first becomes unstable at 462.2 ps.\n"""
    )


def main() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.0), facecolor="white")
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.12, top=0.98, wspace=0.32, hspace=0.45)
    panel_temperature(axes[0, 0])
    unavailable_panel(axes[0, 1], "Normalized HFACF", "c")
    panel_msd(axes[0, 2])
    panel_pressure(axes[1, 0])
    unavailable_panel(axes[1, 1], "Kappa (W/m·K)", None)
    panel_vacf(axes[1, 2])
    for ext, kwargs in [("pdf", {}), ("svg", {}), ("png", {"dpi": 500})]:
        fig.savefig(OUT / f"mace_1ns_strict_six_axis_replica.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    write_manifest()


if __name__ == "__main__":
    main()
