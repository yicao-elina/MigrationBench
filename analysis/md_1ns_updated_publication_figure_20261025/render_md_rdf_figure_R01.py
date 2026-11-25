#!/usr/bin/env python3
"""Render the updated six-panel MD figure with Cr--Cr RDF in panel E.

The transport panels reuse the cached, seed-aware release used for the prior
figure.  The RDF panel is sourced from the remote raw extxyz trajectory audit
under rdf_r01/rdf_groups.csv.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from render_cached_md_figure import (
    CACHE,
    COLORS,
    MODELS,
    OUT,
    RES,
    SCRATCH,
    aggregate,
    axes_style,
    band,
    dxy,
    stats,
)

RDF_GROUPS = OUT / "rdf_r01" / "rdf_groups.csv"
RDF_RUNS = OUT / "rdf_r01" / "rdf_runs.csv"
RDF_META = OUT / "rdf_r01" / "rdf_meta.tsv"
RDF_INPUT = OUT / "rdf_r01" / "rdf_input_manifest.tsv"
RDF_CODE = OUT / "rdf_r01" / "compute_rdf.cpp"
PRESSURE_RUNS = OUT / "rdf_r01" / "pressure_runs.csv"
PRESSURE_CODE = OUT / "rdf_r01" / "compute_pressure.cpp"
OUTPUT_STEM = RES / "fig_md_1ns_updated_A_to_F_RDF_R01"


def read_rdf():
    data = {}
    with RDF_GROUPS.open(newline="") as f:
        for row in csv.DictReader(f):
            fam = row["family"]
            data.setdefault(fam, {"r": [], "mean": [], "sd": []})
            data[fam]["r"].append(float(row["r_A"]))
            data[fam]["mean"].append(float(row["g_mean"]))
            data[fam]["sd"].append(float(row["g_sd"]))
    return {k: {kk: np.asarray(vv) for kk, vv in v.items()} for k, v in data.items()}


def pressure_aggregate(family):
    by_run = {}
    with PRESSURE_RUNS.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["family"] != family:
                continue
            by_run.setdefault(row["run"], [[], []])
            by_run[row["run"]][0].append(float(row["time_ps"]))
            by_run[row["run"]][1].append(float(row["pressure_GPa"]))
    grid = np.linspace(220, 1000, 781)
    vals = []
    for times, pressures in by_run.values():
        order = np.argsort(times)
        x = np.asarray(times)[order]
        y = np.asarray(pressures)[order]
        vals.append(np.interp(grid, x, y, left=np.nan, right=np.nan))
    a = np.vstack(vals)
    return grid, np.nanmean(a, axis=0), np.nanstd(a, axis=0), len(vals)


def render():
    rdf = read_rdf()
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, aa = plt.subplots(2, 3, figsize=(13.2, 7.7), dpi=400)
    aa = aa.ravel()

    for fam in ["FT-600K", "FT-MultiT", "Foundation*"]:
        label = "Foundation" if fam == "Foundation*" else fam
        x, m, s, _ = aggregate(fam, "temperature")
        band(aa[0], x, m, s, COLORS[label], label)
    aa[0].axhline(600, color="#666", ls=(0, (3, 2)), lw=1)
    aa[0].set_xlim(0, 1000)
    aa[0].set_ylim(200, 720)
    axes_style(aa[0], "Temperature stability", "A", "Time (ps)", "Temperature (K)")

    for fam in ["FT-600K", "FT-MultiT", "Foundation"]:
        x, m, s, _ = pressure_aggregate(fam)
        band(aa[1], x, m, s, COLORS[fam], fam)
    aa[1].axhline(0, color="#666", ls=(0, (3, 2)), lw=1)
    aa[1].set_xlim(200, 1000)
    aa[1].set_ylim(-.25, .35)
    axes_style(aa[1], "Pressure stability", "B", "Time (ps)", "Hydrostatic pressure (GPa)")

    for fam in ["FT-600K", "FT-MultiT", "Foundation*"]:
        label = "Foundation" if fam == "Foundation*" else fam
        x, m, s, _ = aggregate(fam, "msd")
        band(aa[2], x, m, s, COLORS[label], label)
    aa[2].set_xlim(0, 800)
    aa[2].set_ylim(bottom=0)
    axes_style(aa[2], "Mean-square displacement", "C", "Lag time (ps)", r"MSD ($\AA^2$)")

    for fam in ["FT-600K", "FT-MultiT", "Foundation*"]:
        label = "Foundation" if fam == "Foundation*" else fam
        x, m, s, _ = aggregate(fam, "vacf")
        band(aa[3], x, m, s, COLORS[label], label)
    aa[3].axhline(0, color="#666", ls=(0, (3, 2)), lw=1)
    aa[3].set_xlim(0, 800)
    aa[3].set_ylim(-.25, 1.05)
    axes_style(aa[3], "Velocity autocorrelation", "D", "Lag time (ps)", "Normalized VACF")

    fams = ["FT-600K", "FT-MultiT", "Foundation*"]
    pos = np.arange(3)
    for fam in ["Foundation", "FT-600K", "FT-MultiT", "Scratch-600K", "Scratch-MultiT"]:
        if fam not in rdf:
            continue
        r = rdf[fam]["r"]
        g = rdf[fam]["mean"]
        sd = rdf[fam]["sd"]
        # The short-range Cr--Cr peak spans two orders of magnitude across
        # models.  A log y-axis keeps both the first coordination peak and
        # the weaker medium-range structure readable without clipping data.
        g_plot = np.where(g > 0, g, np.nan)
        lo = np.maximum(g - sd, 0.05)
        hi = np.maximum(g + sd, 0.05)
        aa[4].fill_between(r, lo, hi, color=COLORS[fam], alpha=.18, linewidth=0)
        aa[4].plot(r, g_plot, color=COLORS[fam], lw=1.8, label=fam)
    aa[4].set_xlim(0, 10)
    aa[4].set_yscale("log")
    aa[4].set_ylim(.05, 250)
    aa[4].axhline(1, color="#666", ls=(0, (3, 2)), lw=1)
    axes_style(aa[4], r"Cr–Cr radial distribution", "E", r"r ($\AA$)", r"$g_{\mathrm{Cr-Cr}}(r)$")

    for i, fam in enumerate(fams):
        label = "Foundation" if fam == "Foundation*" else fam
        m1, s1, _ = stats(fam, 200, 500)
        m2, s2, _ = stats(fam, 500, 900)
        ratio = m2 / m1
        er = ratio * np.sqrt((s1 / m1) ** 2 + (s2 / m2) ** 2)
        aa[5].errorbar(i, ratio, yerr=er, fmt="o", color=COLORS[label], ms=7, mfc="white" if fam == "Foundation*" else COLORS[label], mew=1.5, capsize=3, lw=1.2)
    aa[5].axhline(1, color="#666", ls=(0, (3, 2)), lw=1)
    aa[5].set_xticks(pos, ["FT-600K", "FT-MultiT", "Foundation*"])
    aa[5].set_yscale("log")
    aa[5].set_ylim(.05, 20)
    axes_style(aa[5], r"$D_{xy}$ window ratio", "F", None, r"500–900 / 200–500 ps")

    handles = []
    labels = []
    for label in ["Foundation", "FT-600K", "FT-MultiT", "Scratch-600K", "Scratch-MultiT"]:
        h, = aa[0].plot([], [], color=COLORS[label], lw=2.4, label=label)
        handles.append(h)
        labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.5, .008), fontsize=10, handlelength=2.4, columnspacing=1.6)
    fig.subplots_adjust(left=.075, right=.985, top=.965, bottom=.15, wspace=.32, hspace=.38)

    RES.mkdir(exist_ok=True)
    outs = [OUTPUT_STEM.with_suffix(ext) for ext in (".pdf", ".png", ".svg")]
    fig.savefig(outs[0], bbox_inches="tight", facecolor="white")
    fig.savefig(outs[1], dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(outs[2], bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = OUT / "md_1ns_family_summary_RDF_R01.csv"
    with summary.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "rdf_source", "pressure_source", "rdf_pair", "rdf_rmax_A", "rdf_frame_aggregation", "status"])
        for fam in ["Foundation", "FT-600K", "FT-MultiT", "Scratch-600K", "Scratch-MultiT"]:
            status = "CONVERGED_1NS" if fam.startswith("FT-") else ("PROVISIONAL_1NS" if fam == "Foundation" else "SCIENTIFIC_FAILURE_NO_1NS")
            w.writerow([fam, str(RDF_GROUPS), str(PRESSURE_RUNS), "Cr-Cr", 10, "seed/run mean with run SD band", status])

    manifest = {
        "figure": "fig_md_1ns_updated_A_to_F_RDF_R01",
        "panel_change": "Panel B is hydrostatic pressure stability from raw extxyz stress; panel E replaces the prior observed-horizon panel with Cr-Cr RDF; panel F remains the Dxy window ratio.",
        "style_reference": "FIG168_MODEL_STYLE_YELLOW_SCRATCH_R04.json",
        "palette": COLORS,
        "rdf": {
            "pair": "Cr-Cr",
            "source": str(RDF_GROUPS),
            "remote_compute": "Rockfish raw extxyz, triclinic minimum-image distances, rmax=12 A, dr=0.1 A, frame_stride=20; panel displays r<=10 A",
            "metadata": str(RDF_META),
            "status": "FT 3-seed release plus Foundation true zero-shot and Scratch smoke/failure-mode trajectories; Scratch is not a 1 ns acceptance claim",
        },
        "pressure": {
            "source": str(PRESSURE_RUNS),
            "definition": "P = -trace(stress)/3, converted from eV/A^3 to GPa using 160.21766208",
            "coverage": "production frames from step 220000; run mean and run SD bands",
        },
        "scientific_boundaries": {
            "FT": "accepted 1 ns three-seed trajectories",
            "Foundation": "current true zero-shot continuation found on Rockfish at step 1,000,000; retain provisional comparator status pending full independent acceptance audit",
            "Scratch": "six current smoke runs are short failure-mode trajectories; no 1 ns Dxy or stability claim",
        },
        "cached_source_directory": str(CACHE),
        "source_artifacts": [],
        "outputs": [],
    }
    source_paths = [RDF_INPUT, RDF_CODE, PRESSURE_CODE, RDF_RUNS, RDF_GROUPS, RDF_META, PRESSURE_RUNS]
    for path in source_paths:
        if path.exists():
            manifest["source_artifacts"].append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for path in outs + [summary]:
        manifest["outputs"].append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT / "FIGURE_MANIFEST_RDF_R01.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    render()
