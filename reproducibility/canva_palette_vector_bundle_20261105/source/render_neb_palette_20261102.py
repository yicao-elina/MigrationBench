#!/usr/bin/env python3
"""Re-render the locally reproducible NEB/sliding figures with the new grammar."""

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

PALETTE = {
    "Foundation": "#8FC7AB",
    "Fine-tuning": "#29A7B6",
    "Scratch": "#327FB1",
    "DFT": "#303840",
}


def load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CR3 = load_module(
    "cr_three_path_source",
    Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/cr_interlayer_sliding_grouped_20261025/make_cr_three_path_2x3_R02.py"),
)
CR5 = load_module(
    "cr_three_path_r05_source",
    Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/cr_interlayer_sliding_grouped_20261025/make_cr_three_path_2x3_R05.py"),
)
CR2 = load_module(
    "cr_grouped_source",
    Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/cr_interlayer_sliding_grouped_20261025/scripts/build_cr_grouped_r04.py"),
)
FOUND = load_module(
    "foundation_grouped_source",
    Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/foundation_interlayer_sliding_10model_20261024/scripts/plot_interlayer_grouped_profile_atlas_r01.py"),
)


def safe_current_ft_self():
    """Load accepted FT self profiles while retaining the source failure boundary."""
    records = []
    accepted = set()
    for collector in [CR3.FT / "r02" / "FT_SELF_R02_COLLECT.tsv", CR3.FT / "r03" / "FT_SELF_R03_RETRY_COLLECT.tsv"]:
        if not collector.exists():
            continue
        for row in CR3.read_tsv(collector):
            if row.get("scientific_status") == "CONVERGED" and row.get("publication_status") == "PRIMARY_EVIDENCE":
                accepted.add((row["system"], row["model_id"]))
    for manifest in sorted(CR3.FT.rglob("endpoint_only_manifest.json")):
        try:
            d = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            continue
        engine = d.get("engine", {})
        model_id = manifest.parent.parent.name
        system = manifest.parent.parent.parent.name
        key = {"in_gap": "in_gap", "gap_to_ql": "gap_to_ql", "within_ql": "within_ql"}.get(system)
        if key is None:
            continue
        accepted_here = (system, model_id) in accepted
        if d.get("campaign_id") == "ft_interlayer_sliding_self_20261025_r03_retry":
            accepted_here = False
            status, pub, boundary = "FAILURE_MODE", "DIAGNOSTIC_ONLY", "CURRENT_A2_FT_SELF_R03_RETRY; non-converged residual/scientific gate"
        else:
            status = d.get("scientific_status", "UNKNOWN")
            pub = d.get("publication_status", "UNKNOWN")
            boundary = "CURRENT_A2_FT_SELF_R02; endpoint-only MLIP NEB proxy"
        if not accepted_here:
            continue
        group = "ft_600k" if model_id.startswith("ft_600k_") else "ft_multit"
        for point in engine.get("profile", []):
            CR3.add_record(records, path_key=key, observable="SELF_RELAXED_ENDPOINT_ONLY_NEB", group=group, model_id=model_id, image_index=int(point["image_index"]), energy=float(point["rel_energy_eV"]), source=str(manifest), status=status, publication_status=pub, boundary=boundary)
    return records


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def axes_style(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["DFT"])
        spine.set_linewidth(1.15)
    ax.tick_params(direction="out", width=1.1, length=4.0)


def save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=320 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem


def cr_three_path():
    records = CR5.load_current_foundation_self() + CR5.load_legacy_profiles() + CR5.load_current_ft_self() + CR5.load_current_scratch_600k()
    dft = CR5.dft_reference()
    group_color = {"foundation": PALETTE["Foundation"], "ft_600k": PALETTE["Fine-tuning"], "ft_multit": PALETTE["Fine-tuning"], "scratch_600k": PALETTE["Scratch"], "scratch_multit": PALETTE["Scratch"]}
    group_marker = {"foundation": "D", "ft_600k": "o", "ft_multit": "o", "scratch_600k": "o", "scratch_multit": "o"}
    group_face = {"foundation": "white", "ft_600k": "white", "ft_multit": PALETTE["Fine-tuning"], "scratch_600k": "white", "scratch_multit": PALETTE["Scratch"]}
    group_label = {"foundation": "Foundation", "ft_600k": "Fine-tuning · 600 K", "ft_multit": "Fine-tuning · Multi-T", "scratch_600k": "Scratch · 600 K", "scratch_multit": "Scratch · Multi-T"}
    method_style = {"FIXED_PATH_SINGLE_POINT": "-", "SELF_RELAXED_ENDPOINT_ONLY_NEB": "--"}
    fig, axs = plt.subplots(2, 3, figsize=(12.2, 7.7), squeeze=False)
    for row, obs in enumerate(("FIXED_PATH_SINGLE_POINT", "SELF_RELAXED_ENDPOINT_ONLY_NEB")):
        for col, (key, label, _pid) in enumerate(CR5.PATHS):
            ax = axs[row, col]
            axes_style(ax)
            xd, yd = dft[key]
            ax.plot(xd, yd, color=PALETTE["DFT"], lw=1.55, ls=(0, (4, 2)), marker="D", ms=3.8, mfc="white", mec=PALETTE["DFT"], mew=.85, zorder=5)
            data = CR5.grouped(records, obs, key)
            values = list(yd)
            for group, profile in data.items():
                x, y, sd = profile["x"], profile["mean"], profile["sd"]
                c = group_color[group]
                ax.fill_between(x, y - sd, y + sd, color=c, alpha=.18, lw=0, zorder=2)
                ax.plot(x, y, color=c, lw=2.0, ls=method_style[obs], marker=group_marker[group], ms=4.5, mfc=group_face[group], mec=c, mew=.95, zorder=5)
                values.extend((y - sd).tolist()); values.extend((y + sd).tolist())
            lo, hi = min(values), max(values); pad = max(.18, (hi - lo) * .12)
            ax.set_xlim(0, 1); ax.set_ylim(np.floor((lo - pad) * 2) / 2, np.ceil((hi + pad) * 2) / 2)
            ax.set_xticks([0, .25, .5, .75, 1], ["0", ".25", ".5", ".75", "1"])
            ax.set_title(label, fontsize=12.5, fontweight="bold", pad=8)
            ax.set_xlabel("Reaction Coordinate", fontweight="bold"); ax.set_ylabel("Relative Energy (eV)", fontweight="bold")
            ax.text(-.16, 1.04, chr(ord("a") + row * 3 + col), transform=ax.transAxes, fontsize=14, fontweight="bold")
    handles = [
        Line2D([], [], color=group_color[g], marker=group_marker[g], mfc=group_face[g], mec=group_color[g], lw=2, label=group_label[g])
        for g in ("foundation", "ft_600k", "ft_multit", "scratch_600k", "scratch_multit")
    ]
    handles.append(Line2D([], [], color=PALETTE["DFT"], marker="D", mfc="white", mec=PALETTE["DFT"], ls=(0, (4, 2)), lw=1.55, label="DFT-NEB"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .985), ncol=6, frameon=False, fontsize=8.8, handlelength=2.2, columnspacing=.85)
    fig.subplots_adjust(left=.09, right=.985, bottom=.09, top=.79, wspace=.30, hspace=.42)
    return save(fig, OUT / "CANVA-P06_neb_cr_r05_palette_updated")


def cr_sliding():
    rows = CR2.normalize_rows()
    group_color = {"Foundation": PALETTE["Foundation"], "FT-600K": PALETTE["Fine-tuning"], "FT-MultiT": PALETTE["Fine-tuning"], "Scratch-600K": PALETTE["Scratch"], "Scratch-MultiT": PALETTE["Scratch"]}
    group_face = {"Foundation": "white", "FT-600K": "white", "FT-MultiT": PALETTE["Fine-tuning"], "Scratch-600K": "white", "Scratch-MultiT": PALETTE["Scratch"]}
    group_marker = {"Foundation": "D", "FT-600K": "o", "FT-MultiT": "o", "Scratch-600K": "o", "Scratch-MultiT": "o"}
    fig, axs = plt.subplots(2, 2, figsize=(11.8, 8.7), squeeze=False)
    for idx, (method, system, panel) in enumerate((("fixed", "bulk", "a"), ("fixed", "cr_in_gap", "b"), ("self", "bulk", "c"), ("self", "cr_in_gap", "d"))):
        ax = axs.flat[idx]; axes_style(ax)
        x, profiles = CR2.aggregate(rows, method, system); xd, yd = CR2.reference(system)
        ax.plot(xd, yd, color=PALETTE["DFT"], lw=2.0, ls=(0, (4, 2)), marker="D", ms=4, mfc="white", mec=PALETTE["DFT"], label="DFT reference", zorder=5)
        values = list(yd)
        for group, rec in profiles.items():
            c = group_color[group]; marker = group_marker[group]; face = group_face[group]
            ax.fill_between(x, rec["mean"] - rec["sd"], rec["mean"] + rec["sd"], color=c, alpha=.18, linewidth=0, zorder=2)
            ax.plot(x, rec["mean"], color=c, lw=2, ls="-" if method == "fixed" else "--", marker=marker, ms=4.5, mfc=face, mec=c, mew=.95, label=group, zorder=5)
            values.extend((rec["mean"] - rec["sd"]).tolist()); values.extend((rec["mean"] + rec["sd"]).tolist())
        lo, hi = min(values), max(values); ax.set_ylim(np.floor(lo * 2) / 2, np.ceil(hi * 2) / 2); ax.set_xlim(0, 1)
        ax.set_xticks([0, .25, .5, .75, 1], ["0", ".25", ".5", ".75", "1"])
        ax.set_title(("Bulk" if system == "bulk" else "Cr-in-gap") + " · " + ("fixed path" if method == "fixed" else "self-relaxed"), fontweight="bold")
        ax.set_xlabel("Reaction Coordinate", fontweight="bold"); ax.set_ylabel(r"$\Delta E$ (eV)", fontweight="bold"); ax.text(-.16, 1.04, panel, transform=ax.transAxes, fontsize=15, fontweight="bold")
    handles = [
        Line2D([], [], color=PALETTE["Foundation"], marker="D", mfc="white", mec=PALETTE["Foundation"], lw=2, label="Foundation"),
        Line2D([], [], color=PALETTE["Fine-tuning"], marker="o", mfc="white", mec=PALETTE["Fine-tuning"], lw=2, label="Fine-tuning · 600 K"),
        Line2D([], [], color=PALETTE["Fine-tuning"], marker="o", mfc=PALETTE["Fine-tuning"], mec=PALETTE["Fine-tuning"], lw=2, label="Fine-tuning · Multi-T"),
        Line2D([], [], color=PALETTE["Scratch"], marker="o", mfc="white", mec=PALETTE["Scratch"], lw=2, label="Scratch · 600 K"),
        Line2D([], [], color=PALETTE["Scratch"], marker="o", mfc=PALETTE["Scratch"], mec=PALETTE["Scratch"], lw=2, label="Scratch · Multi-T"),
        Line2D([], [], color=PALETTE["DFT"], marker="D", mfc="white", mec=PALETTE["DFT"], ls=(0, (4, 2)), lw=2, label="DFT reference"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .985), ncol=3, frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=.095, right=.985, top=.84, bottom=.085, wspace=.30, hspace=.42)
    return save(fig, OUT / "CANVA-P09_cr_interlayer_sliding_palette_updated")


def foundation_sliding():
    root = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/foundation_interlayer_sliding_10model_20261024")
    matrix, profiles, qe = FOUND.validate(root)
    colors = {"MACE-MP": "#2F6B4F", "MACE-OMAT-0": "#5F9E73", "MACE-MH": "#8FC7AB"}
    styles = {"MACE-MP": "-", "MACE-OMAT-0": "--", "MACE-MH": "-."}
    markers = {"MACE-MP": "o", "MACE-OMAT-0": "^", "MACE-MH": "s"}
    fig, axs = plt.subplots(2, 2, figsize=(11.8, 8.7), squeeze=False)
    for idx, (method, system, panel) in enumerate((("fixed", "bulk", "a"), ("fixed", "cr_in_gap", "b"), ("self", "bulk", "c"), ("self", "cr_in_gap", "d"))):
        ax = axs.flat[idx]; axes_style(ax)
        x, groups = FOUND.grouped_profiles(method, system, matrix, profiles)
        ref = sorted((row for row in qe if row["system"] == system), key=lambda row: int(row["image_index"]))
        xr = np.asarray([float(row["image_fraction"]) for row in ref]); yr = np.asarray([float(row["relative_energy_eV"]) for row in ref])
        ax.plot(xr, yr, color=PALETTE["DFT"], lw=2.0, ls=(0, (4, 2)), marker="D", ms=4, mfc="white", mec=PALETTE["DFT"], label="DFT reference")
        vals = list(yr)
        for group, rec in groups.items():
            c = colors[group]
            ax.fill_between(x, rec["mean"] - rec["sd"], rec["mean"] + rec["sd"], color=c, alpha=.16, linewidth=0)
            ax.plot(x, rec["mean"], color=c, lw=2, ls=styles[group], marker=markers[group], ms=4.4, mfc="white", mec=c, mew=.95, label=group)
            vals.extend((rec["mean"] - rec["sd"]).tolist()); vals.extend((rec["mean"] + rec["sd"]).tolist())
        ax.set_xlim(0, 1); ax.set_ylim(np.floor(min(vals) * 2) / 2, np.ceil(max(vals) * 2) / 2); ax.set_xticks([0, .25, .5, .75, 1], ["0", ".25", ".5", ".75", "1"])
        ax.set_title(("Bulk" if system == "bulk" else "Cr-in-gap") + " · " + ("fixed path" if method == "fixed" else "self-relaxed"), fontweight="bold")
        ax.set_xlabel("Reaction Coordinate", fontweight="bold"); ax.set_ylabel(r"$\Delta E$ (eV)", fontweight="bold"); ax.text(-.16, 1.04, panel, transform=ax.transAxes, fontsize=15, fontweight="bold")
    handles = [
        Line2D([], [], color="#2F6B4F", marker="o", mfc="white", mec="#2F6B4F", ls="-", lw=2, label="Foundation · MACE-MP"),
        Line2D([], [], color="#5F9E73", marker="^", mfc="white", mec="#5F9E73", ls="--", lw=2, label="Foundation · MACE-OMAT-0"),
        Line2D([], [], color="#8FC7AB", marker="s", mfc="white", mec="#8FC7AB", ls="-.", lw=2, label="Foundation · MACE-MH"),
        Line2D([], [], color=PALETTE["DFT"], marker="D", mfc="white", mec=PALETTE["DFT"], ls=(0, (4, 2)), lw=2, label="DFT reference"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .985), ncol=2, frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=.095, right=.985, top=.84, bottom=.085, wspace=.30, hspace=.42)
    return save(fig, OUT / "CANVA-P10_foundation_interlayer_sliding_palette_updated")


def main():
    outputs = {"CANVA-P06": cr_three_path(), "CANVA-P09": cr_sliding(), "CANVA-P10": foundation_sliding()}
    manifest = {
        "status": "REPRODUCED_FROM_LOCAL_NEb_AND_SLIDING_SOURCES_WITH_PALETTE_REVISION",
        "visual_encoding": {"strategy_color": PALETTE, "temperature_fill": {"600 K": "open", "Multi-T": "filled"}, "foundation_checkpoint": "diamond", "reference": PALETTE["DFT"], "method_line": {"fixed": "solid", "self-relaxed": "dashed"}},
        "outputs": {key: {ext: {"path": str(stem.with_suffix(f".{ext}")), "sha256": sha256(stem.with_suffix(f".{ext}")), "bytes": stem.with_suffix(f".{ext}").stat().st_size} for ext in ("pdf", "svg", "png")} for key, stem in outputs.items()},
    }
    (CAMPAIGN / "NEB_PALETTE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
