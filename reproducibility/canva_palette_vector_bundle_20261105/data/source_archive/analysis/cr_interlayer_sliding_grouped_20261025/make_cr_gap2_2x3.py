#!/usr/bin/env python3
"""Render the Cr gap-2 fixed-path/self-relaxed 2x3 comparison figure.

The two observables are intentionally kept separate.  All plotted rows are
derived from existing, hashed project tables or Rockfish-exported manifests;
no synthetic values are introduced.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

FIXED = ROOT / "source_tables" / "fixed_profiles_R04.csv"
FOUNDATION = ROOT / "source_tables" / "foundation_2x2_IMAGE_PROFILES.tsv"
DFT = ROOT / "source_tables" / "combined_sb2te3_energy.csv"
SCRATCH_SELF = ROOT / "raw" / "rockfish" / "scratch_lmax1_g4_self_20261025_r02" / "runs" / "self" / "cr_in_gap"
FT_SELF = ROOT.parent / "task3_endpoint_only_neb_20261016"  # unused; explicit below
FT_PROFILES = ROOT.parent / "task3_endpoint_only_neb" / "neb_fixed_vs_self_matrix_20261019" / "derived_r01" / "TIER1_IMAGE_PROFILES.tsv"

COLORS = {
    "foundation": "#CF4520",
    "scratch_600k": "#FF9F1B",
    "scratch_multit": "#F4C95D",
    "ft_600k": "#002E71",
    "ft_multit": "#67ACE5",
    "dft": "#222222",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def mean_sd(rows: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = sorted(rows)
    x = np.asarray([a for a, _ in rows], dtype=float)
    y = np.asarray([b for _, b in rows], dtype=float)
    return x, y, np.zeros_like(y)


def grouped(records: list[dict], xkey: str, ykey: str, group_key: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    groups: dict[str, dict[float, list[float]]] = {}
    for r in records:
        g = r[group_key]
        x = float(r[xkey])
        y = float(r[ykey])
        groups.setdefault(g, {}).setdefault(x, []).append(y)
    out = {}
    for g, points in groups.items():
        xs = np.asarray(sorted(points), dtype=float)
        vals = np.asarray([points[x] for x in xs], dtype=float)
        out[g] = (xs, vals.mean(axis=1), vals.std(axis=1, ddof=1) if vals.shape[1] > 1 else np.zeros(len(xs)))
    return out


def fixed_data() -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    rows = [r for r in read_csv(FIXED) if r["system_id"] == "cr_in_gap"]
    for r in rows:
        r["image_fraction"] = float(r["image_index"]) / 9.0
        if r["model_family"] == "foundation": r["plot_group"] = "foundation"
        elif r["model_family"] == "scratch": r["plot_group"] = r["scratch_subfamily"]
        elif r["model_family"] == "ft_600k": r["plot_group"] = "ft_600k"
        elif r["model_family"] == "ft_multit": r["plot_group"] = "ft_multit"
    panel = {
        "Foundation": grouped([r for r in rows if r["plot_group"] == "foundation"], "image_fraction", "relative_energy_eV", "plot_group"),
        "Scratch": grouped([r for r in rows if r["plot_group"] in {"scratch_600k", "scratch_multit"}], "image_fraction", "relative_energy_eV", "plot_group"),
        "FT": grouped([r for r in rows if r["plot_group"] in {"ft_600k", "ft_multit"}], "image_fraction", "relative_energy_eV", "plot_group"),
    }
    dft_rows = read_csv(DFT)
    dft = {"cr_in_gap": (np.asarray([float(r["Reaction_Coord"]) for r in dft_rows]), np.asarray([float(r["2.cr-in-gap"]) for r in dft_rows]), np.zeros(len(dft_rows)))}
    return panel, dft


def foundation_self() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rows = [r for r in read_csv(FOUNDATION, "\t") if r["observable"] == "SELF_RELAXED_ENDPOINT_ONLY_NEB" and r["system"] == "cr_in_gap"]
    for r in rows: r["plot_group"] = "foundation"
    return grouped(rows, "image_fraction", "relative_energy_eV", "plot_group")


def scratch_self() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    records = []
    for manifest in sorted(SCRATCH_SELF.glob("*/attempt_*/endpoint_only_manifest.json")):
        d = json.loads(manifest.read_text())
        family = "scratch_600k" if "600k" in d["model_id"] else "scratch_multit"
        for p in d["engine"]["profile"]:
            records.append({"x": p["image_index"] / 9.0, "y": p["rel_energy_eV"], "plot_group": family})
    return grouped(records, "x", "y", "plot_group")


def ft_self() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rows = [r for r in read_csv(FT_PROFILES, "\t") if r["path_id"] == "cr_canonical_in_gap" and r["observable"] == "SELF_RELAXED_ENDPOINT_ONLY_NEB"]
    for r in rows: r["plot_group"] = "ft_600k" if "ft600k" in r["model_id"] else "ft_multit"
    return grouped(rows, "image_index", "relative_energy_eV", "plot_group")


def draw_panel(ax, title: str, data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], dft: tuple[np.ndarray, np.ndarray, np.ndarray], label: str) -> None:
    x_dft, y_dft, _ = dft
    ax.plot(x_dft, y_dft, color=COLORS["dft"], lw=1.45, ls=(0, (4, 2)), marker="D", ms=3.6, mfc="white", mec=COLORS["dft"], mew=0.8, zorder=5)
    order = ["foundation", "scratch_600k", "scratch_multit", "ft_600k", "ft_multit"]
    for key in order:
        if key not in data: continue
        x, y, sd = data[key]
        c = COLORS[key]
        ax.fill_between(x, y - sd, y + sd, color=c, alpha=0.20, lw=0, zorder=2)
        ax.plot(x, y, color=c, lw=2.0, marker="o", ms=4.3, mfc="white", mec=c, mew=0.95, zorder=5)
    values = [*y_dft]
    for x, y, sd in data.values(): values.extend((y - sd).tolist()); values.extend((y + sd).tolist())
    lo, hi = min(values), max(values)
    pad = max(0.18, (hi - lo) * 0.12)
    ax.set_xlim(0, 1)
    ax.set_ylim(np.floor((lo - pad) * 2) / 2, np.ceil((hi + pad) * 2) / 2)
    ax.set_xticks([0, .25, .5, .75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=8)
    ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    ax.set_xlabel("Reaction Coordinate", fontsize=11.5, fontweight="bold", labelpad=6)
    ax.set_ylabel("Relative Energy (eV)", fontsize=11.5, fontweight="bold", labelpad=6)
    ax.minorticks_on(); ax.tick_params(which="major", direction="out", length=4.5, width=1.0); ax.tick_params(which="minor", direction="out", length=2.5, width=.8)
    for s in ax.spines.values(): s.set_linewidth(1.05); s.set_color("#111111")
    ax.grid(False)


def main() -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 10.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    fixed, dft = fixed_data()
    self_data = {"Foundation": foundation_self(), "Scratch": scratch_self(), "FT": ft_self()}
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.4), sharex=False, sharey=False)
    titles = ["Foundation", "Scratch (600K / MultiT)", "FT (600K / MultiT)"]
    for j, title in enumerate(titles): draw_panel(axes[0, j], title, fixed[title.split()[0]], dft["cr_in_gap"], chr(ord("a") + j))
    for j, title in enumerate(titles): draw_panel(axes[1, j], title, self_data[title.split()[0]], dft["cr_in_gap"], chr(ord("d") + j))
    axes[0, 0].text(-0.27, 0.5, "Fixed path", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="center", fontsize=13, fontweight="bold")
    axes[1, 0].text(-0.27, 0.5, "Self-relaxed", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="center", fontsize=13, fontweight="bold")
    handles = [Line2D([0], [0], color=COLORS["dft"], lw=1.45, ls=(0, (4, 2)), marker="D", ms=3.6, mfc="white", mec=COLORS["dft"], label="DFT reference"),
               Line2D([0], [0], color=COLORS["foundation"], lw=2, marker="o", mfc="white", label="Foundation"),
               Line2D([0], [0], color=COLORS["scratch_600k"], lw=2, marker="o", mfc="white", label="Scratch-600K"),
               Line2D([0], [0], color=COLORS["scratch_multit"], lw=2, marker="o", mfc="white", label="Scratch-MultiT"),
               Line2D([0], [0], color=COLORS["ft_600k"], lw=2, marker="o", mfc="white", label="FT-600K"),
               Line2D([0], [0], color=COLORS["ft_multit"], lw=2, marker="o", mfc="white", label="FT-MultiT")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .985), ncol=3, frameon=False, fontsize=9.3, handlelength=2.2, handletextpad=.45, columnspacing=1.25)
    fig.subplots_adjust(left=.105, right=.985, bottom=.105, top=.80, wspace=.32, hspace=.42)
    stem = OUT / "cr_gap2_foundation_scratch_ft_fixed_vs_self_2x3_R01"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    manifest = {"figure": str(stem.with_suffix('.pdf')), "observable_rows": ["FIXED_PATH_SINGLE_POINT", "SELF_RELAXED_ENDPOINT_ONLY_NEB"], "system": "Cr-in-gap / gap-2", "dft_reference": "combined_sb2te3_energy.csv; COMPLETED_BUT_SCF_WARNING", "self_relaxed_boundary": "endpoint-only MLIP NEB proxy; not a DFT MEP; no same-geometry DFT rescoring", "sources": {str(p): sha256(p) for p in [FIXED, FOUNDATION, DFT, FT_PROFILES]}, "scratch_self_source": str(SCRATCH_SELF), "colors": COLORS}
    (OUT / "cr_gap2_foundation_scratch_ft_fixed_vs_self_2x3_R01.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
