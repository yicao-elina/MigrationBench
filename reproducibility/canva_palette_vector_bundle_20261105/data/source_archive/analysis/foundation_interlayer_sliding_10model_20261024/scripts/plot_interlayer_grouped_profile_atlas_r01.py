#!/usr/bin/env python3
"""Grouped 2×2 profile figure using the established MACE class grammar.

The ten campaign models are grouped as MACE-MP, MACE-OMAT-0, and MACE-MH.
Each curve is the pointwise class mean and each translucent band is the sample
SD across the models in that class. Fixed-path and self-relaxed endpoint-only
profiles remain separate rows; the QE trace remains a screened SCF-warning
reference rather than a ground-truth label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODEL_GROUPS = {
    "MACE-MP": [
        "foundation_agnesi_large",
        "foundation_agnesi_medium_density",
        "foundation_agnesi_small_density",
        "foundation_agnesi_medium",
        "foundation_mp0b3",
    ],
    "MACE-OMAT-0": [
        "foundation_omat0_medium",
        "foundation_omat0_small",
        "foundation_matpes_pbe_omat_ft",
    ],
    "MACE-MH": ["foundation_mh0", "foundation_mh1"],
}
GROUP_COLORS = {
    "MACE-MP": "#D5A238",
    "MACE-OMAT-0": "#D85A45",
    "MACE-MH": "#5D8B56",
}
METHODS = {
    "fixed": "FIXED_PATH_SINGLE_POINT",
    "self": "SELF_RELAXED_ENDPOINT_ONLY_NEB",
}
SYSTEMS = ("bulk", "cr_in_gap")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    matrix = read_tsv(root / "results/tables/RESULT_MATRIX.tsv")
    profiles = read_tsv(root / "results/tables/IMAGE_PROFILES.tsv")
    qe = read_tsv(root / "results/tables/QE_REFERENCE.tsv")
    if len(matrix) != 40 or len(profiles) != 400 or len(qe) != 20:
        raise ValueError(f"expected 40 matrix, 400 profiles, 20 QE rows; got {len(matrix)}, {len(profiles)}, {len(qe)}")
    for method, observable in METHODS.items():
        rows = [row for row in matrix if row["observable"] == observable]
        expected_status = "FINITE_DIAGNOSTIC" if method == "fixed" else "CONVERGED"
        if len(rows) != 20 or any(row["scientific_status"] != expected_status for row in rows):
            raise ValueError(f"{method} acceptance gate failed")
        for row in rows:
            selected = [item for item in profiles if item["record_id"] == row["record_id"]]
            if len(selected) != 10 or [int(item["image_index"]) for item in selected] != list(range(10)):
                raise ValueError(f"profile completeness/order failed for {row['record_id']}")
    return matrix, profiles, qe


def grouped_profiles(method: str, system: str, matrix, profiles):
    observable = METHODS[method]
    records = {
        row["model_id"]: row["record_id"]
        for row in matrix
        if row["observable"] == observable and row["system"] == system
    }
    result = {}
    x = None
    for group, model_ids in MODEL_GROUPS.items():
        arrays = []
        for model_id in model_ids:
            rows = sorted(
                (row for row in profiles if row["record_id"] == records[model_id]),
                key=lambda row: int(row["image_index"]),
            )
            if x is None:
                x = np.asarray([float(row["image_fraction"]) for row in rows])
            arrays.append([float(row["relative_energy_eV"]) for row in rows])
        values = np.asarray(arrays, dtype=float)
        result[group] = {"mean": values.mean(axis=0), "sd": values.std(axis=0, ddof=1)}
    return x, result


def draw_profile(ax, method: str, system: str, matrix, profiles, qe, panel_label: str) -> dict:
    x, grouped = grouped_profiles(method, system, matrix, profiles)
    lines = {}
    for group in MODEL_GROUPS:
        color = GROUP_COLORS[group]
        record = grouped[group]
        ax.fill_between(x, record["mean"] - record["sd"], record["mean"] + record["sd"], color=color, alpha=0.20, linewidth=0, zorder=2)
        (line,) = ax.plot(x, record["mean"], color=color, lw=2.0, marker="o", markersize=4.3, markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.95, label=group, zorder=5)
        lines[group] = line

    reference_rows = sorted((row for row in qe if row["system"] == system), key=lambda row: int(row["image_index"]))
    reference_x = np.asarray([float(row["image_fraction"]) for row in reference_rows])
    reference_y = np.asarray([float(row["relative_energy_eV"]) for row in reference_rows])
    (reference_line,) = ax.plot(reference_x, reference_y, color="#222222", lw=2.2, ls=(0, (4, 2)), marker="D", markersize=3.9, markerfacecolor="white", markeredgecolor="#222222", markeredgewidth=1.2, label="DFT-NEB", zorder=5)

    ax.set_xlim(0.0, 1.0)
    ax.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0))
    ax.set_xticklabels(("0", "0.25", "0.5", "0.75", "1"))
    ax.tick_params(axis="both", which="major", direction="out", length=5.0, width=1.4, pad=3, labelsize=16.0)
    ax.tick_params(axis="both", which="minor", direction="out", length=3.0, width=1.1)
    ax.minorticks_on()
    ax.grid(False)
    for text in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        text.set_fontfamily("Arial")
        text.set_fontsize(16.0)
        text.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111111")
        spine.set_linewidth(1.5)
    method_title = "fixed-path" if method == "fixed" else "self-relaxed path"
    system_title = "Bulk" if system == "bulk" else "Cr-in-gap"
    ax.set_title(f"{system_title} · {method_title}", fontsize=18.0, fontweight="bold", pad=8, family="Arial")
    ax.set_xlabel("Reaction Coordinate", fontsize=16.8, fontweight="bold", labelpad=7, family="Arial")
    ax.set_ylabel("$\\Delta E$ (eV)", fontsize=17.0, fontweight="bold", labelpad=8, family="Arial")
    ax.text(-0.16, 1.06, panel_label, transform=ax.transAxes, fontsize=19.0, family="Arial", fontweight="bold", va="bottom")
    return {"lines": lines, "reference": reference_line}


def save(fig: plt.Figure, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(ext) for ext in (".png", ".svg", ".pdf")]
    for path in paths:
        fig.savefig(path, dpi=400 if path.suffix == ".png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def render(root: Path) -> dict[str, object]:
    matrix, profiles, qe = validate(root)
    out = root / "results/figures/plot_atlas_grouped_r04"
    out.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.0), sharex=False, sharey=False)
    first = draw_profile(axes[0, 0], "fixed", "bulk", matrix, profiles, qe, "a")
    draw_profile(axes[0, 1], "fixed", "cr_in_gap", matrix, profiles, qe, "b")
    draw_profile(axes[1, 0], "self", "bulk", matrix, profiles, qe, "c")
    draw_profile(axes[1, 1], "self", "cr_in_gap", matrix, profiles, qe, "d")
    handles = [*first["lines"].values(), first["reference"]]
    labels = [handle.get_label() for handle in handles]
    legend = fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4, frameon=False, fontsize=16.0, handlelength=2.2, columnspacing=1.25, handletextpad=0.45)
    for text in legend.get_texts():
        text.set_fontfamily("Arial")
        text.set_fontweight("bold")
    fig.subplots_adjust(left=0.095, right=0.985, top=0.88, bottom=0.085, wspace=0.34, hspace=0.48)
    outputs.extend(save(fig, out / "interlayer_sliding_10model_grouped_2x2_plot_atlas_r04"))

    for method, system, label in (("fixed", "bulk", "a"), ("fixed", "cr_in_gap", "b"), ("self", "bulk", "c"), ("self", "cr_in_gap", "d")):
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        result = draw_profile(ax, method, system, matrix, profiles, qe, label)
        legend = ax.legend(handles=[*result["lines"].values(), result["reference"]], frameon=False, fontsize=16.0, ncol=2, handlelength=2.2, columnspacing=1.25, handletextpad=0.45, loc="upper center", bbox_to_anchor=(0.5, -0.23))
        for text in legend.get_texts():
            text.set_fontfamily("Arial")
            text.set_fontweight("bold")
        fig.subplots_adjust(left=0.15, right=0.98, top=0.90, bottom=0.31)
        outputs.extend(save(fig, out / f"{method}_{system}_grouped_plot_atlas_r04"))

    source_paths = [root / "results/tables/RESULT_MATRIX.tsv", root / "results/tables/IMAGE_PROFILES.tsv", root / "results/tables/QE_REFERENCE.tsv"]
    manifest = {
        "schema_version": "interlayer_grouped_profile_plot_atlas_manifest_v4",
        "campaign_id": "foundation_interlayer_sliding_10model_20261024",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "render_status": "RECONSTRUCTION",
        "layout": "2 systems × 2 methods; grouped model classes as series",
        "grouping": MODEL_GROUPS,
        "uncertainty": "pointwise sample SD across model identities within each class",
        "source_tables": [{"path": str(path), "sha256": sha256(path)} for path in source_paths],
        "outputs": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs],
        "visual_grammar": "plot_atlas.profile_ensemble / R04 strict grouped style / self-relaxed path label / PDF-referenced bold typography with strengthened DFT reference trace",
        "scientific_boundaries": {
            "fixed_path": "FINITE_DIAGNOSTIC; fixed-path single-point scoring",
            "self_relaxed": "CONVERGED endpoint-only MLIP NEB path quantity, not DFT MEP barrier",
            "qe_reference": "SCF_WARNING retained as screened reference context, not unconditional ground truth",
            "model_lineage": "MatPES-PBE OMAT-FT included in MACE-OMAT-0 grouping as MatPES-OMAT-FT",
        },
    }
    (root / "manifests/FIGURE_PROVENANCE_PLOT_ATLAS_GROUPED_R04.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
