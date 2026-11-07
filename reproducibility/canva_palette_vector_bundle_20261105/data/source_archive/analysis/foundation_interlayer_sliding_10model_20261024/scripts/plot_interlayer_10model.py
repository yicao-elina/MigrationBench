#!/usr/bin/env python3
"""Render fixed-path 10-model interlayer-sliding panels and provenance."""

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


MODELS = [
    "foundation_agnesi_large", "foundation_agnesi_medium_density", "foundation_agnesi_small_density",
    "foundation_agnesi_medium", "foundation_mp0b3", "foundation_omat0_medium", "foundation_omat0_small",
    "foundation_mh0", "foundation_mh1", "foundation_matpes_pbe_omat_ft",
]
COLORS = {"MACE-MP": "#d5a238", "MACE-OMAT-0": "#d85a45", "MACE-MH": "#5d8b56"}
MARKERS = ["o", "s", "^", "v", "P", "D", "X", "<", ">", "*" ]
LINESTYLES = ["-", "--", ":", "-.", "-", "--", ":", "-.", "-", "--"]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(root: Path) -> tuple[list[dict], list[dict]]:
    matrix = read_tsv(root / "results/tables/RESULT_MATRIX.tsv")
    profiles = read_tsv(root / "results/tables/IMAGE_PROFILES.tsv")
    fixed = [row for row in matrix if row["observable"] == "FIXED_PATH_SINGLE_POINT"]
    if len(matrix) != 40 or len(fixed) != 20:
        raise ValueError(f"plot gate requires 40 total and 20 fixed rows, found {len(matrix)} and {len(fixed)}")
    if {row["model_id"] for row in fixed} != set(MODELS) or {row["system"] for row in fixed} != {"bulk", "cr_in_gap"}:
        raise ValueError("fixed matrix model/system identity gate failed")
    usable = [row for row in fixed if row["scientific_status"] == "FINITE_DIAGNOSTIC" and row["publication_status"] == "PRIMARY_EVIDENCE"]
    if len(usable) != 20:
        raise ValueError(f"fixed scientific gate failed: {len(usable)}/20 usable")
    selected = []
    for row in usable:
        path = [item for item in profiles if item["record_id"] == row["record_id"]]
        if len(path) != 10:
            raise ValueError(f"{row['record_id']} has {len(path)} profile rows")
        if [int(item["image_index"]) for item in path] != list(range(10)):
            raise ValueError(f"{row['record_id']} image order failed")
        if abs(float(path[0]["relative_energy_eV"])) > 1e-12:
            raise ValueError(f"{row['record_id']} image-0 normalization failed")
        selected.extend(path)
    return usable, selected


def model_meta(root: Path) -> dict[str, dict[str, str]]:
    return {row["model_id"]: row for row in read_tsv(root / "MODEL_REGISTRY.tsv")}


def draw_panel(ax, system: str, rows: list[dict], qe: list[dict], meta: dict[str, dict[str, str]], legend: bool = True) -> None:
    for model_index, model_id in enumerate(MODELS):
        rowset = sorted((row for row in rows if row["model_id"] == model_id and row["system"] == system), key=lambda x: int(x["image_index"]))
        if len(rowset) != 10:
            raise ValueError(f"missing fixed profile for {model_id}/{system}")
        m = meta[model_id]
        label = m["plot_label"]
        if model_id == "foundation_matpes_pbe_omat_ft":
            label += " (FT)"
        ax.plot(
            [float(x["image_fraction"]) for x in rowset],
            [float(x["relative_energy_eV"]) for x in rowset],
            color=COLORS[m["model_family"]], marker=MARKERS[model_index], linestyle=LINESTYLES[model_index],
            linewidth=1.7, markersize=5.3, markerfacecolor="white", markeredgewidth=1.1,
            label=label, zorder=3,
        )
    reference = sorted((row for row in qe if row["system"] == system), key=lambda x: int(x["image_index"]))
    if len(reference) != 10:
        raise ValueError(f"missing QE reference for {system}")
    ax.plot(
        [float(x["image_fraction"]) for x in reference], [float(x["relative_energy_eV"]) for x in reference],
        color="black", linestyle="--", marker="D", markerfacecolor="white", markeredgecolor="black",
        linewidth=1.3, markersize=4.8, label="QE b-path2 retained reference (SCF-warning)", zorder=2,
    )
    ax.axhline(0.0, color="0.35", linewidth=0.8, linestyle=":", zorder=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("normalized image coordinate (0 → 1)", fontsize=11.5, fontfamily="Arial")
    ax.set_ylabel("ΔE from image 0 (eV)", fontsize=11.5, fontfamily="Arial")
    ax.set_title("Bulk" if system == "bulk" else "Cr-in-gap", fontsize=12.5, fontfamily="Arial")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="both", which="major", labelsize=10.5, width=1.2, length=4)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
    ax.grid(False)
    if legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=8.7, handlelength=2.8, columnspacing=1.1)


def save_panel(root: Path, system: str, rows: list[dict], qe: list[dict], meta: dict[str, dict[str, str]]) -> list[Path]:
    out = root / "results/figures"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    draw_panel(ax, system, rows, qe, meta, legend=True)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.34)
    stem = out / f"fixed_path_{system}_10model"
    paths = [stem.with_suffix(ext) for ext in (".pdf", ".svg", ".png")]
    for path in paths:
        fig.savefig(path, dpi=300 if path.suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)
    return paths


def save_composite(root: Path, rows: list[dict], qe: list[dict], meta: dict[str, dict[str, str]]) -> list[Path]:
    out = root / "results/figures"
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.1), sharey=False)
    draw_panel(axes[0], "bulk", rows, qe, meta, legend=False)
    draw_panel(axes[1], "cr_in_gap", rows, qe, meta, legend=False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.00), ncol=3, frameon=False, fontsize=8.8, handlelength=2.8)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.92, bottom=0.24, wspace=0.23)
    for index, ax in enumerate(axes):
        ax.text(-0.08, 1.04, "a" if index == 0 else "b", transform=ax.transAxes, fontsize=14, fontweight="bold", fontfamily="Arial")
    stem = out / "fixed_path_interlayer_sliding_10model_2system"
    paths = [stem.with_suffix(ext) for ext in (".pdf", ".svg", ".png")]
    for path in paths:
        fig.savefig(path, dpi=300 if path.suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    fixed, profiles = validate(root)
    qe = read_tsv(root / "results/tables/QE_REFERENCE.tsv")
    meta = model_meta(root)
    outputs = save_panel(root, "bulk", profiles, qe, meta) + save_panel(root, "cr_in_gap", profiles, qe, meta) + save_composite(root, profiles, qe, meta)
    manifest = {
        "schema_version": "fixed_path_10model_figure_manifest_v1", "campaign_id": "foundation_interlayer_sliding_10model_20261024",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS", "observable": "FIXED_PATH_SINGLE_POINT",
        "source_matrix": str(root / "results/tables/RESULT_MATRIX.tsv"), "source_profiles": str(root / "results/tables/IMAGE_PROFILES.tsv"),
        "source_qe_reference": str(root / "results/tables/QE_REFERENCE.tsv"), "source_matrix_sha256": sha256(root / "results/tables/RESULT_MATRIX.tsv"),
        "source_profiles_sha256": sha256(root / "results/tables/IMAGE_PROFILES.tsv"), "source_qe_reference_sha256": sha256(root / "results/tables/QE_REFERENCE.tsv"),
        "fixed_rows": len(fixed), "fixed_profile_rows": len(profiles), "systems": ["bulk", "cr_in_gap"],
        "palette": COLORS, "outputs": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs],
        "visual_grammar": {"axes": "boxed", "grid": False, "normalization": "image_0", "qe_style": "black dashed open diamond", "font": "Arial first choice"},
        "interpretation_boundary": "QE reference is COMPLETED_BUT_SCF_WARNING context; self-relaxed is not plotted here and remains a separate observable.",
    }
    manifest_path = root / "manifests/FIGURE_PROVENANCE.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
