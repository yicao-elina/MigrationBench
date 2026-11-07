#!/usr/bin/env python3
"""Re-render the fixed-path campaign figures with Plot Atlas profile grammar.

This is additive: the original figure products are not overwritten. The
scientific inputs remain the frozen RESULT_MATRIX, IMAGE_PROFILES, and
QE_REFERENCE tables. Fixed-path results retain their FINITE_DIAGNOSTIC status;
the QE trace is labelled as an SCF-warning reference, and self-relaxed values
are intentionally not mixed into these panels.
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

MODEL_ORDER = [
    "foundation_agnesi_large",
    "foundation_agnesi_medium_density",
    "foundation_agnesi_small_density",
    "foundation_agnesi_medium",
    "foundation_mp0b3",
    "foundation_omat0_medium",
    "foundation_omat0_small",
    "foundation_mh0",
    "foundation_mh1",
    "foundation_matpes_pbe_omat_ft",
]

# Family-grouped shades preserve the campaign's established MACE family colors
# while making all ten model traces legible in one panel.
MODEL_COLORS = {
    "foundation_agnesi_large": "#A87C18",
    "foundation_agnesi_medium_density": "#C18F22",
    "foundation_agnesi_small_density": "#D5A238",
    "foundation_agnesi_medium": "#E2BD68",
    "foundation_mp0b3": "#8F2F23",
    "foundation_omat0_medium": "#B94738",
    "foundation_omat0_small": "#D85A45",
    "foundation_mh0": "#3F6B45",
    "foundation_mh1": "#5D8B56",
    "foundation_matpes_pbe_omat_ft": "#91AF8E",
}

MODEL_LABELS = {
    "foundation_agnesi_large": "Agnesi-L",
    "foundation_agnesi_medium_density": "Agnesi-MD",
    "foundation_agnesi_small_density": "Agnesi-SD",
    "foundation_agnesi_medium": "Agnesi-M",
    "foundation_mp0b3": "MP-0b3",
    "foundation_omat0_medium": "OMAT-M",
    "foundation_omat0_small": "OMAT-S",
    "foundation_mh0": "MH-0",
    "foundation_mh1": "MH-1",
    "foundation_matpes_pbe_omat_ft": "MatPES-OMAT-FT",
}


def plot_profile_ensemble(ax, profiles, *, x, reference, series_order, palette, title, xlabel, ylabel, panel_label=None, legend=False):
    """Campaign-local adapter implementing the registered Plot Atlas grammar.

    The reusable primitive is the source of this contract; this small adapter
    keeps the campaign render independent of the atlas package's unrelated
    chart imports while preserving the same artist defaults.
    """
    cfg = {
        "font": "Arial", "tick": 12.0, "xlabel": 12.8, "ylabel": 13.0,
        "title": 14.0, "panel": 15.0, "line": 2.0, "marker": 4.3,
        "edge": 0.95, "spine": 1.05, "reference_line": 1.45,
        "reference_marker": 3.6,
    }
    colors = palette["categorical"]
    lines = {}
    for index, label in enumerate(series_order):
        record = profiles[label]
        (line,) = ax.plot(
            x, record["mean"], color=colors[index], lw=cfg["line"], marker="o",
            markersize=cfg["marker"], markerfacecolor="white",
            markeredgecolor=colors[index], markeredgewidth=cfg["edge"],
            label=label, zorder=5,
        )
        lines[label] = line
    (reference_line,) = ax.plot(
        reference["x"], reference["y"], color="#222222", lw=cfg["reference_line"],
        ls=(0, (4, 2)), marker="D", markersize=cfg["reference_marker"],
        markerfacecolor="white", markeredgecolor="#222222", markeredgewidth=0.8,
        label=reference["label"], zorder=5,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0))
    ax.set_xticklabels(("0", "0.25", "0.5", "0.75", "1"))
    ax.tick_params(axis="both", which="major", direction="out", length=4.5, width=1.0, pad=3, labelsize=cfg["tick"])
    ax.tick_params(axis="both", which="minor", direction="out", length=2.5, width=0.8)
    ax.minorticks_on()
    ax.grid(False)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontfamily(cfg["font"])
        label.set_fontsize(cfg["tick"])
        label.set_fontweight("normal")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#111111")
        spine.set_linewidth(cfg["spine"])
    ax.set_title(title, fontsize=cfg["title"], fontweight="bold", pad=8, family=cfg["font"])
    ax.set_xlabel(xlabel, fontsize=cfg["xlabel"], labelpad=7, family=cfg["font"])
    ax.set_ylabel(ylabel, fontsize=cfg["ylabel"], fontweight="bold", labelpad=8, family=cfg["font"])
    if panel_label is not None:
        ax.text(-0.16, 1.06, panel_label, transform=ax.transAxes, fontsize=cfg["panel"], family=cfg["font"], fontweight="bold", va="bottom")
    if legend:
        artist = ax.legend(handles=[*lines.values(), reference_line], frameon=False, fontsize=12.0, ncol=2, handlelength=2.2, columnspacing=1.25, handletextpad=0.45, loc="upper center", bbox_to_anchor=(0.5, -0.23))
        for text in artist.get_texts():
            text.set_fontfamily(cfg["font"])
    return {"series_lines": lines, "reference_line": reference_line}


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
    fixed = [row for row in matrix if row["observable"] == "FIXED_PATH_SINGLE_POINT"]
    if len(matrix) != 40 or len(fixed) != 20:
        raise ValueError(f"expected 40 total and 20 fixed rows, found {len(matrix)} and {len(fixed)}")
    if {row["model_id"] for row in fixed} != set(MODEL_ORDER):
        raise ValueError("fixed-path model identity gate failed")
    if {row["system"] for row in fixed} != {"bulk", "cr_in_gap"}:
        raise ValueError("fixed-path system identity gate failed")
    if any(row["scientific_status"] != "FINITE_DIAGNOSTIC" for row in fixed):
        raise ValueError("fixed-path scientific status gate failed")
    if len(qe) != 20 or {row["system"] for row in qe} != {"bulk", "cr_in_gap"}:
        raise ValueError("QE reference gate failed")
    for row in fixed:
        path = [item for item in profiles if item["record_id"] == row["record_id"]]
        if len(path) != 10 or [int(item["image_index"]) for item in path] != list(range(10)):
            raise ValueError(f"profile completeness/order failed for {row['record_id']}")
    return fixed, profiles, qe


def palette() -> dict[str, object]:
    return {"categorical": [MODEL_COLORS[model_id] for model_id in MODEL_ORDER], "reference": "#222222"}


def panel_data(system: str, fixed: list[dict[str, str]], profiles: list[dict[str, str]], qe: list[dict[str, str]]) -> tuple[dict, dict, list[str]]:
    rows_by_record = {row["record_id"]: row for row in fixed if row["system"] == system}
    profile_map: dict[str, dict[str, list[float]]] = {}
    labels: list[str] = []
    for model_id in MODEL_ORDER:
        record = rows_by_record[model_id if model_id in rows_by_record else next(k for k, v in rows_by_record.items() if v["model_id"] == model_id)]
        model_rows = sorted(
            (row for row in profiles if row["record_id"] == record["record_id"]),
            key=lambda row: int(row["image_index"]),
        )
        label = MODEL_LABELS[model_id]
        labels.append(label)
        profile_map[label] = {
            "x": [float(row["image_fraction"]) for row in model_rows],
            "mean": [float(row["relative_energy_eV"]) for row in model_rows],
        }
    reference_rows = sorted((row for row in qe if row["system"] == system), key=lambda row: int(row["image_index"]))
    reference = {
        "x": [float(row["image_fraction"]) for row in reference_rows],
        "y": [float(row["relative_energy_eV"]) for row in reference_rows],
        "label": "QE reference (SCF-warning)",
    }
    return profile_map, reference, labels


def draw(ax, system: str, fixed: list[dict[str, str]], profiles: list[dict[str, str]], qe: list[dict[str, str]], panel_label: str | None = None, legend: bool = False) -> dict:
    profile_map, reference, labels = panel_data(system, fixed, profiles, qe)
    result = plot_profile_ensemble(
        ax,
        profile_map,
        x=profile_map[labels[0]]["x"],
        reference=reference,
        series_order=labels,
        palette=palette(),
        title="Bulk" if system == "bulk" else "Cr-in-gap",
        xlabel="normalized image coordinate, $s$",
        ylabel="$\\Delta E$ from image 0 (eV)",
        panel_label=panel_label,
        legend=legend,
    )
    return result


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(ext) for ext in (".png", ".svg", ".pdf")]
    for path in paths:
        fig.savefig(path, dpi=400 if path.suffix == ".png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def render(root: Path) -> dict[str, object]:
    fixed, profiles, qe = validate(root)
    out = root / "results/figures/plot_atlas_r01"
    out.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for system in ("bulk", "cr_in_gap"):
        fig, ax = plt.subplots(figsize=(8.4, 6.8))
        draw(ax, system, fixed, profiles, qe, legend=True)
        fig.subplots_adjust(left=0.15, right=0.98, top=0.90, bottom=0.29)
        outputs.extend(save_figure(fig, out / f"fixed_path_{system}_10model_plot_atlas_r01"))

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.0), sharey=False)
    handles = None
    labels = None
    for index, system in enumerate(("bulk", "cr_in_gap")):
        result = draw(axes[index], system, fixed, profiles, qe, panel_label=chr(97 + index), legend=False)
        if handles is None:
            handles = list(result["series_lines"].values()) + [result["reference_line"]]
            labels = [handle.get_label() for handle in handles]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.52, 0.025), ncol=5, frameon=False, fontsize=12.0, handlelength=2.2, columnspacing=1.25, handletextpad=0.45)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.93, bottom=0.22, wspace=0.30)
    outputs.extend(save_figure(fig, out / "fixed_path_interlayer_sliding_10model_2system_plot_atlas_r01"))

    source_paths = [root / "results/tables/RESULT_MATRIX.tsv", root / "results/tables/IMAGE_PROFILES.tsv", root / "results/tables/QE_REFERENCE.tsv"]
    manifest = {
        "schema_version": "fixed_path_10model_plot_atlas_manifest_v1",
        "campaign_id": "foundation_interlayer_sliding_10model_20261024",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "render_status": "RECONSTRUCTION",
        "observable": "FIXED_PATH_SINGLE_POINT",
        "source_tables": [{"path": str(path), "sha256": sha256(path)} for path in source_paths],
        "fixed_rows": 20,
        "profile_rows": 200,
        "qe_reference_rows": 20,
        "models": MODEL_ORDER,
        "systems": ["bulk", "cr_in_gap"],
        "palette": MODEL_COLORS,
        "visual_grammar": "plot_atlas.profile_ensemble",
        "outputs": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs],
        "scientific_boundaries": {
            "fixed_path": "FINITE_DIAGNOSTIC; fixed-path single-point scoring, not a self-relaxed MEP",
            "qe_reference": "SCF_WARNING retained as diagnostic/reference context, not unconditional ground truth",
            "self_relaxed": "not mixed into these fixed-path panels; endpoint-only MLIP NEB path quantity remains separate",
            "model_lineage": "MatPES-PBE OMAT-FT represented as MatPES-OMAT-FT and retained as its verified model identity",
        },
    }
    (root / "manifests").mkdir(exist_ok=True)
    (root / "manifests/FIGURE_PROVENANCE_PLOT_ATLAS_R01.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
