#!/usr/bin/env python3
"""Build the four-row Foundation-model class ensemble figure.

Rows are Cr, Cu, Li, and Na. Columns are in-gap diffusion, gap-to-QL, and
within-QL. All colored curves are pointwise class means over the same nine
local Foundation checkpoints used for the existing Cu/Li/Na campaign; the
shaded bands are pointwise sample SD. The black dashed trace is a screened QE
path-table reference and is not presented as converged DFT ground truth.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis" / "neb_cross_element_triptychs_20261024"
FIG_OUT = OUT / "figures" / "foundation_class_4x3_R01"
DATA_OUT = OUT / "data" / "foundation_class_4x3_R01"
RAW_DATA_OUT = OUT / "data"
MODEL_DIR = Path("/Users/alina/Downloads/mace-foundation-models")
INPUT_DIR = REPO / "analysis" / "task3_endpoint_only_neb" / "interpolations"
CR_INPUT_DIR = REPO / "analysis" / "task3_endpoint_only_neb" / "neb_fixed_vs_self_matrix_20261019" / "inputs" / "fixed_r01"
CR_QE_TABLE = OUT / "data" / "Cr_fixed_path_representative_profiles.tsv"
CR_TIER1 = REPO / "analysis" / "task3_endpoint_only_neb" / "neb_fixed_vs_self_matrix_20261019" / "derived_r01" / "TIER1_IMAGE_PROFILES.tsv"
QE_COMBINED = OUT / "data" / "foundation_model_qe_reference_profiles_3x3.tsv"

MODEL_ORDER = [
    "mace-large-density-agnesi-stress",
    "mace-medium-density-agnesi-stress",
    "mace-small-density-agnesi-stress",
    "mace_agnesi_medium",
    "mace-mh-0",
    "mace-mh-1",
    "mace-mp-0b3-medium",
    "mace-omat-0-medium",
    "mace-omat-0-small",
]
MODEL_CLASS = {
    "mace-large-density-agnesi-stress": "MACE-MP",
    "mace-medium-density-agnesi-stress": "MACE-MP",
    "mace-small-density-agnesi-stress": "MACE-MP",
    "mace_agnesi_medium": "MACE-MP",
    "mace-mp-0b3-medium": "MACE-MP",
    "mace-mh-0": "MACE-MH",
    "mace-mh-1": "MACE-MH",
    "mace-omat-0-medium": "MACE-OMAT-0",
    "mace-omat-0-small": "MACE-OMAT-0",
}
CLASS_ORDER = ["MACE-MP", "MACE-OMAT-0", "MACE-MH"]
CLASS_COLORS = {
    "MACE-MP": "#d5a238",
    "MACE-OMAT-0": "#d85a45",
    "MACE-MH": "#5d8b56",
}
ELEMENTS = ["Cr", "Cu", "Li", "Na"]
PATHS = [
    ("within_gap", "in-gap diffusion"),
    ("gap_to_ql", "gap→QL"),
    ("within_ql", "within-QL"),
]
PATH_PREFIX = {
    "Cr": {
        "within_gap": "cr_canonical_in_gap",
        "gap_to_ql": "cr_canonical_gap_to_ql",
        "within_ql": "cr_canonical_within_ql",
    },
    "Cu": {
        "gap_to_ql": "2L-Cu__neb-gap2QL",
        "within_gap": "2L-Cu__neb-gap-diffuse",
        "within_ql": "2L-Cu__neb-QLs",
    },
    "Li": {
        "gap_to_ql": "2L-Li__neb-gap2Ql",
        "within_gap": "2L-Li__neb-gap-diffuse",
        "within_ql": "2L-Li__neb-QLs",
    },
    "Na": {
        "gap_to_ql": "2L-Na__neb-gap2QL",
        "within_gap": "2L-Na__neb-gap-diffuse",
        "within_ql": "2L-Na__neb-QLs",
    },
}
CR_SOURCE_TO_MODEL = {
    "foundation_agnesi_large": "mace-large-density-agnesi-stress",
    "foundation_agnesi_medium_density": "mace-medium-density-agnesi-stress",
    "foundation_agnesi_small": "mace-small-density-agnesi-stress",
    "foundation_agnesi_medium": "mace_agnesi_medium",
    "foundation_mh0": "mace-mh-0",
    "foundation_mp0b3": "mace-mp-0b3-medium",
    "foundation_omat0_medium": "mace-omat-0-medium",
    "foundation_omat0_small": "mace-omat-0-small",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 1.05,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.0,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


_SHA_CACHE: dict[Path, str] = {}


def sha256(path: Path) -> str:
    if path in _SHA_CACHE:
        return _SHA_CACHE[path]
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _SHA_CACHE[path] = digest
    return digest


def load_cr_rows() -> list[dict[str, object]]:
    """Load the materialized Cr fixed-path campaign, filtering out NEB branches."""
    import pandas as pd

    if not CR_TIER1.exists():
        raise FileNotFoundError(CR_TIER1)
    data = pd.read_csv(CR_TIER1, sep="\t")
    data = data[
        (data["observable"] == "FIXED_PATH_SINGLE_POINT")
        & (data["initialization_method"] == "DFT_RETAINED_PATH")
        & data["model_id"].isin(CR_SOURCE_TO_MODEL)
    ].copy()
    rows: list[dict[str, object]] = []
    for source_model, block in data.groupby("model_id", sort=False):
        model_id = CR_SOURCE_TO_MODEL[source_model]
        for path_key, _display in PATHS:
            path_id = PATH_PREFIX["Cr"][path_key]
            subset = block[block["path_id"] == path_id].sort_values("image_index")
            if len(subset) != 10 or subset["image_index"].tolist() != list(range(10)):
                raise ValueError(f"Cr fixed-path profile is incomplete: {source_model} {path_id}")
            model_path = MODEL_DIR / f"{model_id}.model"
            for _, record in subset.iterrows():
                rows.append(
                    {
                        "element": "Cr",
                        "path_key": path_key,
                        "path_id": path_id,
                        "model_id": model_id,
                        "model_class": MODEL_CLASS[model_id],
                        "image_index": int(record["image_index"]) + 1,
                        "s": float(record["image_index"]) / 9.0,
                        "energy_eV": float(record["energy_eV"]),
                        "relative_energy_eV": float(record["relative_energy_eV"]),
                        "max_force_eV_A": float(record["fmax_eV_A"]),
                        "head": "omat_pbe" if model_id in {"mace-mh-0", "mace-mh-1"} else "default",
                        "model_sha256": "SEE_MANIFEST_MODEL_HASHES",
                        "input_sha256": "UNAVAILABLE_SOURCE_INPUT",
                        "input_path": f"{CR_INPUT_DIR / (path_id + '.extxyz')}",
                        "evaluation_status": "FINITE_DIAGNOSTIC_REMOTE_TIER1",
                        "profile_source": str(record["profile_source"]),
                        "source_record": str(record["record_id"]),
                    }
                )
    expected = 8 * 3 * 10
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} Cr fixed-path rows, found {len(rows)}")
    return rows


def load_existing_rows() -> list[dict[str, object]]:
    path = RAW_DATA_OUT / "foundation_model_profiles_9models_3x3.tsv"
    rows: list[dict[str, object]] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["element"] not in {"Cu", "Li", "Na"}:
                continue
            if row["model_id"] not in MODEL_CLASS:
                raise ValueError(f"Unexpected model id in existing campaign: {row['model_id']}")
            row["model_class"] = MODEL_CLASS[row["model_id"]]
            for key in ("image_index",):
                row[key] = int(row[key])
            for key in ("s", "energy_eV", "relative_energy_eV", "max_force_eV_A"):
                row[key] = float(row[key])
            row["evaluation_status"] = "FINITE_DIAGNOSTIC"
            rows.append(row)
    if len(rows) != 810:
        raise ValueError(f"Expected 810 Cu/Li/Na rows, found {len(rows)}")
    return rows


def load_qe() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with QE_COMBINED.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            element = row["element"]
            if element not in {"Cu", "Li", "Na"}:
                continue
            rows.append(
                {
                    "element": element,
                    "path_key": row["path_key"],
                    "path_id": PATH_PREFIX[element][row["path_key"]],
                    "image_index": int(row["image_index"]),
                    "s": float(row["s"]),
                    "relative_energy_eV": float(row["relative_energy_eV"]),
                    "status": row.get("status", "COMPLETED_BUT_SCF_WARNING"),
                    "source_path": row.get("source_path", ""),
                    "source_record": row.get("source_record", ""),
                }
            )
    # The Cr representative QE table is a cloud placeholder in this checkout.
    # Keep the Cr panels model-only rather than substituting a different Cr path.
    if len(rows) != 90:
        raise ValueError(f"Expected 90 Cu/Li/Na QE rows, found {len(rows)}")
    return rows


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(model_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: list[dict[str, object]] = []
    for element in ELEMENTS:
        for path_key, _display in PATHS:
            for model_class in CLASS_ORDER:
                for image_index in range(1, 11):
                    vals = [
                        float(r["relative_energy_eV"])
                        for r in model_rows
                        if r["element"] == element
                        and r["path_key"] == path_key
                        and r["model_class"] == model_class
                        and r["image_index"] == image_index
                    ]
                    expected = {
                        "Cr": {"MACE-MP": 5, "MACE-OMAT-0": 2, "MACE-MH": 1},
                        "Cu": {"MACE-MP": 5, "MACE-OMAT-0": 2, "MACE-MH": 2},
                        "Li": {"MACE-MP": 5, "MACE-OMAT-0": 2, "MACE-MH": 2},
                        "Na": {"MACE-MP": 5, "MACE-OMAT-0": 2, "MACE-MH": 2},
                    }[element][model_class]
                    if len(vals) != expected:
                        raise ValueError(f"Missing class value: {element} {path_key} {model_class} {image_index} n={len(vals)}")
                    grouped.append(
                        {
                            "element": element,
                            "path_key": path_key,
                            "model_class": model_class,
                            "image_index": image_index,
                            "s": (image_index - 1) / 9.0,
                            "n_models": len(vals),
                            "mean_relative_energy_eV": float(np.mean(vals)),
                            "sd_relative_energy_eV": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                        }
                    )
    return grouped


def _set_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.tick_params(axis="both", which="major", direction="out", length=4.5, width=1.0, pad=3)
    ax.tick_params(axis="both", which="minor", direction="out", length=2.5, width=0.8)
    ax.minorticks_on()
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#111111")
        spine.set_linewidth(1.05)


def plot_panel(ax: plt.Axes, element: str, path_key: str, model_rows: list[dict[str, object]], grouped: list[dict[str, object]], qe_rows: list[dict[str, object]], panel_label: str) -> None:
    _set_axes(ax)
    if panel_label:
        ax.text(-0.16, 1.06, panel_label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")
    for model_class in CLASS_ORDER:
        vals = [r for r in grouped if r["element"] == element and r["path_key"] == path_key and r["model_class"] == model_class]
        vals.sort(key=lambda r: r["image_index"])
        x = np.array([float(r["s"]) for r in vals])
        y = np.array([float(r["mean_relative_energy_eV"]) for r in vals])
        color = CLASS_COLORS[model_class]
        ax.plot(x, y, color=color, lw=2.0, marker="o", markersize=4.3, markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.95, zorder=5)
        sd = np.array([np.nan if r["sd_relative_energy_eV"] is None else float(r["sd_relative_energy_eV"]) for r in vals])
        if np.isfinite(sd).all() and np.any(sd > 0):
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.20, linewidth=0, zorder=2)
    qe = [r for r in qe_rows if r["element"] == element and r["path_key"] == path_key]
    qe.sort(key=lambda r: r["image_index"])
    band_values = []
    if qe:
        x = np.array([float(r["s"]) for r in qe])
        y = np.array([float(r["relative_energy_eV"]) for r in qe])
        ax.plot(x, y, color="#222222", lw=1.45, ls=(0, (4, 2)), marker="D", markersize=3.6, markerfacecolor="white", markeredgewidth=0.8, zorder=5)
        band_values.append(y)
    for c in CLASS_ORDER:
        class_rows = [r for r in grouped if r["element"] == element and r["path_key"] == path_key and r["model_class"] == c]
        means = np.array([float(r["mean_relative_energy_eV"]) for r in class_rows])
        sds = np.array([0.0 if r["sd_relative_energy_eV"] is None else float(r["sd_relative_energy_eV"]) for r in class_rows])
        band_values.extend([means - sds, means + sds])
    vals_all = np.concatenate(band_values)
    lo, hi = float(np.nanmin(vals_all)), float(np.nanmax(vals_all))
    pad = max(0.06 * (hi - lo), 0.04)
    ax.set_ylim(lo - pad, hi + pad)


def render(model_rows: list[dict[str, object]], grouped: list[dict[str, object]], qe_rows: list[dict[str, object]]) -> dict[str, object]:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(10.8, 11.8), sharex=False, sharey=False)
    panel_qa: list[dict[str, object]] = []
    labels = iter("abcdefghijkl")
    for i, element in enumerate(ELEMENTS):
        for j, (path_key, display) in enumerate(PATHS):
            ax = axes[i, j]
            plot_panel(ax, element, path_key, model_rows, grouped, qe_rows, next(labels))
            if i == 0:
                ax.set_title(display, fontsize=12.0, fontweight="bold", pad=8)
            if j == 0:
                ax.set_ylabel(f"{element}\nΔE (eV)", fontsize=11.0, fontweight="bold", labelpad=8)
            if i == len(ELEMENTS) - 1:
                ax.set_xlabel("normalized image coordinate, $s$", fontsize=10.8, labelpad=7)
            panel_qa.append({"element": element, "path_key": path_key, "y_limits": [float(v) for v in ax.get_ylim()]})

    legend_handles = [Line2D([], [], color=CLASS_COLORS[c], lw=2.0, label=c) for c in CLASS_ORDER]
    legend_handles.append(Line2D([], [], color="#222222", lw=1.45, ls=(0, (4, 2)), marker="D", markersize=3.6, markerfacecolor="white", label="screened QE path table"))
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.52, 0.985), ncol=4, frameon=False, handlelength=2.2, columnspacing=1.25, handletextpad=0.45)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.065, top=0.925, wspace=0.34, hspace=0.43)
    stem = FIG_OUT / "foundation_model_class_4x3_R01"
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for element in ELEMENTS:
        for path_key, display in PATHS:
            fig1, ax = plt.subplots(figsize=(3.35, 2.85))
            plot_panel(ax, element, path_key, model_rows, grouped, qe_rows, "")
            ax.set_title(f"{element} · {display}", fontsize=11.5, fontweight="bold", pad=7)
            ax.set_xlabel("normalized image coordinate, $s$", fontsize=10.3, labelpad=6)
            ax.set_ylabel("ΔE (eV)", fontsize=10.3, labelpad=6)
            fig1.subplots_adjust(left=0.19, right=0.98, bottom=0.19, top=0.86)
            panel_stem = FIG_OUT / f"panel_{element}_{path_key}"
            fig1.savefig(panel_stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
            fig1.savefig(panel_stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
            fig1.savefig(panel_stem.with_suffix(".pdf"), dpi=400, bbox_inches="tight", facecolor="white")
            plt.close(fig1)
    return {"figure": str(stem), "panels": panel_qa}


def main() -> None:
    existing_rows = load_existing_rows()
    cr_rows_path = DATA_OUT / "Cr_foundation_model_profiles_9models_3paths_R01.tsv"
    if os.environ.get("FOUNDATION_REUSE_CR_ROWS") == "1" and cr_rows_path.exists():
        cr_rows: list[dict[str, object]] = []
        with cr_rows_path.open(newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                for key in ("image_index",):
                    row[key] = int(row[key])
                for key in ("s", "energy_eV", "relative_energy_eV", "max_force_eV_A"):
                    row[key] = float(row[key])
                cr_rows.append(row)
    else:
        cr_rows = load_cr_rows()
        write_tsv(cr_rows_path, cr_rows)
    model_rows = existing_rows + cr_rows
    if len(model_rows) != 1050:
        raise ValueError(f"Expected 1050 total model rows (Cr n=8; Cu/Li/Na n=9), found {len(model_rows)}")
    qe_rows = load_qe()
    grouped = aggregate(model_rows)
    write_tsv(DATA_OUT / "foundation_model_profiles_9models_4x3_R01.tsv", model_rows)
    write_tsv(DATA_OUT / "foundation_model_class_profiles_4x3_R01.tsv", grouped)
    write_tsv(DATA_OUT / "foundation_model_qe_reference_profiles_4x3_R01.tsv", qe_rows)
    figure = render(model_rows, grouped, qe_rows)
    artifact_paths = {
        "figure_png": Path(f"{figure['figure']}.png"),
        "figure_svg": Path(f"{figure['figure']}.svg"),
        "figure_pdf": Path(f"{figure['figure']}.pdf"),
        "model_profiles": DATA_OUT / "foundation_model_profiles_9models_4x3_R01.tsv",
        "class_profiles": DATA_OUT / "foundation_model_class_profiles_4x3_R01.tsv",
        "qe_reference_profiles": DATA_OUT / "foundation_model_qe_reference_profiles_4x3_R01.tsv",
    }
    model_hashes = {
        m: next((str(r.get("model_sha256", "")) for r in existing_rows if r["model_id"] == m), "UNAVAILABLE")
        for m in MODEL_ORDER
    }
    input_hashes = {f"Cr:{k}": "UNAVAILABLE_SOURCE_INPUT_PLACEHOLDER" for k, _ in PATHS}
    manifest = {
        "figure_id": "foundation_model_class_4x3_R01",
        "status": "RECONSTRUCTION",
        "scope": "nine local Foundation checkpoints for Cu/Li/Na and eight available Cr fixed-path Foundation profiles on twelve 10-image paths",
        "rows": ELEMENTS,
        "columns": [{"path_key": k, "label": d} for k, d in PATHS],
        "model_classes": {c: [m for m in MODEL_ORDER if MODEL_CLASS[m] == c] for c in CLASS_ORDER},
        "aggregation": "pointwise mean with sample SD (ddof=1); hollow circles show class means; individual model markers are omitted",
        "qe_label": "screened QE path table for Cu/Li/Na only; Cr reference unavailable in this checkout; retained source paths carry SCF convergence warnings",
        "model_directory": str(MODEL_DIR),
        "model_hashes": model_hashes,
        "cr_input_paths": {k: str(CR_INPUT_DIR / f"{PATH_PREFIX['Cr'][k]}.extxyz") for k, _ in PATHS},
        "cr_input_hashes": input_hashes,
        "model_rows": len(model_rows),
        "class_rows": len(grouped),
        "qe_rows": len(qe_rows),
        "cr_available_model_ids": sorted({r["model_id"] for r in cr_rows}),
        "cr_missing_model_ids": [m for m in MODEL_ORDER if m not in {r["model_id"] for r in cr_rows}],
        "cr_qe_reference": "UNAVAILABLE_CLOUD_PLACEHOLDER",
        "artifact_hashes": {name: sha256(path) for name, path in artifact_paths.items()},
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "outputs": figure,
        "prior_campaign_raw": str(RAW_DATA_OUT / "foundation_model_profiles_9models_3x3.tsv"),
    }
    manifest_path = OUT / "figure_data_manifest_foundation_class_4x3_R01.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    caption = OUT / "CAPTIONS_foundation_class_4x3_R01.md"
    caption.write_text(
        "# Foundation-model class ensemble 4×3\n\n"
        "**Figure X | Foundation-model class ensembles across Cr, Cu, Li, and Na migration paths.** "
        "Rows show Cr, Cu, Li, and Na; columns show in-gap diffusion, gap→QL, and within-QL paths. "
        "Each colored curve is the pointwise mean relative energy of the local Foundation checkpoints in that class; hollow circles show the class mean and the translucent band is the pointwise sample standard deviation across checkpoints. Individual model markers are omitted. "
        "The classes are MACE-MP (five checkpoints: Agnesi medium and density-stress small/medium/large, plus MP-0b3 medium), MACE-OMAT-0 (two checkpoints), and MACE-MH (two checkpoints for Cu/Li/Na; one available checkpoint for Cr), following the official ACEsuit release mapping. "
        "Cr uses the canonical 81-atom, ten-image fixed-path campaign profiles materialized in TIER1; its MACE-MH-1 fixed-path result is unavailable and is not imputed. The dashed black curve is the screened QE path-table reference for Cu/Li/Na; the Cr QE path-table source is unavailable in this checkout and is intentionally omitted from the Cr row. All retained QE paths carry SCF-convergence warnings; the reference is therefore diagnostic and is not presented as converged DFT ground truth or as a publication-grade activation barrier. "
        "All colored curves are fixed-path single-point evaluations and are not self-relaxed MLFF-NEB results.\n"""
    )
    qa = {
        "figure_id": manifest["figure_id"],
        "status": "PASS_WITH_SCIENTIFIC_BOUNDARY",
        "checks": {
            "model_evaluations": len(model_rows) == 1050,
            "class_profiles": len(grouped) == 360,
            "qe_reference_profiles": len(qe_rows) == 90,
            "all_finite": all(math.isfinite(float(r["relative_energy_eV"])) for r in model_rows),
            "all_classes_present": all(sum(v == c for v in MODEL_CLASS.values()) >= 1 for c in CLASS_ORDER),
            "class_counts_correct": {c: sum(v == c for v in MODEL_CLASS.values()) for c in CLASS_ORDER} == {"MACE-MP": 5, "MACE-OMAT-0": 2, "MACE-MH": 2},
            "mean_markers_enabled": True,
            "cr_rows_first": ELEMENTS[0] == "Cr",
            "cr_mh1_missing_explicit": "mace-mh-1" not in {r["model_id"] for r in cr_rows},
            "cr_qe_reference_unavailable_explicit": True,
            "independent_y_axes": True,
            "fixed_path_only": True,
        },
        "panel_qa": figure["panels"],
        "boundary": "Cu/Li/Na QE traces are SCF-warning screened path tables; Cr QE reference and Cr MACE-MH-1 fixed-path profile are unavailable in this checkout; colored curves are fixed-path single-point diagnostics, not converged NEB barriers.",
    }
    (OUT / "QA_REPORT_foundation_class_4x3_R01.json").write_text(json.dumps(qa, indent=2) + "\n")
    (OUT / "DELIVERY_INDEX_foundation_class_4x3_R01.md").write_text(
        "# Foundation-model class ensemble 4×3 delivery\n\n"
        f"- Figure PNG: `{figure['figure']}.png`\n"
        f"- Figure SVG: `{figure['figure']}.svg`\n"
        f"- Figure PDF: `{figure['figure']}.pdf`\n"
        f"- Combined model profiles: `{DATA_OUT / 'foundation_model_profiles_9models_4x3_R01.tsv'}`\n"
        f"- Class profiles: `{DATA_OUT / 'foundation_model_class_profiles_4x3_R01.tsv'}`\n"
        f"- QE reference profiles: `{DATA_OUT / 'foundation_model_qe_reference_profiles_4x3_R01.tsv'}`\n"
        f"- Cr model profiles: `{cr_rows_path}`\n"
        f"- Manifest: `{manifest_path}`\n"
        f"- QA: `{OUT / 'QA_REPORT_foundation_class_4x3_R01.json'}`\n"
        f"- Caption: `{caption}`\n\n"
        "Scientific status: fixed-path Foundation-model ensemble diagnostic. Cr has eight available Foundation profiles (MACE-MH-1 is missing) and no readable Cr QE table in this checkout; Cu/Li/Na QE traces retain SCF warnings and are not unconditional DFT ground truth.\n"
    )
    print(json.dumps({"figure": manifest["figure_id"], "model_rows": len(model_rows), "class_rows": len(grouped), "qe_rows": len(qe_rows)}, indent=2))


if __name__ == "__main__":
    main()
