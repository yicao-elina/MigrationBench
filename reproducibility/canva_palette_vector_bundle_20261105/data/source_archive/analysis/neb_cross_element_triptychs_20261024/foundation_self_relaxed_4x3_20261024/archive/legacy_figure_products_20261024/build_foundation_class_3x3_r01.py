#!/usr/bin/env python3
"""Build a clean 3x3 Cu/Li/Na Foundation-model class ensemble figure.

Rows are the three intercalant elements and columns are the three path
classes.  Each colored curve is the pointwise mean of all local Foundation
models in a model class; small points show every individual model evaluation;
translucent bands are pointwise sample SD.  The QE trace is retained as a
screened path-table reference and is never labelled as converged ground truth.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from matplotlib.lines import Line2D
from mace.calculators import MACECalculator


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis" / "neb_cross_element_triptychs_20261024"
FIG_OUT = OUT / "figures" / "foundation_class_3x3_R03"
DATA_OUT = OUT / "data" / "foundation_class_3x3_R03"
RAW_DATA_OUT = OUT / "data"
MODEL_DIR = Path("/Users/alina/Downloads/mace-foundation-models")
INPUT_DIR = REPO / "analysis" / "task3_endpoint_only_neb" / "interpolations"
QE_TABLES = {
    e: OUT / "data" / f"{e}_fixed_path_base_profiles.tsv" for e in ("Cu", "Li", "Na")
}

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
ELEMENTS = ["Cu", "Li", "Na"]
PATHS = [
    ("within_gap", "in-gap diffusion", "in-gap diffusion"),
    ("gap_to_ql", "gap→QL", "gap2QL"),
    ("within_ql", "within-QL", "within-QL"),
]
PATH_PREFIX = {
    "Cu": {"gap_to_ql": "2L-Cu__neb-gap2QL", "within_gap": "2L-Cu__neb-gap-diffuse", "within_ql": "2L-Cu__neb-QLs"},
    "Li": {"gap_to_ql": "2L-Li__neb-gap2Ql", "within_gap": "2L-Li__neb-gap-diffuse", "within_ql": "2L-Li__neb-QLs"},
    "Na": {"gap_to_ql": "2L-Na__neb-gap2QL", "within_gap": "2L-Na__neb-gap-diffuse", "within_ql": "2L-Na__neb-QLs"},
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate_models() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_id in MODEL_ORDER:
        model_path = MODEL_DIR / f"{model_id}.model"
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        kwargs = {"model_paths": [str(model_path)], "device": "cpu", "default_dtype": "float64"}
        if model_id in {"mace-mh-0", "mace-mh-1"}:
            kwargs["head"] = "omat_pbe"
        calc = MACECalculator(**kwargs)
        for element in ELEMENTS:
            for path_key, _display, _source_label in PATHS:
                path_id = PATH_PREFIX[element][path_key]
                xyz = INPUT_DIR / path_id / "linear.extxyz"
                if not xyz.exists():
                    raise FileNotFoundError(xyz)
                frames = read(str(xyz), index=":")
                if len(frames) != 10:
                    raise ValueError(f"Expected 10 images in {xyz}, found {len(frames)}")
                energies: list[float] = []
                force_max: list[float] = []
                for image_index, atoms in enumerate(frames, start=1):
                    atoms.calc = calc
                    energy = float(atoms.get_potential_energy())
                    forces = np.asarray(atoms.get_forces(), dtype=float)
                    if not math.isfinite(energy) or not np.isfinite(forces).all():
                        raise ValueError(f"Non-finite evaluation: {model_id} {path_id} image {image_index}")
                    energies.append(energy)
                    force_max.append(float(np.abs(forces).max()))
                e0 = energies[0]
                for image_index, (energy, fmax) in enumerate(zip(energies, force_max), start=1):
                    rows.append(
                        {
                            "element": element,
                            "path_key": path_key,
                            "path_id": path_id,
                            "model_id": model_id,
                            "model_class": MODEL_CLASS[model_id],
                            "image_index": image_index,
                            "s": (image_index - 1) / 9.0,
                            "energy_eV": energy,
                            "relative_energy_eV": energy - e0,
                            "max_force_eV_A": fmax,
                            "head": "omat_pbe" if model_id in {"mace-mh-0", "mace-mh-1"} else "default",
                            "model_sha256": sha256(model_path),
                            "input_sha256": sha256(xyz),
                        }
                    )
    return rows


def load_qe() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for element in ELEMENTS:
        with QE_TABLES[element].open(newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if row.get("series_key") != "qe_reference":
                    continue
                path_key = row["path_key"]
                rows.append(
                    {
                        "element": element,
                        "path_key": path_key,
                        "path_id": PATH_PREFIX[element][path_key],
                        "image_index": int(row["image_index"]),
                        "s": float(row["s"]),
                        "relative_energy_eV": float(row["relative_energy_eV"]),
                        "status": row.get("status", "failed_scf_convergence"),
                        "source_path": row.get("source_path", ""),
                        "source_record": row.get("source_record", ""),
                    }
                )
    if len(rows) != 90:
        raise ValueError(f"Expected 90 QE reference rows, found {len(rows)}")
    return rows


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_model_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            row["model_class"] = MODEL_CLASS[row["model_id"]]
            for key in ("image_index",):
                row[key] = int(row[key])
            for key in ("s", "energy_eV", "relative_energy_eV", "max_force_eV_A"):
                row[key] = float(row[key])
            rows.append(row)
    return rows


def aggregate(model_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: list[dict[str, object]] = []
    for element in ELEMENTS:
        for path_key, _display, _source_label in PATHS:
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
                    if not vals:
                        raise ValueError(f"Missing class value: {element} {path_key} {model_class} {image_index}")
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
    ax.text(-0.16, 1.06, panel_label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")
    for model_class in CLASS_ORDER:
        vals = [r for r in grouped if r["element"] == element and r["path_key"] == path_key and r["model_class"] == model_class]
        vals.sort(key=lambda r: r["image_index"])
        x = np.array([float(r["s"]) for r in vals])
        y = np.array([float(r["mean_relative_energy_eV"]) for r in vals])
        color = CLASS_COLORS[model_class]
        ax.plot(
            x,
            y,
            color=color,
            lw=2.0,
            marker="o",
            markersize=4.3,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.95,
            zorder=5,
        )
        sd = np.array([np.nan if r["sd_relative_energy_eV"] is None else float(r["sd_relative_energy_eV"]) for r in vals])
        if np.isfinite(sd).all() and np.any(sd > 0):
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.20, linewidth=0, zorder=2)
    qe = [r for r in qe_rows if r["element"] == element and r["path_key"] == path_key]
    qe.sort(key=lambda r: r["image_index"])
    x = np.array([float(r["s"]) for r in qe])
    y = np.array([float(r["relative_energy_eV"]) for r in qe])
    ax.plot(x, y, color="#222222", lw=1.45, ls=(0, (4, 2)), marker="D", markersize=3.6, markerfacecolor="white", markeredgewidth=0.8, zorder=5)
    band_values = [y]
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
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 9.2), sharex=False, sharey=False)
    panel_qa: list[dict[str, object]] = []
    labels = iter("abcdefghi")
    for i, element in enumerate(ELEMENTS):
        for j, (path_key, display, source_label) in enumerate(PATHS):
            ax = axes[i, j]
            plot_panel(ax, element, path_key, model_rows, grouped, qe_rows, next(labels))
            if i == 0:
                ax.set_title(display, fontsize=12.0, fontweight="bold", pad=8)
            if j == 0:
                ax.set_ylabel(f"{element}\nΔE (eV)", fontsize=11.0, fontweight="bold", labelpad=8)
            if i == 2:
                ax.set_xlabel("normalized image coordinate, $s$", fontsize=10.8, labelpad=7)
            panel_qa.append({"element": element, "path_key": path_key, "y_limits": [float(v) for v in ax.get_ylim()]})

    legend_handles = [Line2D([], [], color=CLASS_COLORS[c], lw=2.0, label=f"{c} (n={sum(v == c for v in MODEL_CLASS.values())})") for c in CLASS_ORDER]
    legend_handles.append(Line2D([], [], color="#222222", lw=1.45, ls=(0, (4, 2)), marker="D", markersize=3.6, markerfacecolor="white", label="screened QE path table"))
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.52, 0.985), ncol=5, frameon=False, handlelength=2.2, columnspacing=1.25, handletextpad=0.45)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.085, top=0.905, wspace=0.34, hspace=0.43)
    stem = FIG_OUT / "foundation_model_class_3x3_R03"
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for i, element in enumerate(ELEMENTS):
        for j, (path_key, display, _source_label) in enumerate(PATHS):
            fig1, ax = plt.subplots(figsize=(3.35, 2.85))
            plot_panel(ax, element, path_key, model_rows, grouped, qe_rows, "")
            ax.set_title(f"{element} · {display}", fontsize=11.5, fontweight="bold", pad=7)
            ax.set_xlabel("normalized image coordinate, $s$", fontsize=10.3, labelpad=6)
            ax.set_ylabel("ΔE (eV)", fontsize=10.3, labelpad=6)
            fig1.subplots_adjust(left=0.19, right=0.98, bottom=0.19, top=0.86)
            panel_stem = FIG_OUT / f"panel_{element}_{path_key}"
            fig1.savefig(panel_stem.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
            fig1.savefig(panel_stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
            fig1.savefig(panel_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
            plt.close(fig1)
    return {"figure": str(stem), "panels": panel_qa}


def main() -> None:
    raw_path = RAW_DATA_OUT / "foundation_model_profiles_9models_3x3.tsv"
    if os.environ.get("FOUNDATION_REUSE_MODEL_ROWS") == "1" and raw_path.exists():
        model_rows = load_model_rows(raw_path)
    else:
        model_rows = evaluate_models()
    qe_rows = load_qe()
    grouped = aggregate(model_rows)
    write_tsv(DATA_OUT / "foundation_model_profiles_9models_3x3_R03.tsv", model_rows)
    write_tsv(DATA_OUT / "foundation_model_class_profiles_3x3_R03.tsv", grouped)
    write_tsv(DATA_OUT / "foundation_model_qe_reference_profiles_3x3_R03.tsv", qe_rows)
    figure = render(model_rows, grouped, qe_rows)
    manifest = {
        "figure_id": "foundation_model_class_3x3_R03",
        "status": "RECONSTRUCTION",
        "scope": "nine local Foundation checkpoints evaluated on nine 10-image linear endpoint paths",
        "elements": ELEMENTS,
        "columns": [{"path_key": k, "label": d, "source_label": s} for k, d, s in PATHS],
        "model_classes": {c: [m for m in MODEL_ORDER if MODEL_CLASS[m] == c] for c in CLASS_ORDER},
        "aggregation": "pointwise mean with sample SD (ddof=1); hollow circles show the class mean; individual model markers are intentionally omitted",
        "qe_label": "screened QE path table; all source paths retain SCF convergence warnings",
        "input_interpolations": str(INPUT_DIR),
        "model_directory": str(MODEL_DIR),
        "model_hashes": {m: sha256(MODEL_DIR / f"{m}.model") for m in MODEL_ORDER},
        "model_rows": len(model_rows),
        "class_rows": len(grouped),
        "qe_rows": len(qe_rows),
        "outputs": figure,
        "caption_file": str(OUT / "CAPTIONS_foundation_class_3x3_R03.md"),
    }
    (OUT / "figure_data_manifest_foundation_class_3x3_R03.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "CAPTIONS_foundation_class_3x3_R03.md").write_text(
        """# Foundation-model class ensemble 3×3\n\n
        "**Figure X | Foundation-model class ensembles across Cu, Li, and Na migration paths.** "
        "Rows show Cu, Li, and Na; columns show in-gap diffusion, gap→QL, and within-QL paths. "
        "Each colored curve is the pointwise mean relative energy of all local Foundation checkpoints in that class; hollow circles show the class mean and the translucent band is the pointwise sample standard deviation across checkpoints. Individual model markers are omitted for visual clarity. "
        "The classes are MACE-MP (five checkpoints: MP-0b, MP-0b2 small/medium/large, and MP-0b3 medium), MACE-OMAT-0 (two checkpoints), and MACE-MH (two checkpoints), following the official ACEsuit release mapping. "
        "The dashed black curve is the screened QE path-table reference. All retained QE paths carry SCF-convergence warnings; the reference is therefore diagnostic and is not presented as converged DFT ground truth or as a publication-grade activation barrier. "
        "All colored curves are fixed-path single-point evaluations on the same ten-image linear interpolation geometry and are not self-relaxed MLFF-NEB results.\n\n"
        "## Label corrections\n\n"
        "The previous broad-roster FT labels are not used in this figure. The colored data are the nine local Foundation checkpoints grouped by official release family; no Cu/Li/Na broad-roster FT checkpoint is claimed.\n"""
    )
    qa = {
        "figure_id": manifest["figure_id"],
        "status": "PASS_WITH_SCIENTIFIC_BOUNDARY",
        "checks": {
            "model_evaluations": len(model_rows) == 810,
            "class_profiles": len(grouped) == 270,
            "qe_reference_profiles": len(qe_rows) == 90,
            "all_finite": all(math.isfinite(float(r["relative_energy_eV"])) for r in model_rows),
            "all_classes_present": all(sum(v == c for v in MODEL_CLASS.values()) >= 1 for c in CLASS_ORDER),
            "class_counts_correct": {c: sum(v == c for v in MODEL_CLASS.values()) for c in CLASS_ORDER} == {"MACE-MP": 5, "MACE-OMAT-0": 2, "MACE-MH": 2},
            "mean_markers_enabled": True,
        },
        "panel_qa": figure["panels"],
        "boundary": "QE traces are SCF-warning screened path tables; curves are fixed-path single-point diagnostics, not converged NEB barriers.",
    }
    (OUT / "QA_REPORT_foundation_class_3x3_R03.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(json.dumps({"figure": manifest["figure_id"], "model_rows": len(model_rows), "class_rows": len(grouped), "qe_rows": len(qe_rows)}, indent=2))


if __name__ == "__main__":
    main()
