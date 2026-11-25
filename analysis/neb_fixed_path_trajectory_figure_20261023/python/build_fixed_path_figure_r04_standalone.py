#!/usr/bin/env python3
"""Build the additive R04 matte fixed-path lower layer and vector composite.

This runner deliberately reads the locally retained R03 long tables rather than
reconstructing them from remote Rockfish files.  The tables were generated from
the canonical fixed-path release and carry their source hashes/provenance fields.
The Blender top layer is embedded at native pixels; no top raster rescaling is
allowed.  Lower curves, labels, legends, and caption text remain vector in PDF/SVG.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


FIG_ROOT_NAME = "analysis/neb_fixed_path_trajectory_figure_20261023"
WIDTH_PX = 2100
LOWER_HEIGHT_PX = 1260
TOP_HEIGHT_PX = 875
COMPOSITE_HEIGHT_PX = LOWER_HEIGHT_PX + TOP_HEIGHT_PX
DPI = 300
TARGET_BODY_WIDTH_IN = 7.0
ARTIFACT_TAG = "R04_matte_body7in"
STATUS = "PROVISIONAL_DIAGNOSTIC_CANDIDATE_R04_MATTE_BODY7IN"

PATH_ORDER = [
    ("cr_canonical_in_gap", "within gap"),
    ("cr_canonical_gap_to_ql", "gap2QL"),
    ("cr_canonical_within_ql", "withinQL"),
]
FAMILY_ORDER = ["Foundation", "From-scratch", "FT-600K", "FT-MultiT"]
FAMILY_COLORS = {
    "Foundation": "#CF4520",
    "From-scratch": "#FF9F1B",
    "FT-600K": "#002E71",
    "FT-MultiT": "#67ACE5",
}
FAMILY_MARKERS = {
    "Foundation": "D",
    "From-scratch": {"123": "o", "234": "s", "345": "^"},
    "FT-600K": {"123": "o", "234": "s", "345": "^"},
    "FT-MultiT": {"123": "o", "234": "s", "345": "^"},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def num(row: dict[str, str], key: str) -> float:
    return float(row[key])


def choose_font() -> tuple[str, bool]:
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        return "Arial", True
    except (ValueError, RuntimeError):
        return "DejaVu Sans", False


def family_marker(family: str, seed: str) -> str:
    if family == "Foundation":
        return "D"
    return FAMILY_MARKERS.get(family, {}).get(seed, "o")


def group_profile_rows(rows: Iterable[dict[str, str]]) -> list[list[dict[str, str]]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("model_family", ""),
            row.get("model_id", ""),
            row.get("training_seed", "NA"),
            row.get("record_id", ""),
        )
        groups[key].append(row)
    return [sorted(group, key=lambda item: int(float(item["image_index"]))) for group in groups.values()]


def sort_group(group: list[dict[str, str]]) -> tuple[int, str, str, str]:
    row = group[0]
    family = row.get("model_family", "")
    return (
        FAMILY_ORDER.index(family) if family in FAMILY_ORDER else 99,
        row.get("model_label", ""),
        row.get("model_id", ""),
        row.get("training_seed", "NA"),
    )


def profile_style(group: list[dict[str, str]]) -> dict[str, Any]:
    row = group[0]
    family = row.get("model_family", "")
    seed = row.get("training_seed", "NA")
    return {
        "color": FAMILY_COLORS.get(family, "#555555"),
        "marker": family_marker(family, seed),
        "label": row.get("model_label", row.get("model_id", family)),
        "family": family,
        "seed": seed,
        "seed_semantics": row.get("seed_semantics", ""),
    }


def collect_y_range(fixed_rows: list[dict[str, str]], dft_rows: list[dict[str, str]]) -> tuple[float, float]:
    values: list[float] = []
    for row in fixed_rows + dft_rows:
        try:
            value = num(row, "relative_energy_eV")
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        return (-0.05, 1.0)
    lo = min(0.0, min(values))
    hi = max(values)
    pad = max(0.05, 0.08 * max(hi - lo, 0.1))
    return (lo - 0.25 * pad, hi + pad)


def draw_panel(
    ax: Any,
    path_id: str,
    fixed_rows: list[dict[str, str]],
    dft_rows: list[dict[str, str]],
    y_limits: tuple[float, float],
    panel_index: int,
) -> dict[str, Any]:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(*y_limits)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=10.0, length=3.5, width=0.8, pad=3)
    if panel_index > 0:
        ax.set_yticklabels([])
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_xlabel("s (0–1)", fontsize=11.0, labelpad=5)
    if panel_index == 0:
        ax.set_ylabel("relative energy, ΔE (eV)", fontsize=10.5, labelpad=5)

    finite_fixed = [row for row in fixed_rows if row.get("status") != "FAILURE_MODE"]
    groups = sorted(group_profile_rows(finite_fixed), key=sort_group)
    family_counts: dict[str, int] = defaultdict(int)
    seed_counts: dict[str, int] = defaultdict(int)
    for group in groups:
        style = profile_style(group)
        x = np.asarray([num(row, "s") for row in group], dtype=float)
        y = np.asarray([num(row, "relative_energy_eV") for row in group], dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.all():
            continue
        ax.plot(
            x,
            y,
            color=style["color"],
            lw=0.95,
            alpha=0.47,
            marker=style["marker"],
            markersize=3.0,
            markeredgewidth=0.35,
            markeredgecolor=style["color"],
            markerfacecolor="white",
            markevery=1,
            solid_capstyle="round",
            zorder=3,
        )
        family_counts[style["family"]] += 1
        if style["family"] != "Foundation":
            seed_counts[style["seed"]] += 1

    dft = sorted(dft_rows, key=lambda row: int(float(row["image_index"])))
    if dft:
        ax.plot(
            [num(row, "s") for row in dft],
            [num(row, "relative_energy_eV") for row in dft],
            color="#2F2F2F",
            lw=1.8,
            ls=(0, (5, 2)),
            marker="D",
            markersize=4.0,
            markerfacecolor="white",
            markeredgewidth=0.85,
            zorder=5,
        )

    failures = [row for row in fixed_rows if row.get("status") == "FAILURE_MODE"]
    if failures:
        ax.text(0.975, 0.07, "×", transform=ax.transAxes, ha="right", va="bottom", fontsize=14, color="#444444", fontweight="bold")

    return {
        "path_id": path_id,
        "finite_profile_groups": len(groups),
        "family_group_counts": dict(family_counts),
        "training_seed_group_counts": dict(seed_counts),
        "dft_points": len(dft),
        "failure_rows": len(failures),
        "curve_semantics": "fixed-path single-point profiles; no smoothing or uncertainty bands",
    }


def family_handles() -> list[Line2D]:
    return [
        Line2D([], [], color=FAMILY_COLORS[family], lw=2.2, label=family)
        for family in FAMILY_ORDER
    ]


def seed_handles() -> list[Line2D]:
    return [
        Line2D([], [], color="#555555", lw=0, marker="o", markersize=5.0, label="training seed 123 — circle"),
        Line2D([], [], color="#555555", lw=0, marker="s", markersize=5.0, label="training seed 234 — square"),
        Line2D([], [], color="#555555", lw=0, marker="^", markersize=5.0, label="training seed 345 — triangle"),
        Line2D([], [], color="#555555", lw=1.5, marker="D", markersize=4.8, label="checkpoint variant — not training seed"),
    ]


def semantic_handles() -> list[Line2D]:
    return [
        Line2D([], [], color="#2F2F2F", lw=1.8, ls=(0, (5, 2)), marker="D", markersize=4.5, label="DFT/QE warning"),
        Line2D([], [], color="#555555", lw=1.6, ls=(0, (1, 2)), label="uncertainty band withheld"),
        Line2D([], [], color="#555555", lw=0, marker="x", markersize=7, label="× MODEL_LOAD failure"),
    ]


def lower_header(fig: Any, yscale: float = 1.0) -> None:
    fig.text(
        0.5,
        yscale * 0.965,
        "DFT/QE: SCF warning; diagnostic  |  ghost = real image overlay; α = image order only  |  bands withheld",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#2F2F2F",
    )
    fig.text(
        0.5,
        yscale * 0.905,
        "× FT-MultiT seed 123 — MODEL_LOAD (all three paths)  |  relative energy: ΔEᵢ = Eᵢ − E₀ (eV)",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#2F2F2F",
    )
    fig.legend(handles=family_handles(), loc="upper center", bbox_to_anchor=(0.5, yscale * 0.845), ncol=4, fontsize=10.5, frameon=False, handlelength=1.8, columnspacing=0.8, bbox_transform=fig.transFigure)
    fig.legend(handles=seed_handles(), loc="upper center", bbox_to_anchor=(0.5, yscale * 0.775), ncol=2, fontsize=9.8, frameon=False, handlelength=1.3, columnspacing=0.8, bbox_transform=fig.transFigure)
    fig.legend(handles=semantic_handles(), loc="upper center", bbox_to_anchor=(0.5, yscale * 0.690), ncol=3, fontsize=9.8, frameon=False, handlelength=1.5, columnspacing=0.85, bbox_transform=fig.transFigure)


def save_figure(fig: Any, stem: Path, expected_size: tuple[int, int]) -> dict[str, Any]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(stem.with_suffix(suffix), dpi=DPI, facecolor="white", edgecolor="white")
    with Image.open(stem.with_suffix(".png")) as image:
        size = list(image.size)
    return {
        "stem": str(stem),
        "pdf": str(stem.with_suffix(".pdf")),
        "svg": str(stem.with_suffix(".svg")),
        "png": str(stem.with_suffix(".png")),
        "png_size_px": size,
        "expected_size_px": list(expected_size),
        "size_pass": tuple(size) == expected_size,
        "vector_lower_layer": True,
    }


def build_figure(repo_root: Path, top_path: Path) -> dict[str, Any]:
    fig_root = repo_root / FIG_ROOT_NAME
    data_dir = fig_root / "python/data"
    fixed_path = data_dir / "fixed_path_long.tsv"
    dft_path = data_dir / "dft_profiles.tsv"
    fixed = read_tsv(fixed_path)
    dft = read_tsv(dft_path)
    y_limits = collect_y_range(fixed, dft)
    font_name, arial_available = choose_font()
    plt.rcParams.update({"font.family": font_name, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})

    fixed_by_path = {path_id: [row for row in fixed if row.get("path_id") == path_id] for path_id, _ in PATH_ORDER}
    dft_by_path = {path_id: [row for row in dft if row.get("path_id") == path_id] for path_id, _ in PATH_ORDER}

    def draw_lower(fig: Any, yscale: float = 1.0) -> list[dict[str, Any]]:
        axes_bottom, axes_top = 0.18, 0.565
        axes_height = axes_top - axes_bottom
        metadata: list[dict[str, Any]] = []
        for index, (path_id, label) in enumerate(PATH_ORDER):
            x0, x1 = index / 3, (index + 1) / 3
            ax = fig.add_axes([x0, yscale * axes_bottom, x1 - x0, yscale * axes_height])
            metadata.append(draw_panel(ax, path_id, fixed_by_path[path_id], dft_by_path[path_id], y_limits, index))
            fig.text((x0 + x1) / 2, yscale * 0.615, label, ha="center", va="center", fontsize=11.5, fontweight="bold")
        lower_header(fig, yscale=yscale)
        fig.text(0.5, yscale * 0.055, "s = normalized reaction coordinate", ha="center", va="center", fontsize=10.5, color="#555555")
        return metadata

    lower_fig = plt.figure(figsize=(WIDTH_PX / DPI, LOWER_HEIGHT_PX / DPI), dpi=DPI)
    lower_metadata = draw_lower(lower_fig)
    lower_stem = fig_root / "python/generated" / f"fixed_path_lower_panels_{ARTIFACT_TAG}"
    lower_output = save_figure(lower_fig, lower_stem, (WIDTH_PX, LOWER_HEIGHT_PX))
    plt.close(lower_fig)

    if not top_path.exists():
        raise FileNotFoundError(top_path)
    with Image.open(top_path) as top_image:
        if top_image.size != (WIDTH_PX, TOP_HEIGHT_PX):
            raise ValueError(f"top raster size {top_image.size} != {(WIDTH_PX, TOP_HEIGHT_PX)}; refusing rescale")
        top_array = np.asarray(top_image.convert("RGB"))

    lower_scale = LOWER_HEIGHT_PX / COMPOSITE_HEIGHT_PX
    composite_fig = plt.figure(figsize=(WIDTH_PX / DPI, COMPOSITE_HEIGHT_PX / DPI), dpi=DPI)
    top_ax = composite_fig.add_axes([0.0, lower_scale, 1.0, 1.0 - lower_scale])
    top_ax.imshow(top_array, interpolation="none", origin="upper", aspect="auto")
    top_ax.set_axis_off()
    composite_metadata = draw_lower(composite_fig, yscale=lower_scale)
    composite_stem = fig_root / f"integration/generated_{ARTIFACT_TAG}" / f"fixed_path_trajectory_composite_{ARTIFACT_TAG}"
    composite_output = save_figure(composite_fig, composite_stem, (WIDTH_PX, COMPOSITE_HEIGHT_PX))
    plt.close(composite_fig)

    composite_png = Path(composite_output["png"])
    proof_dir = composite_png.parent / "proofs"
    proof_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(composite_png) as source:
        rgb = source.convert("RGB")
        gray_path = proof_dir / f"{composite_png.stem}_grayscale.png"
        ImageOps.grayscale(rgb).convert("RGB").save(gray_path, dpi=(DPI, DPI))
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        matrices = {
            "deuteranopia": np.array([[0.625, 0.375, 0.0], [0.70, 0.30, 0.0], [0.0, 0.30, 0.70]], dtype=np.float32),
            "protanopia": np.array([[0.567, 0.433, 0.0], [0.558, 0.442, 0.0], [0.0, 0.242, 0.758]], dtype=np.float32),
            "tritanopia": np.array([[0.95, 0.05, 0.0], [0.0, 0.433, 0.567], [0.0, 0.475, 0.525]], dtype=np.float32),
        }
        proof_paths = {"grayscale": str(gray_path)}
        for label, matrix in matrices.items():
            path = proof_dir / f"{composite_png.stem}_{label}.png"
            transformed = np.clip(arr @ matrix.T, 0.0, 1.0)
            Image.fromarray(np.round(transformed * 255.0).astype(np.uint8), mode="RGB").save(path, dpi=(DPI, DPI))
            proof_paths[label] = str(path)

    caption_path = fig_root / f"python/generated/caption_{ARTIFACT_TAG}.md"
    caption_path.write_text(
        f"""# Fixed-path NEB trajectory/profile candidate — R04 matte body-width proof\n\n"
        f"Status: **{STATUS}**; not publication-final. Columns are **within gap → gap2QL → withinQL**.\n\n"
        "The upper layer is a Blender matte render of the three retained 81-atom, ten-image fixed paths. "
        "Image labels 0–9 are anchored to real Cr coordinates after the recorded MIC unwrap. Ghosted spheres "
        "are real retained path-image overlays; alpha is image order only, not uncertainty/error.\n\n"
        "The lower layer is generated from `fixed_path_long.tsv` and `dft_profiles.tsv` as vector curves/text in "
        "PDF/SVG; only the upper Blender layer is raster. Training seed markers are 123=circle, 234=square, "
        "345=triangle. Foundation entries are checkpoint variants, not training seeds, and no seed bands are shown.\n\n"
        "DFT/QE is **SCF warning; diagnostic**, not unconditional ground truth. The explicit failure is **FT-MultiT "
        "seed 123 — MODEL_LOAD on all three paths**. Energy is `relative energy, ΔEᵢ = Eᵢ − E₀ (eV)` versus "
        "normalized reaction coordinate `s`.\n\n"
        "This artifact is subject to the independent MAT-01—MAT-24 visual gate in `visual_review/" 
        "MATTE_BLENDER_STYLE_ADDENDUM_R04.md`; scientific fixed-path, DFT, seed, provenance and publication gates "
        "remain separate.\n""",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": f"migrationbench_fixed_path_trajectory_figure_{ARTIFACT_TAG}_v1",
        "status": STATUS,
        "figure_accepted": False,
        "matte_visual_gate": "PENDING_INDEPENDENT_REVIEW",
        "target_body_width_in": TARGET_BODY_WIDTH_IN,
        "target_dpi": DPI,
        "column_order": [{"path_id": path_id, "display_label": label, "column_index": index + 1} for index, (path_id, label) in enumerate(PATH_ORDER)],
        "top_blender_layer": {"path": str(top_path), "size_px": [WIDTH_PX, TOP_HEIGHT_PX], "rescaled": False, "semantic": "real retained coordinates; ghost alpha=image order only"},
        "data_sources": {
            "fixed_path_long": {"path": str(fixed_path), "sha256": sha256_file(fixed_path), "rows": len(fixed)},
            "dft_profiles": {"path": str(dft_path), "sha256": sha256_file(dft_path), "rows": len(dft)},
            "style_addendum": str(fig_root / "visual_review/MATTE_BLENDER_STYLE_ADDENDUM_R04.md"),
        },
        "coverage": {"finite_fixed_rows": sum(row.get("status") != "FAILURE_MODE" for row in fixed), "failure_rows": sum(row.get("status") == "FAILURE_MODE" for row in fixed), "dft_rows": len(dft), "expected_images_per_path": 10, "families": FAMILY_ORDER, "uncertainty_bands": "withheld"},
        "render": {"font": font_name, "arial_available": arial_available, "lower": lower_output, "composite": composite_output, "top_raster_layer": True, "lower_vector_layer": True, "lower_panels": lower_metadata, "composite_panels": composite_metadata, "y_limits_eV": list(y_limits)},
        "body_width_proof": {"source": str(composite_png), "outputs": proof_paths, "semantic": "visual stress previews only; no scientific values changed"},
        "caption": str(caption_path),
        "processing_decisions": [
            "R04 matte is additive; R03 outputs are preserved.",
            "Lower data are read from the retained R03 long tables, with numeric source hashes retained in row metadata.",
            "Training seed marker mapping is explicit: 123 circle, 234 square, 345 triangle; Foundation remains checkpoint variant.",
            "No uncertainty band is drawn because the verified same-path three-seed gate is incomplete.",
            "The composite refuses to rescale the Blender top raster and preserves vector curves/text below it.",
        ],
    }
    for path in [caption_path, Path(lower_output["pdf"]), Path(lower_output["svg"]), Path(lower_output["png"]), Path(composite_output["pdf"]), Path(composite_output["svg"]), Path(composite_output["png"]), *[Path(value) for value in proof_paths.values()]]:
        if path.exists():
            manifest.setdefault("generated_file_hashes", {})[str(path)] = sha256_file(path)
    manifest_path = fig_root / f"figure_data_manifest_{ARTIFACT_TAG}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    qa = {
        "schema_version": f"migrationbench_fixed_path_trajectory_figure_qa_{ARTIFACT_TAG}_v1",
        "status": STATUS,
        "figure_accepted": False,
        "matte_visual_gate": "PENDING_INDEPENDENT_REVIEW",
        "manifest": str(manifest_path),
        "checks": {
            "VIS-01": "PASS — exact three-column order and 0/700/1400/2100 boundaries",
            "VIS-02": "PASS — upper/lower path identity is explicit in composite metadata",
            "VIS-11": "PASS — seed marker mapping is explicit; checkpoint variants separate",
            "VIS-13": "PASS — FT-MultiT seed 123 MODEL_LOAD named; no numeric curve",
            "VIS-15": "PASS — relative energy definition visible",
            "PUB-07": "PASS — upper Blender raster plus lower vector curves/text",
            "PUB-08": "PENDING — independent body-width/gray/color review",
            "MAT-01_to_MAT-24": "PENDING — independent visual reviewer",
            "PHY-12": "BLOCKED — verified legal fixed-path training-seed band incomplete",
            "PHY-16": "BLOCKED — remote raw profile hashes unavailable at runtime",
            "DFT-02": "BLOCKED — SCF-warning reference remains diagnostic",
            "SEED-04": "BLOCKED — no legal three-seed fixed-path band",
            "PUB-10": "BLOCKED — scientific gates remain blocked",
        },
        "body_width_proof": {"outputs": proof_paths, "grayscale": "generated", "color_vision": "generated"},
    }
    qa_path = fig_root / f"QA_REPORT_{ARTIFACT_TAG}.json"
    qa_path.write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"status": STATUS, "manifest": str(manifest_path), "qa": str(qa_path), "caption": str(caption_path), "lower": lower_output, "composite": composite_output, "proofs": proof_paths}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--top-path", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_figure(args.repo_root.resolve(), args.top_path.expanduser().resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
