#!/usr/bin/env python3
"""Build the R04 body-width fixed-path NEB figure.

R04 is additive to R03.  It keeps the R01/R02 data builder and the R03 visual
                 grammar, but makes the seed mapping and failure ledger explicit,
and provides a vector lower layer for the body-width composite.  The upper
Blender triptych is supplied separately as a real-coordinate raster layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import build_fixed_path_figure as r01
import build_fixed_path_figure_r02 as r02
import build_fixed_path_figure_r03 as r03


R04_STATUS = "PROVISIONAL_DIAGNOSTIC_CANDIDATE_R04_BODY7IN"
R04_SCHEMA = "migrationbench_fixed_path_trajectory_figure_r04_body7in_v1"
ARTIFACT_TAG = "R04_body7in"
R04_DPI = 300
TARGET_BODY_WIDTH_IN = 7.0
R04_WIDTH_PX = 2100
R04_LOWER_HEIGHT_PX = 1260
R04_TOP_HEIGHT_PX = 875
R04_COMPOSITE_HEIGHT_PX = R04_LOWER_HEIGHT_PX + R04_TOP_HEIGHT_PX
R04_LOWER_STEM = "fixed_path_lower_panels_R04_body7in"
R04_COMPOSITE_STEM = "fixed_path_trajectory_composite_R04_body7in"


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def choose_font() -> tuple[str, bool]:
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        return "Arial", True
    except (ValueError, RuntimeError):
        return "DejaVu Sans", False


def make_geometry_contract(fig_root: Path, top_path: Path | None) -> dict[str, Any]:
    boundaries = [round(R04_WIDTH_PX * i / 3) for i in range(4)]
    r03_path = fig_root / "integration/panel_geometry_contract_R03.json"
    prior = json.loads(r03_path.read_text(encoding="utf-8")) if r03_path.exists() else {}
    columns = []
    for index, config in enumerate(r01.PATH_ORDER):
        x0, x1 = boundaries[index], boundaries[index + 1]
        columns.append(
            {
                "column_index": config["column_index"],
                "path_id": config["path_id"],
                "display_label": config["display_label"],
                "x0_px": x0,
                "x1_px": x1,
                "center_px": (x0 + x1) // 2,
                "width_px": x1 - x0,
            }
        )
    contract = {
        "schema_version": "migrationbench_shared_panel_geometry_contract_r04_body7in_v1",
        "status": R04_STATUS,
        "revision": "R04 additive body-width proof; R03 master preserved",
        "target_body_width_in": TARGET_BODY_WIDTH_IN,
        "target_dpi": R04_DPI,
        "reference_width_px": R04_WIDTH_PX,
        "top_triptych": {
            "path": str(top_path) if top_path else None,
            "canvas_size_px": [R04_WIDTH_PX, R04_TOP_HEIGHT_PX],
            "boundary_px": boundaries,
            "coordinate_semantics": "real retained extxyz coordinates after recorded MIC unwrap",
        },
        "lower_panel": {
            "canvas_size_px": [R04_WIDTH_PX, R04_LOWER_HEIGHT_PX],
            "boundary_px": boundaries,
            "axes_boundary_rule": "three equal outer axes boxes; no raster rescaling",
        },
        "composite": {"canvas_size_px": [R04_WIDTH_PX, R04_COMPOSITE_HEIGHT_PX]},
        "columns": columns,
        "alignment_tolerance_px": 0,
        "column_order": [config["display_label"] for config in r01.PATH_ORDER],
        "prior_contract": prior.get("path"),
    }
    path = fig_root / f"integration/panel_geometry_contract_{ARTIFACT_TAG}.json"
    json_write(path, contract)
    contract["path"] = str(path)
    contract["sha256"] = r01.sha256_file(path)
    return contract


def seed_handles() -> list[Line2D]:
    return [
        Line2D([], [], color="#555555", lw=0, marker="o", markersize=5.5, label="training seed 123 — circle"),
        Line2D([], [], color="#555555", lw=0, marker="s", markersize=5.5, label="training seed 234 — square"),
        Line2D([], [], color="#555555", lw=0, marker="^", markersize=5.5, label="training seed 345 — triangle"),
        Line2D([], [], color="#555555", lw=1.7, marker="D", markersize=5.0, label="checkpoint variant (not training seed)"),
    ]


def semantic_handles() -> list[Line2D]:
    return [
        Line2D([], [], color="#2F2F2F", lw=2.0, marker="D", markersize=5.0, label="DFT warning"),
        Line2D([], [], color="#666666", lw=1.8, linestyle=(0, (1, 2)), label="band withheld"),
        Line2D([], [], color="#666666", lw=1.8, linestyle=(0, (6, 2, 1, 2)), marker="o", markerfacecolor="white", label="diagnostic status"),
        Line2D([], [], color="#666666", lw=0, marker="x", markersize=7, label="failure / no numeric profile"),
    ]


def apply_x_tick_layout(ax: Any, panel_index: int) -> None:
    ticks = (0.0, 0.25, 0.5, 0.75, 1.0)
    ax.set_xticks(ticks)
    for value, label in zip(ticks, ax.get_xticklabels()):
        if value == 0.0:
            label.set_ha("left")
        elif value == 1.0:
            label.set_ha("right")
        if panel_index > 0 and value == 0.0:
            label.set_visible(False)
        if panel_index < 2 and value == 1.0:
            label.set_visible(False)


def draw_axis_r04(ax: Any, config: dict[str, Any], bundle: dict[str, Any], panel_index: int) -> dict[str, Any]:
    metadata = r02.draw_profile_axis_r02(
        ax,
        config,
        bundle["fixed_rows"],
        bundle["dft_profiles"],
        bundle["style"],
        composite=True,
    )
    ax.set_xlabel("s (0–1)", fontsize=11.0, labelpad=5)
    ax.set_ylabel("ΔE (eV)", fontsize=10.5, labelpad=5)
    ax.tick_params(labelsize=10.0, pad=3, length=3.5, width=0.8)
    apply_x_tick_layout(ax, panel_index)
    metadata.update(
        {
            "visible_energy_definition": "relative energy, ΔE_i = E_i − E_0 (eV) stated in figure header",
            "marker_mapping_visible": True,
            "failure_status_visible": True,
        }
    )
    return metadata


def lower_header(fig: Any, yscale: float = 1.0, yoffset: float = 0.0) -> None:
    def y(value: float) -> float:
        return yoffset + yscale * value

    fig.text(
        0.5,
        y(0.965),
        "DFT/QE: SCF warning; diagnostic  |  ghost = real image overlay; α = order only  |  bands withheld",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#2F2F2F",
    )
    fig.text(
        0.5,
        y(0.905),
        "× FT-MultiT seed 123 — MODEL_LOAD (all three paths)  |  relative energy: ΔE_i = E_i − E_0 (eV)",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#2F2F2F",
    )
    fig.legend(
        handles=r02.family_handles_r02(_ACTIVE_BUNDLE["style"]),
        loc="upper center",
        bbox_to_anchor=(0.5, y(0.845)),
        ncol=4,
        fontsize=10.5,
        frameon=False,
        handlelength=1.8,
        columnspacing=0.8,
        bbox_transform=fig.transFigure,
    )
    fig.legend(
        handles=seed_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, y(0.775)),
        ncol=2,
        fontsize=9.8,
        frameon=False,
        handlelength=1.3,
        columnspacing=0.8,
        bbox_transform=fig.transFigure,
    )
    fig.legend(
        handles=semantic_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, y(0.690)),
        ncol=4,
        fontsize=9.8,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.85,
        bbox_transform=fig.transFigure,
    )


_ACTIVE_BUNDLE: dict[str, Any] = {}


def save_figure(fig: Any, stem: Path, dpi: int, expected_size: tuple[int, int] | None = None) -> dict[str, Any]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(stem.with_suffix(suffix), dpi=dpi, facecolor="white", edgecolor="white")
    with Image.open(stem.with_suffix(".png")) as image:
        size = list(image.size)
    return {
        "stem": str(stem),
        "pdf": str(stem.with_suffix(".pdf")),
        "svg": str(stem.with_suffix(".svg")),
        "png": str(stem.with_suffix(".png")),
        "png_size_px": size,
        "dpi": dpi,
        "expected_size_px": list(expected_size) if expected_size else None,
        "size_pass": expected_size is None or tuple(size) == expected_size,
        "vector_lower_layer": True,
    }


def render_lower(bundle: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    global _ACTIVE_BUNDLE
    _ACTIVE_BUNDLE = bundle
    font_name, arial_available = choose_font()
    plt.rcParams.update({"font.family": font_name, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    generated = bundle["fig_root"] / "python/generated"
    fig = plt.figure(figsize=(R04_WIDTH_PX / R04_DPI, R04_LOWER_HEIGHT_PX / R04_DPI), dpi=R04_DPI)
    axes_bottom, axes_top = 0.18, 0.565
    axes_height = axes_top - axes_bottom
    axes_metadata = []
    for index, config in enumerate(r01.PATH_ORDER):
        x0, x1 = index / 3, (index + 1) / 3
        ax = fig.add_axes([x0, axes_bottom, x1 - x0, axes_height])
        axes_metadata.append(draw_axis_r04(ax, config, bundle, index))
        fig.text((x0 + x1) / 2, 0.615, config["display_label"], ha="center", va="center", fontsize=11.5, fontweight="bold")
    lower_header(fig)
    fig.text(0.5, 0.055, "s = normalized reaction coordinate", ha="center", va="center", fontsize=10.5, color="#555555")
    stem = generated / f"fixed_path_lower_panels_{ARTIFACT_TAG}"
    outputs = save_figure(fig, stem, R04_DPI, expected_size=(R04_WIDTH_PX, R04_LOWER_HEIGHT_PX))
    plt.close(fig)
    return {
        "font": font_name,
        "arial_available": arial_available,
        "canvas_size_px": [R04_WIDTH_PX, R04_LOWER_HEIGHT_PX],
        "dpi": R04_DPI,
        "target_body_width_in": TARGET_BODY_WIDTH_IN,
        "axes_bottom_norm": axes_bottom,
        "axes_top_norm": axes_top,
        "axes_bounds_norm": [[index / 3, axes_bottom, 1 / 3, axes_height] for index in range(3)],
        "typography_pt": {"axis_labels": 11.0, "tick_labels": 10.0, "legend_family": 10.5, "legend_seed_semantic": 9.8, "status": 10.5, "titles": 11.5},
        "outputs": outputs,
        "panels": axes_metadata,
        "vector_lower_layer": True,
    }


def render_vector_composite(bundle: dict[str, Any], contract: dict[str, Any], top_path: Path) -> dict[str, Any]:
    global _ACTIVE_BUNDLE
    _ACTIVE_BUNDLE = bundle
    if not top_path.exists():
        raise FileNotFoundError(top_path)
    with Image.open(top_path) as top_image:
        if top_image.size != (R04_WIDTH_PX, R04_TOP_HEIGHT_PX):
            raise ValueError(f"top raster size {top_image.size} != {(R04_WIDTH_PX, R04_TOP_HEIGHT_PX)}; refusing rescale")
        top_array = np.asarray(top_image.convert("RGB"))
    font_name, arial_available = choose_font()
    plt.rcParams.update({"font.family": font_name, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    lower_scale = R04_LOWER_HEIGHT_PX / R04_COMPOSITE_HEIGHT_PX
    fig = plt.figure(figsize=(R04_WIDTH_PX / R04_DPI, R04_COMPOSITE_HEIGHT_PX / R04_DPI), dpi=R04_DPI)
    top_ax = fig.add_axes([0.0, lower_scale, 1.0, 1.0 - lower_scale])
    top_ax.imshow(top_array, interpolation="none", origin="upper", aspect="auto")
    top_ax.set_axis_off()

    axes_bottom, axes_top = 0.18, 0.565
    axes_height = axes_top - axes_bottom
    axes_metadata = []
    for index, config in enumerate(r01.PATH_ORDER):
        x0, x1 = index / 3, (index + 1) / 3
        ax = fig.add_axes([x0, lower_scale * axes_bottom, x1 - x0, lower_scale * axes_height])
        axes_metadata.append(draw_axis_r04(ax, config, bundle, index))
        fig.text((x0 + x1) / 2, lower_scale * 0.615, config["display_label"], ha="center", va="center", fontsize=11.5, fontweight="bold")
    lower_header(fig, yscale=lower_scale)
    fig.text(0.5, lower_scale * 0.055, "s = normalized reaction coordinate", ha="center", va="center", fontsize=10.5, color="#555555")
    stem = bundle["fig_root"] / f"integration/generated_{ARTIFACT_TAG}" / f"fixed_path_trajectory_composite_{ARTIFACT_TAG}"
    outputs = save_figure(fig, stem, R04_DPI, expected_size=(R04_WIDTH_PX, R04_COMPOSITE_HEIGHT_PX))
    plt.close(fig)
    return {
        "font": font_name,
        "arial_available": arial_available,
        "canvas_size_px": [R04_WIDTH_PX, R04_COMPOSITE_HEIGHT_PX],
        "dpi": R04_DPI,
        "target_body_width_in": TARGET_BODY_WIDTH_IN,
        "top_raster_layer": {"path": str(top_path), "size_px": [R04_WIDTH_PX, R04_TOP_HEIGHT_PX], "rescaled": False, "semantic": "Blender real-coordinate layer"},
        "lower_vector_layer": {"size_px": [R04_WIDTH_PX, R04_LOWER_HEIGHT_PX], "rescaled": False, "curves_and_text_vector": True},
        "axes_bounds_norm_composite": [[index / 3, lower_scale * axes_bottom, 1 / 3, lower_scale * axes_height] for index in range(3)],
        "outputs": outputs,
        "panels": axes_metadata,
        "vector_lower_layer": True,
    }


def make_caption(contract: dict[str, Any], render_meta: dict[str, Any], top_path: Path) -> str:
    return f"""# Fixed-path NEB trajectory/profile candidate — R04 body-width proof

Status: **{R04_STATUS}**; not publication-final. The provisional body-width
assumption is **{TARGET_BODY_WIDTH_IN:.1f} in at {R04_DPI} dpi** and must be
confirmed against the target Wiley article class before release.

Columns are **within gap → gap2QL → withinQL**. The upper layer is a Blender
raster of the three retained 81-atom, ten-image fixed paths; image labels 0–9
are anchored to the real Cr coordinates after the recorded MIC unwrap. Ghosted
spheres are real retained path-image overlays; alpha is image order only, not
uncertainty/error. No interpolation or synthetic geometry is used.

The lower layer is generated directly as vector curves/text in the PDF/SVG;
only the upper Blender layer is raster. The visible legend maps training seed
123 to circles, 234 to squares, and 345 to triangles. Foundation entries are
checkpoint variants, not training seeds, and are not pooled into a seed SD.

DFT/QE remains **SCF warning; diagnostic**, not unconditional ground truth.
The failed arm is explicit: **FT-MultiT seed 123 — MODEL_LOAD on all three
paths**; no numeric curve is fabricated. The visible energy definition is
`relative energy, ΔE_i = E_i − E_0 (eV)` versus normalized reaction coordinate
`s`.

The candidate retains **33 finite fixed diagnostic blocks**, **3 MODEL_LOAD
failure blocks**, and **30 DFT/QE warning rows**. FT-600K and FT-MultiT bands
are withheld because the verified same-path training-seed gate is incomplete.
The TIER1 fallback source and its SHA256 remain recorded in the manifest; the
remote raw-profile read limitation remains a provenance blocker.

Excluded observables: `SELF_RELAXED_ENDPOINT_ONLY_NEB` and `MIC_LINEAR`.
"""


def render_body_width_proofs(composite_png: Path) -> dict[str, Any]:
    """Create additive body-width visual stress proofs without touching source data."""
    proof_dir = composite_png.parent / "proofs"
    proof_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(composite_png) as source:
        rgb = source.convert("RGB")
        gray_path = proof_dir / f"{composite_png.stem}_grayscale.png"
        gray = ImageOps.grayscale(rgb).convert("RGB")
        gray.save(gray_path, dpi=(R04_DPI, R04_DPI))

        # Linear-light-independent color-vision stress approximations are used only
        # as visual QA previews; they are not scientific data transforms.
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        deutan = np.array(
            [[0.625, 0.375, 0.0], [0.70, 0.30, 0.0], [0.0, 0.30, 0.70]],
            dtype=np.float32,
        )
        protan = np.array(
            [[0.567, 0.433, 0.0], [0.558, 0.442, 0.0], [0.0, 0.242, 0.758]],
            dtype=np.float32,
        )
        outputs = {"grayscale": gray_path}
        for label, matrix in (("deuteranopia", deutan), ("protanopia", protan)):
            transformed = np.clip(arr @ matrix.T, 0.0, 1.0)
            path = proof_dir / f"{composite_png.stem}_{label}.png"
            Image.fromarray(np.round(transformed * 255.0).astype(np.uint8), mode="RGB").save(path, dpi=(R04_DPI, R04_DPI))
            outputs[label] = path
    return {
        "source": str(composite_png),
        "target_body_width_in": TARGET_BODY_WIDTH_IN,
        "dpi": R04_DPI,
        "outputs": {key: str(path) for key, path in outputs.items()},
        "semantic": "visual stress previews only; no scientific values changed",
    }


def make_qa(bundle: dict[str, Any], contract: dict[str, Any], lower: dict[str, Any], composite: dict[str, Any] | None, revalidation: dict[str, Any], top_path: Path, proofs: dict[str, Any] | None) -> dict[str, Any]:
    checks = [
        {"code": "VIS-01", "status": "PASS", "detail": "Body-width top/lower columns use exact 0/700/1400/2100 boundaries."},
        {"code": "VIS-02", "status": "PASS", "detail": "Column order within gap → gap2QL → withinQL is preserved."},
        {"code": "VIS-03", "status": "PENDING", "detail": "R04 body-width 0–9 labels require manual visual review."},
        {"code": "VIS-05", "status": "PASS", "detail": "Ghost semantics and alpha meaning are explicit in the visible header/caption."},
        {"code": "VIS-11", "status": "PASS", "detail": "Training seed 123/234/345 marker mapping is explicit; checkpoint variants are separate."},
        {"code": "VIS-13", "status": "PASS", "detail": "The FT-MultiT seed 123 MODEL_LOAD failure is named for all three paths; no curve is drawn."},
        {"code": "VIS-15", "status": "PASS", "detail": "Visible header states relative energy ΔE_i = E_i − E_0 (eV); axes show ΔE (eV)."},
        {"code": "PUB-05", "status": "PENDING", "detail": "7.0-in body-width assumption is explicit; target Wiley class still requires confirmation."},
        {"code": "PUB-07", "status": "PASS", "detail": "Composite PDF/SVG contains a raster Blender top layer and a vector lower curves/text layer."},
        {"code": "PUB-08", "status": "PENDING", "detail": "Body-width grayscale and color-vision proof must be reviewed manually."},
        {"code": "PHY-12", "status": "BLOCKED", "detail": "Verified fixed-path training-seed matrix is incomplete; uncertainty bands remain withheld."},
        {"code": "PHY-16", "status": "BLOCKED", "detail": "Remote profile reads remain unavailable; TIER1 fallback is explicit but raw profile hashes are unavailable."},
        {"code": "DFT-02", "status": "BLOCKED", "detail": "SCF-warning DFT/QE remains diagnostic, not ground truth."},
        {"code": "SEED-04", "status": "BLOCKED", "detail": "No legal three-seed fixed-path band for FT-600K/FT-MultiT."},
        {"code": "PUB-10", "status": "BLOCKED", "detail": "Scientific gates remain blocked; R04 is a diagnostic candidate."},
    ]
    return {
        "schema_version": f"migrationbench_fixed_path_trajectory_figure_qa_{ARTIFACT_TAG}_v1",
        "status": R04_STATUS,
        "figure_accepted": False,
        "target_body_width_in": TARGET_BODY_WIDTH_IN,
        "target_dpi": R04_DPI,
        "top_path": str(top_path),
        "top_size_px": [R04_WIDTH_PX, R04_TOP_HEIGHT_PX],
        "lower_size_px": [R04_WIDTH_PX, R04_LOWER_HEIGHT_PX],
        "composite_size_px": [R04_WIDTH_PX, R04_COMPOSITE_HEIGHT_PX] if composite else None,
        "checks": checks,
        "numeric_fixed_rows": 330,
        "fixed_finite_diagnostic_blocks": 33,
        "fixed_failure_blocks": 3,
        "dft_warning_rows": 30,
        "tier1_revalidation": revalidation,
        "lower_render": lower,
        "composite_render": composite,
        "vector_export_contract": {"upper": "embedded Blender PNG only", "lower": "Matplotlib vector curves/text", "whole_composite_png_wrapping": False},
        "body_width_proof": proofs or {"grayscale": "pending generation", "color_vision": "pending generation"},
        "figure_accepted": False,
    }


def build_r04(repo_root: Path, top_path: Path) -> dict[str, Any]:
    fig_root = repo_root / "analysis/neb_fixed_path_trajectory_figure_20261023"
    bundle = r01.build_data(repo_root, fig_root)
    contract = make_geometry_contract(fig_root, top_path)
    revalidation = r02.independent_tier1_revalidation(bundle)
    lower = render_lower(bundle, contract)
    composite = render_vector_composite(bundle, contract, top_path)
    caption_path = fig_root / f"python/generated/caption_{ARTIFACT_TAG}.md"
    caption_path.write_text(make_caption(contract, composite, top_path), encoding="utf-8")
    manifest = dict(bundle["manifest"])
    manifest.update(
        {
            "schema_version": R04_SCHEMA,
            "status": R04_STATUS,
            "target_body_width_in": TARGET_BODY_WIDTH_IN,
            "target_dpi": R04_DPI,
            "top_blender_layer": str(top_path),
            "shared_panel_geometry_contract": contract,
            "tier1_revalidation": revalidation,
            "lower_render": lower,
            "composite_render": composite,
            "caption": str(caption_path),
            "r04_processing_decisions": [
                "R04 is additive and preserves R03 outputs.",
                "Seed marker mapping is explicit: 123 circle, 234 square, 345 triangle.",
                "Foundation checkpoint variants remain separate from training seeds.",
                "FT-MultiT seed 123 MODEL_LOAD failure is named for all three paths and is not numeric-filled.",
                "Upper layer is a real-coordinate Blender raster; lower curves/text are vector in PDF/SVG.",
                "7.0-in body-width is a provisional Wiley full-width assumption and requires journal-class confirmation.",
            ],
        }
    )
    output_paths = [caption_path, fig_root / f"integration/panel_geometry_contract_{ARTIFACT_TAG}.json"]
    output_paths += [Path(lower["outputs"][key]) for key in ("pdf", "svg", "png")]
    output_paths += [Path(composite["outputs"][key]) for key in ("pdf", "svg", "png")]
    manifest["generated_file_hashes"] = {str(path): r01.sha256_file(path) for path in output_paths if path.exists()}
    composite_png = Path(composite["outputs"]["png"])
    proofs = render_body_width_proofs(composite_png)
    output_paths += [Path(path) for path in proofs["outputs"].values()]
    manifest["body_width_proof"] = proofs
    manifest_path = fig_root / f"figure_data_manifest_{ARTIFACT_TAG}.json"
    json_write(manifest_path, manifest)
    qa = make_qa(bundle, contract, lower, composite, revalidation, top_path, proofs)
    qa["manifest"] = str(manifest_path)
    qa["caption"] = str(caption_path)
    qa_path = fig_root / f"QA_REPORT_{ARTIFACT_TAG}.json"
    json_write(qa_path, qa)
    return {"status": R04_STATUS, "manifest": str(manifest_path), "qa_report": str(qa_path), "caption": str(caption_path), "lower": lower, "composite": composite, "contract": contract}


def main(argv: list[str] | None = None) -> int:
    global ARTIFACT_TAG, R04_LOWER_STEM, R04_COMPOSITE_STEM, R04_SCHEMA, R04_STATUS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=r01.default_repo_root())
    parser.add_argument("--top-path", type=Path, required=True)
    parser.add_argument("--artifact-tag", default="R04_body7in")
    parser.add_argument("--status", default=None)
    args = parser.parse_args(argv)
    ARTIFACT_TAG = args.artifact_tag
    R04_LOWER_STEM = f"fixed_path_lower_panels_{ARTIFACT_TAG}"
    R04_COMPOSITE_STEM = f"fixed_path_trajectory_composite_{ARTIFACT_TAG}"
    R04_SCHEMA = f"migrationbench_fixed_path_trajectory_figure_{ARTIFACT_TAG}_v1"
    if args.status:
        R04_STATUS = args.status
    result = build_r04(args.repo_root.resolve(), args.top_path.expanduser().resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
