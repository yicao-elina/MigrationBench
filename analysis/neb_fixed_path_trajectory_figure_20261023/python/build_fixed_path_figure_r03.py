#!/usr/bin/env python3
"""Additive R03 lower-panel revision after the R02 visual review.

R03 keeps the R02 data and geometry contract intact, but uses a compact three
row figure-level header: status strip, family handles, and short semantic
handles.  It never writes raw data, frozen criteria, or the Blender directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import build_fixed_path_figure as r01
import build_fixed_path_figure_r02 as r02


R03_STATUS = "PROVISIONAL_DIAGNOSTIC_CANDIDATE_R03"
R03_SCHEMA = "migrationbench_fixed_path_trajectory_figure_r03_v1"
R03_DPI = 300
R03_WIDTH_PX = 4200
R03_HEIGHT_PX = 2100
R03_OUTPUT_STEM = "fixed_path_lower_panels_R03"


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def choose_font() -> tuple[str, bool]:
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        return "Arial", True
    except (ValueError, RuntimeError):
        return "DejaVu Sans", False


def make_geometry_contract(fig_root: Path) -> dict[str, Any]:
    r02_contract_path = fig_root / "integration/panel_geometry_contract_R02.json"
    if not r02_contract_path.exists():
        raise FileNotFoundError(f"R02 geometry contract is required before R03: {r02_contract_path}")
    contract = json.loads(r02_contract_path.read_text(encoding="utf-8"))
    contract["schema_version"] = "migrationbench_shared_panel_geometry_contract_r03_v1"
    contract["status"] = R03_STATUS
    contract["revision"] = "R03 additive layout revision; R02 contract preserved"
    path = fig_root / "integration/panel_geometry_contract_R03.json"
    json_write(path, contract)
    contract["path"] = str(path)
    contract["sha256"] = r01.sha256_file(path)
    return contract


def short_semantic_handles() -> list[Line2D]:
    return [
        Line2D([], [], color="#2F2F2F", lw=2.0, marker="D", markersize=5.0, label="DFT warning"),
        Line2D([], [], color="#666666", lw=1.8, linestyle=(0, (1, 2)), label="band withheld"),
        Line2D([], [], color="#666666", lw=1.8, linestyle=(0, (6, 2, 1, 2)), marker="o", markerfacecolor="white", label="diagnostic status"),
        Line2D([], [], color="#666666", lw=0, marker="x", markersize=7, label="failure"),
        Line2D([], [], color="#555555", lw=1.8, marker="D", markersize=5.0, label="checkpoint variant"),
        Line2D([], [], color="#555555", lw=0, marker="o", markersize=5, label="seeds: 123/234/345"),
    ]


def apply_x_tick_layout(ax: Any, panel_index: int | None = None) -> None:
    """Keep the three exact column boxes while preventing boundary-label collisions."""
    ticks = (0.0, 0.25, 0.5, 0.75, 1.0)
    ax.set_xticks(ticks)
    labels = ax.get_xticklabels()
    for value, label in zip(ticks, labels):
        if value == 0.0:
            label.set_ha("left")
        elif value == 1.0:
            label.set_ha("right")
        if panel_index is not None:
            # The outer endpoints are labelled once; all interior quarter points
            # remain visible in every column for independent panel reading.
            if panel_index > 0 and value == 0.0:
                label.set_visible(False)
            if panel_index < 2 and value == 1.0:
                label.set_visible(False)


def draw_axis_r03(
    ax: Any,
    config: dict[str, Any],
    bundle: dict[str, Any],
    panel_index: int | None = None,
) -> dict[str, Any]:
    metadata = r02.draw_profile_axis_r02(
        ax,
        config,
        bundle["fixed_rows"],
        bundle["dft_profiles"],
        bundle["style"],
        composite=True,
    )
    ax.set_xlabel("s (0–1)", fontsize=11.5, labelpad=8)
    ax.set_ylabel("")
    ax.tick_params(labelsize=11.0, pad=4)
    apply_x_tick_layout(ax, panel_index=panel_index)
    metadata["r03_axes_free_text"] = True
    metadata["short_axis_labels"] = True
    metadata["x_tick_boundary_policy"] = "outer endpoints only; interior quarter ticks retained"
    return metadata


def draw_r03_status(fig: Any) -> None:
    fig.text(
        0.5,
        0.965,
        "DFT/QE: SCF warning; diagnostic  |  ghosted spheres = real retained path-image overlay  |  α = image order only, not uncertainty/error  |  bands withheld  |  × failure",
        ha="center",
        va="center",
        fontsize=11.0,
        color="#2F2F2F",
    )


def save_formats(fig: Any, stem: Path, dpi: int) -> dict[str, Any]:
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
        "fixed_canvas": size == [R03_WIDTH_PX, R03_HEIGHT_PX],
    }


def render_r03(bundle: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    font_name, arial_available = choose_font()
    plt.rcParams.update({"font.family": font_name, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    generated = bundle["fig_root"] / "python/generated"
    standalone = []
    for config in r01.PATH_ORDER:
        fig = plt.figure(figsize=(5.2, 5.6), dpi=R03_DPI)
        ax = fig.add_axes([0.16, 0.18, 0.78, 0.48])
        metadata = draw_axis_r03(ax, config, bundle)
        fig.text(0.55, 0.695, config["display_label"], ha="center", va="center", fontsize=12.0, fontweight="bold")
        draw_r03_status(fig)
        fig.legend(handles=r02.family_handles_r02(bundle["style"]), loc="upper center", bbox_to_anchor=(0.55, 0.88), ncol=2, fontsize=11.0, frameon=False, handlelength=2.0, columnspacing=1.0)
        fig.legend(handles=short_semantic_handles(), loc="upper center", bbox_to_anchor=(0.55, 0.76), ncol=2, fontsize=11.0, frameon=False, handlelength=1.9, columnspacing=1.0)
        stem = generated / "panels_R03" / f"panel_{config['short_label']}_R03"
        standalone.append({**metadata, "outputs": save_formats(fig, stem, R03_DPI)})
        plt.close(fig)

    fig = plt.figure(figsize=(R03_WIDTH_PX / R03_DPI, R03_HEIGHT_PX / R03_DPI), dpi=R03_DPI)
    axes_bottom = 0.17
    axes_top = 0.64
    axes_height = axes_top - axes_bottom
    axes_metadata = []
    for index, config in enumerate(r01.PATH_ORDER):
        x0, x1 = index / 3, (index + 1) / 3
        ax = fig.add_axes([x0, axes_bottom, x1 - x0, axes_height])
        axes_metadata.append(draw_axis_r03(ax, config, bundle, panel_index=index))
        fig.text((x0 + x1) / 2, 0.685, config["display_label"], ha="center", va="center", fontsize=12.0, fontweight="bold")
        fig.text(x0 + 0.018, axes_bottom + axes_height / 2, "ΔE (eV)", ha="center", va="center", rotation=90, fontsize=11.5)

    draw_r03_status(fig)
    fig.legend(handles=r02.family_handles_r02(bundle["style"]), loc="upper center", bbox_to_anchor=(0.5, 0.895), ncol=4, fontsize=11.0, frameon=False, handlelength=2.0, columnspacing=1.0)
    fig.legend(handles=short_semantic_handles(), loc="upper center", bbox_to_anchor=(0.5, 0.815), ncol=6, fontsize=11.0, frameon=False, handlelength=1.8, columnspacing=0.85)
    fig.text(0.5, 0.055, "s = normalized reaction coordinate", ha="center", va="center", fontsize=11.0, color="#555555")
    stem = generated / R03_OUTPUT_STEM
    outputs = save_formats(fig, stem, R03_DPI)
    plt.close(fig)

    with Image.open(stem.with_suffix(".png")) as image:
        width, height = image.size
    columns = []
    for index, config in enumerate(r01.PATH_ORDER):
        x0, x1 = round(width * index / 3), round(width * (index + 1) / 3)
        columns.append({"path_id": config["path_id"], "x0_px": x0, "x1_px": x1, "center_px": (x0 + x1) // 2})
    geometry = {
        "contract_path": contract["path"],
        "contract_sha256": contract["sha256"],
        "canvas_size_px": [width, height],
        "axes_bounds_norm": [[index / 3, axes_bottom, 1 / 3, axes_height] for index in range(3)],
        "columns": columns,
        "matches_contract_width": width == int(contract["reference_width_px"]),
        "matches_contract_boundaries": all(
            row["x0_px"] == contract["columns"][index]["x0_px"]
            and row["x1_px"] == contract["columns"][index]["x1_px"]
            and row["center_px"] == contract["columns"][index]["center_px"]
            for index, row in enumerate(columns)
        ),
        "typography_pt": {"axis_labels": 11.5, "tick_labels": 11.0, "legend": 11.0, "status_strip": 11.0, "titles": 12.0},
        "header_layers": ["status_strip", "family_handles", "short_semantic_handles"],
        "x_tick_boundary_policy": "outer endpoints only; interior quarter ticks retained",
        "axes_top_clearance": "column titles at y=0.685; axes top at y=0.64; no header text inside axes",
    }
    return {
        "font": font_name,
        "arial_available": arial_available,
        "final_width_px": width,
        "final_height_px": height,
        "dpi": R03_DPI,
        "standalone_panels": standalone,
        "composite": {"outputs": outputs, "panels": axes_metadata, "geometry": geometry},
    }


def make_caption(bundle: dict[str, Any], contract: dict[str, Any], revalidation: dict[str, Any]) -> str:
    return r02.make_caption(bundle, contract, revalidation).replace("R02", "R03") + "\n\nR03 layout note: the figure-level header has three non-overlapping layers—one status strip, one four-family row, and one short semantic row. Full semantic definitions are retained here rather than placed inside axes.\n"


def build_r03(repo_root: Path) -> dict[str, Any]:
    fig_root = repo_root / "analysis/neb_fixed_path_trajectory_figure_20261023"
    bundle = r01.build_data(repo_root, fig_root)
    contract = make_geometry_contract(fig_root)
    revalidation = r02.independent_tier1_revalidation(bundle)
    render_meta = render_r03(bundle, contract)
    caption_path = fig_root / "python/generated/caption_R03.md"
    caption_path.write_text(make_caption(bundle, contract, revalidation), encoding="utf-8")

    manifest = dict(bundle["manifest"])
    manifest.update({
        "schema_version": R03_SCHEMA,
        "status": R03_STATUS,
        "r03_review_source": str(fig_root / "visual_review/REVIEW_R01.md"),
        "shared_panel_geometry_contract": contract,
        "tier1_revalidation": revalidation,
        "render_metadata": render_meta,
        "caption": str(caption_path),
        "r03_processing_decisions": [
            "R03 is additive and preserves all R02 outputs.",
            "Header is three non-overlapping figure-level layers; no long semantic text is placed inside axes.",
            "Top/lower geometry remains the shared three equal-column contract with zero pixel tolerance.",
            "Composite lower canvas remains exactly 4200×2100 px at 300 dpi.",
            "TIER1 fallback revalidation and remote read limitation remain explicit; no raw hash is fabricated.",
        ],
        "generated_file_hashes": {},
    })
    output_paths = [caption_path, fig_root / "integration/panel_geometry_contract_R03.json"]
    output_paths.extend(Path(render_meta["composite"]["outputs"][key]) for key in ("pdf", "svg", "png"))
    for panel in render_meta["standalone_panels"]:
        output_paths.extend(Path(panel["outputs"][key]) for key in ("pdf", "svg", "png"))
    manifest["generated_file_hashes"] = {str(path): r01.sha256_file(path) for path in output_paths if path.exists()}
    manifest_path = fig_root / "figure_data_manifest_R03.json"
    json_write(manifest_path, manifest)
    qa = r02.make_qa(bundle, contract, render_meta, revalidation)
    qa["schema_version"] = "migrationbench_fixed_path_trajectory_figure_qa_r03_v1"
    qa["status"] = R03_STATUS
    qa["manifest"] = str(manifest_path)
    for check in qa.get("checks", []):
        if isinstance(check.get("detail"), str):
            check["detail"] = check["detail"].replace("R02", "R03")
    qa["r03_layout"] = {
        "header_layers": ["one-line status strip", "four family handles", "short semantic handles"],
        "overlap_check": "pending integration plus visual reviewer inspection",
        "r02_preserved": True,
    }
    qa_path = fig_root / "QA_REPORT_R03.json"
    json_write(qa_path, qa)
    return {
        "status": R03_STATUS,
        "manifest": str(manifest_path),
        "qa_report": str(qa_path),
        "caption": str(caption_path),
        "lower_outputs": render_meta["composite"]["outputs"],
        "contract": contract,
        "revalidation_status": revalidation["status"],
        "coverage": revalidation["coverage"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=r01.default_repo_root())
    args = parser.parse_args(argv)
    result = build_r03(args.repo_root.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
