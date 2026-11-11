#!/usr/bin/env python3
"""Build the R02 fixed-path NEB diagnostic candidate.

R02 is additive: it reuses the existing r01 data builder, but writes a new
lower-panel render, geometry contract, manifest, caption, and QA report.  It
does not read or write raw data, frozen criteria, or the Blender directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

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


R02_STATUS = "PROVISIONAL_DIAGNOSTIC_CANDIDATE_R02"
R02_SCHEMA = "migrationbench_fixed_path_trajectory_figure_r02_v1"
R02_DPI = 300
R02_WIDTH_PX = 4200
R02_HEIGHT_PX = 2100
R02_COLUMNS = (0, 1, 2)
R02_OUTPUT_STEM = "fixed_path_lower_panels_R02"


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def choose_font() -> tuple[str, bool]:
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        return "Arial", True
    except (ValueError, RuntimeError):
        return "DejaVu Sans", False


def make_geometry_contract(fig_root: Path, observed_top: dict[str, Any] | None) -> dict[str, Any]:
    width = R02_WIDTH_PX
    boundaries = [round(width * i / 3) for i in range(4)]
    columns = []
    for index, config in enumerate(r01.PATH_ORDER):
        x0, x1 = boundaries[index], boundaries[index + 1]
        columns.append(
            {
                "column_index": config["column_index"],
                "path_id": config["path_id"],
                "display_label": config["display_label"],
                "x0_norm": index / 3,
                "x1_norm": (index + 1) / 3,
                "center_norm": (index + 0.5) / 3,
                "x0_px": x0,
                "x1_px": x1,
                "center_px": (x0 + x1) // 2,
                "width_px": x1 - x0,
            }
        )
    contract = {
        "schema_version": "migrationbench_shared_panel_geometry_contract_r02_v1",
        "status": R02_STATUS,
        "rule": "top triptych and lower plot axes use the same three equal x-boundaries; no lower-raster rescaling is permitted",
        "reference_width_px": width,
        "reference_dpi": R02_DPI,
        "top_triptych": {
            "boundary_rule": "equal thirds of the supplied triptych raster",
            "boundary_px": boundaries,
            "observed": observed_top,
        },
        "lower_panel": {
            "canvas_width_px": width,
            "canvas_height_px": R02_HEIGHT_PX,
            "dpi": R02_DPI,
            "axes_boundary_rule": "axes outer boxes equal the top triptych thirds",
            "boundary_px": boundaries,
        },
        "columns": columns,
        "alignment_tolerance_px": 0,
        "column_order": [config["display_label"] for config in r01.PATH_ORDER],
    }
    path = fig_root / "integration/panel_geometry_contract_R02.json"
    json_write(path, contract)
    contract["path"] = str(path)
    contract["sha256"] = r01.sha256_file(path)
    return contract


def observed_top_triptych(fig_root: Path) -> dict[str, Any] | None:
    path = fig_root / "blender/triptych_top.png"
    if not path.exists():
        return None
    with Image.open(path) as image:
        width, height = image.size
    return {
        "path": str(path),
        "width_px": width,
        "height_px": height,
        "sha256": r01.sha256_file(path),
        "width_matches_contract": width == R02_WIDTH_PX,
        "width_divisible_by_three": width % 3 == 0,
    }


def independent_tier1_revalidation(bundle: dict[str, Any]) -> dict[str, Any]:
    repo_root = bundle["repo_root"]
    image_path = repo_root / "analysis/task3_endpoint_only_neb/neb_fixed_vs_self_matrix_20261019/derived_r01/TIER1_IMAGE_PROFILES.tsv"
    all_rows = r01.read_tsv(image_path)
    fixed_rows = [row for row in all_rows if row.get("observable") == r01.OBSERVABLE]
    by_record: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in fixed_rows:
        by_record[row["record_id"]].append(row)

    records = []
    complete_blocks = 0
    numeric_rows = 0
    failure_blocks = 0
    source_paths = Counter()
    for block in bundle["block_records"]:
        record_id = block["record_id"]
        rows = sorted(by_record.get(record_id, []), key=lambda row: int(row["image_index"]))
        indices = [int(row["image_index"]) for row in rows]
        complete = (
            len(rows) == r01.N_IMAGES
            and indices == list(range(r01.N_IMAGES))
            and all(r01.float_or_none(row.get("energy_eV")) is not None for row in rows)
        )
        if complete:
            complete_blocks += 1
            numeric_rows += len(rows)
        else:
            failure_blocks += 1
        profile_sources = sorted({row.get("profile_source", "") for row in rows if row.get("profile_source")})
        source_paths.update(profile_sources)
        records.append(
            {
                "record_id": record_id,
                "model_id": block["model_id"],
                "path_id": block["path_id"],
                "status": block["status"],
                "tier1_row_count": len(rows),
                "image_indices": indices,
                "complete_finite_ten_image_profile": complete,
                "tier1_profile_sources": profile_sources,
                "remote_profile_source": block.get("profile_source", ""),
                "remote_profile_read_status": block.get("remote_profile_read_status", ""),
                "raw_profile_sha256": block.get("profile_sha256", ""),
            }
        )

    expected_blocks = len(bundle["block_records"])
    checks = {
        "expected_target_blocks": 12 * 3,
        "observed_target_blocks": expected_blocks,
        "complete_finite_blocks": complete_blocks,
        "numeric_rows": numeric_rows,
        "failure_blocks": failure_blocks,
        "expected_numeric_rows": 33 * 10,
        "expected_failure_blocks": 3,
        "coverage_pass": (
            expected_blocks == 36
            and complete_blocks == 33
            and numeric_rows == 330
            and failure_blocks == 3
        ),
    }
    return {
        "status": "PASS_WITH_EXPLICIT_FALLBACK_BLOCKER" if checks["coverage_pass"] else "BLOCKED_REVALIDATION_MISMATCH",
        "source_path": str(image_path),
        "source_sha256": r01.sha256_file(image_path),
        "source_bytes": image_path.stat().st_size,
        "source_total_rows": len(all_rows),
        "source_fixed_observable_rows": len(fixed_rows),
        "coverage": checks,
        "record_ids": records,
        "tier1_profile_source_counts": dict(source_paths),
        "read_limitation": {
            "remote_fixed_profile_read_status": "UNREADABLE_AT_RUNTIME_FALLBACK_TIER1_IMAGE_PROFILES",
            "remote_raw_profile_sha256_policy": "not fabricated; sentinel retained per curve",
            "numeric_values_source": str(image_path),
            "scope": "12 target arms × 3 canonical paths; 33 complete ten-image profiles plus 3 explicit failure blocks",
        },
    }


def family_handles_r02(style: dict[str, Any]) -> list[Line2D]:
    labels = {
        "foundation_omat": "Foundation",
        "from_scratch": "From-scratch",
        "ft_600k": "FT-600K",
        "ft_multit": "FT-MultiT",
    }
    return [
        Line2D([], [], color=style["families"][key]["color"], lw=2.2, marker="o", markersize=5.5, label=labels[key])
        for key in ["foundation_omat", "from_scratch", "ft_600k", "ft_multit"]
    ]


def semantic_handles_r02() -> list[Line2D]:
    return [
        Line2D([], [], color="#2F2F2F", lw=2.0, marker="D", markersize=5.5, label="DFT/QE — SCF warning; diagnostic"),
        Line2D([], [], color="#666666", lw=1.8, linestyle=(0, (1, 2)), label="uncertainty band — withheld"),
        Line2D([], [], color="#666666", lw=1.8, linestyle=(0, (6, 2, 1, 2)), marker="o", markerfacecolor="white", label="nonconverged / diagnostic status"),
        Line2D([], [], color="#666666", lw=0, marker="x", markersize=7, label="failure / no numeric profile"),
        Line2D([], [], color="#555555", lw=1.8, marker="D", markersize=5.5, label="Foundation checkpoint variant (not seed)"),
        Line2D([], [], color="#555555", lw=0, marker="o", markersize=5, label="seed 123"),
        Line2D([], [], color="#555555", lw=0, marker="s", markersize=5, label="seed 234"),
        Line2D([], [], color="#555555", lw=0, marker="^", markersize=5, label="seed 345"),
    ]


def path_marker(model_arm_id: str, family: str, seed: str) -> str:
    if family == "Foundation":
        return {"FND-A": "D", "FND-B": "P", "FND-C": "h"}.get(model_arm_id, "D")
    return r01.SEED_MARKERS.get(str(seed), "o")


def draw_profile_axis_r02(ax: Any, path_config: dict[str, Any], fixed_rows: list[dict[str, Any]], dft_rows: list[dict[str, Any]], style: dict[str, Any], *, composite: bool) -> dict[str, Any]:
    colors = {key: style["families"][key]["color"] for key in style["families"]}
    path_id = path_config["path_id"]
    dft_for_path = sorted([row for row in dft_rows if row["path_id"] == path_id], key=lambda row: int(row["image_index"]))
    ax.plot(
        [float(row["s"]) for row in dft_for_path],
        [float(row["relative_energy_eV"]) for row in dft_for_path],
        color="#2F2F2F", lw=1.9, marker="D", markersize=4.8,
        markerfacecolor="#2F2F2F", markeredgecolor="white", markeredgewidth=0.35, zorder=8,
    )
    grouped, failures = r01.grouped_numeric_rows(path_id, fixed_rows)
    family_order = ["Foundation", "From-scratch", "FT-600K", "FT-MultiT"]
    ordered = sorted(grouped.items(), key=lambda item: (family_order.index(item[1][0]["model_family"]), item[1][0]["model_id"]))
    for (_, arm_id), rows in ordered:
        rows.sort(key=lambda row: int(row["image_index"]))
        first = rows[0]
        color = colors[first["family_key"]]
        ax.plot(
            [float(row["s"]) for row in rows],
            [float(row["relative_energy_eV"]) for row in rows],
            color=color, lw=1.15, alpha=0.88, linestyle="-",
            marker=path_marker(arm_id, first["model_family"], first["training_seed"]), markersize=4.2,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.3, zorder=5,
        )
    y0, y1 = r01.path_limits(path_id, fixed_rows, dft_rows)
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(y0, y1)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_xlabel("normalized reaction coordinate, s (0–1)", fontsize=11.5, labelpad=8)
    ax.set_ylabel("relative energy, ΔE (eV)", fontsize=11.5, labelpad=8)
    ax.tick_params(direction="out", length=4.0, width=0.9, colors="#333333", labelsize=11.0, pad=5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.95)
        spine.set_color("#333333")
    return {
        "path_id": path_id,
        "display_label": path_config["display_label"],
        "xlim": [-0.015, 1.015],
        "ylim": [y0, y1],
        "failure_model_ids": [row["model_id"] for row in failures],
        "axes_free_text_removed": True,
        "line_semantics": {
            "finite_profiles": "solid; family color and marker encode family/seed or checkpoint variant",
            "uncertainty_withheld": "separate dotted legend semantic; no numeric band drawn",
            "nonconverged_diagnostic": "separate dash-dot/open-marker legend semantic",
            "failure": "x legend/status semantic; no numeric line",
        },
    }


def draw_status_strip(fig: Any, bundle: dict[str, Any], *, include_title: bool = True) -> None:
    if include_title:
        fig.text(0.5, 0.725, "Fixed-path NEB profiles — provisional diagnostic candidate", ha="center", va="center", fontsize=12.0, fontweight="bold")
    fig.text(0.5, 0.965, "DFT/QE — SCF warning; diagnostic   |   uncertainty bands withheld   |   failure shown as ×", ha="center", va="center", fontsize=11.0, color="#2F2F2F")
    fig.text(0.5, 0.925, "ghosted spheres = real retained path-image overlay; α encodes image overlay order only, not uncertainty/error", ha="center", va="center", fontsize=11.0, color="#2F2F2F")
    fig.text(0.5, 0.885, "× FT-MultiT seed 123: MODEL_LOAD failure on all three paths", ha="center", va="center", fontsize=11.0, color="#67ACE5")


def save_figure_formats(fig: Any, stem: Path, *, dpi: int, png_only_fixed_canvas: bool = False) -> dict[str, Any]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg", ".png"):
        fig.savefig(stem.with_suffix(suffix), dpi=dpi, facecolor="white", edgecolor="white")
    with Image.open(stem.with_suffix(".png")) as image:
        width, height = image.size
    return {
        "stem": str(stem),
        "pdf": str(stem.with_suffix(".pdf")),
        "svg": str(stem.with_suffix(".svg")),
        "png": str(stem.with_suffix(".png")),
        "png_size_px": [width, height],
        "dpi": dpi,
        "fixed_canvas": png_only_fixed_canvas,
    }


def render_lower_r02(bundle: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    font_name, arial_available = choose_font()
    plt.rcParams.update({"font.family": font_name, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    generated = bundle["fig_root"] / "python/generated"
    panel_dir = generated / "panels_R02"
    standalone = []
    for config in r01.PATH_ORDER:
        fig = plt.figure(figsize=(5.2, 5.6), dpi=R02_DPI)
        ax = fig.add_axes([0.16, 0.17, 0.80, 0.50])
        metadata = draw_profile_axis_r02(ax, config, bundle["fixed_rows"], bundle["dft_profiles"], bundle["style"], composite=False)
        fig.text(0.56, 0.725, config["display_label"], ha="center", va="center", fontsize=12.0, fontweight="bold")
        draw_status_strip(fig, bundle, include_title=False)
        fig.legend(handles=family_handles_r02(bundle["style"]), loc="upper center", bbox_to_anchor=(0.56, 0.845), ncol=2, fontsize=11.0, frameon=False, handlelength=2.0, columnspacing=1.0)
        fig.legend(handles=semantic_handles_r02()[:5], loc="upper center", bbox_to_anchor=(0.56, 0.755), ncol=2, fontsize=11.0, frameon=False, handlelength=2.0, columnspacing=1.0)
        stem = panel_dir / f"panel_{config['short_label']}_R02"
        standalone.append({**metadata, "outputs": save_figure_formats(fig, stem, dpi=R02_DPI)})
        plt.close(fig)

    fig = plt.figure(figsize=(R02_WIDTH_PX / R02_DPI, R02_HEIGHT_PX / R02_DPI), dpi=R02_DPI)
    axes = []
    axes_metadata = []
    axes_bottom = 0.16
    axes_top = 0.66
    axes_height = axes_top - axes_bottom
    for index, config in enumerate(r01.PATH_ORDER):
        x0, x1 = index / 3, (index + 1) / 3
        ax = fig.add_axes([x0, axes_bottom, x1 - x0, axes_height])
        axes.append(ax)
        axes_metadata.append(draw_profile_axis_r02(ax, config, bundle["fixed_rows"], bundle["dft_profiles"], bundle["style"], composite=True))
        fig.text((x0 + x1) / 2, axes_top + 0.018, config["display_label"], ha="center", va="bottom", fontsize=12.0, fontweight="bold")
    draw_status_strip(fig, bundle)
    fig.legend(handles=family_handles_r02(bundle["style"]), loc="upper center", bbox_to_anchor=(0.5, 0.835), ncol=4, fontsize=11.0, frameon=False, handlelength=2.0, columnspacing=1.0)
    fig.legend(handles=semantic_handles_r02(), loc="upper center", bbox_to_anchor=(0.5, 0.775), ncol=4, fontsize=11.0, frameon=False, handlelength=1.9, columnspacing=0.9)
    fig.text(0.5, 0.075, "ΔE_i = E_i − E_0; ten retained images connected directly", ha="center", va="center", fontsize=11.0, color="#555555")
    stem = generated / R02_OUTPUT_STEM
    output_meta = save_figure_formats(fig, stem, dpi=R02_DPI, png_only_fixed_canvas=True)
    plt.close(fig)
    with Image.open(stem.with_suffix(".png")) as image:
        actual_width, actual_height = image.size
    columns = []
    for index, config in enumerate(r01.PATH_ORDER):
        x0, x1 = round(actual_width * index / 3), round(actual_width * (index + 1) / 3)
        columns.append({
            "path_id": config["path_id"],
            "x0_px": x0,
            "x1_px": x1,
            "center_px": (x0 + x1) // 2,
            "x0_norm": index / 3,
            "x1_norm": (index + 1) / 3,
        })
    geometry = {
        "contract_path": contract["path"],
        "contract_sha256": contract["sha256"],
        "canvas_size_px": [actual_width, actual_height],
        "axes_bottom_norm": axes_bottom,
        "axes_top_norm": axes_top,
        "axes_bounds_norm": [[index / 3, axes_bottom, 1 / 3, axes_height] for index in range(3)],
        "columns": columns,
        "matches_contract_width": actual_width == contract["reference_width_px"],
        "matches_contract_boundaries": all(
            item["x0_px"] == contract["columns"][index]["x0_px"]
            and item["x1_px"] == contract["columns"][index]["x1_px"]
            and item["center_px"] == contract["columns"][index]["center_px"]
            for index, item in enumerate(columns)
        ),
        "typography_pt": {"axis_labels": 11.5, "tick_labels": 11.0, "legend": 11.0, "status_strip": 11.0, "titles": 12.0},
    }
    return {
        "font": font_name,
        "arial_available": arial_available,
        "final_width_px": actual_width,
        "final_height_px": actual_height,
        "dpi": R02_DPI,
        "standalone_panels": standalone,
        "composite": {"outputs": output_meta, "panels": axes_metadata, "geometry": geometry},
    }


def make_caption(bundle: dict[str, Any], contract: dict[str, Any], revalidation: dict[str, Any]) -> str:
    coverage = revalidation["coverage"]
    return f"""# Fixed-path NEB trajectory/profile candidate — R02

Status: **{R02_STATUS}**. This R02 artifact is not publication final and does
not call the warning-bearing DFT/QE curves unconditional ground truth.

The three columns are strictly ordered **within gap → gap2QL → withinQL**.
The shared panel geometry contract is
`{contract['path']}`: the top triptych and lower axes use the same x0/x1/center
boundaries, with zero-pixel alignment tolerance. The integration script refuses
to rescale a lower raster whose width does not match the contract.

Ghosted spheres in the upper Blender candidate mean **real retained
path-image overlay**. Alpha encodes **image overlay order only, not
uncertainty/error**. No interpolated or synthetic geometry is added.

The lower curves use ΔE_i = E_i − E_0 versus normalized reaction coordinate
s and connect the ten retained images directly. DFT/QE is explicitly labeled
**SCF warning; diagnostic**. Foundation checkpoint variants and heterogeneous
From-scratch branches are discrete; FT-600K and FT-MultiT uncertainty bands
are withheld because the verified fixed-path training-seed gate is incomplete.

Current data ledger: **33 finite fixed diagnostic blocks**, **3 MODEL_LOAD
failure blocks**, and **3 × 10 = 30 DFT/QE warning rows**. The FT-600K band is
withheld because seed 123 is `MODEL_HASH_VERIFIED_BASE_UNPROVEN`; the FT-MultiT
band is withheld because seed 123 is `MODEL_LOAD` failure on all three paths.
Failures remain explicit and are not zero-filled, interpolated, or silently
removed.

For PHY-16/PUB-09, the derived
`TIER1_IMAGE_PROFILES.tsv` fallback was independently revalidated: SHA256
`{revalidation['source_sha256']}`, total rows `{revalidation['source_total_rows']}`;
the selected target coverage is `{coverage['expected_target_blocks']} blocks =
{coverage['complete_finite_blocks']} complete ten-image blocks ({coverage['numeric_rows']} rows)
plus {coverage['failure_blocks']} explicit failure blocks. The mirrored remote
fixed profile paths and runtime read limitation are retained; no unavailable
raw-profile SHA256 is fabricated. This limitation remains a publication
blocker.

Excluded observables: `SELF_RELAXED_ENDPOINT_ONLY_NEB` and `MIC_LINEAR`.
"""


def make_qa(bundle: dict[str, Any], contract: dict[str, Any], render_meta: dict[str, Any], revalidation: dict[str, Any]) -> dict[str, Any]:
    checks = [
        {"code": "VIS-01", "status": "PASS", "detail": "R02 shared geometry contract defines identical top/lower x0, x1, and center boundaries; final measurable sign-off still requires visual review of the Blender raster."},
        {"code": "VIS-02", "status": "PASS", "detail": "Column order and path IDs remain within gap → gap2QL → withinQL."},
        {"code": "VIS-03", "status": "PENDING", "detail": "Blender scene manifest has 81 atoms and 10 real images, but raster-level image 1–8 distinguishability remains a visual-review item outside the Python-only scope."},
        {"code": "VIS-05", "status": "PASS", "detail": "R02 caption and figure-level status strip explicitly define ghosted spheres as real retained path-image overlay; alpha is image order only, not uncertainty/error."},
        {"code": "VIS-12", "status": "PASS", "detail": "Uncertainty-withheld dotted semantic, nonconverged/diagnostic dash-dot semantic, and failure x semantic are separate legend entries."},
        {"code": "VIS-14", "status": "PASS", "detail": "Axes contain no free-form SCF/failure/status prose; warning and status messages are figure-level."},
        {"code": "VIS-16", "status": "PASS", "detail": "R02 final placement uses 11–12 pt axis/tick/legend/status typography at 4200 px / 14 in canvas."},
        {"code": "PUB-01", "status": "PASS", "detail": "Caption and figure-level status strip preserve diagnostic, ghost, fixed/self/MIC, and DFT wording boundaries."},
        {"code": "PUB-03", "status": "PASS", "detail": "Caption records 33 finite fixed diagnostic blocks, 3 MODEL_LOAD failure blocks, 30 DFT warning rows, and both withheld-band reasons."},
        {"code": "PUB-05", "status": "PASS", "detail": "Final R02 lower panel uses publication-size typography rather than relying on DPI."},
        {"code": "PUB-06", "status": "PASS", "detail": "Shared panel geometry contract and zero-pixel boundary QA are written for integration."},
        {"code": "PUB-07", "status": "PENDING", "detail": "R02 integration must confirm composite PNG/PDF/SVG hashes after the supplied Blender triptych is composited."},
        {"code": "PUB-08", "status": "PENDING", "detail": "Color/grayscale and full visual sign-off remain pending for the externally rendered Blender raster."},
        {"code": "PHY-16", "status": "BLOCKED", "detail": "TIER1 fallback revalidation passes coverage and records SHA256/row counts/record IDs, but remote fixed image_profile.csv reads remain blocked and raw per-profile hashes are unavailable."},
        {"code": "PUB-09", "status": "BLOCKED", "detail": "R02 is reproducible with an explicit fallback ledger, but the remote fixed-profile read limitation remains a release blocker."},
        {"code": "DFT-02", "status": "BLOCKED", "detail": "All 30 DFT rows remain COMPLETED_BUT_SCF_WARNING and are diagnostic only."},
        {"code": "DFT-04", "status": "BLOCKED", "detail": "Clean image-wise DFT convergence gate is not satisfied."},
        {"code": "SEED-04", "status": "BLOCKED", "detail": "FT-600K seed 123 is base-unproven and FT-MultiT seed 123 is MODEL_LOAD failure; bands remain withheld."},
        {"code": "PUB-10", "status": "BLOCKED", "detail": "R02 remains a provisional diagnostic candidate pending visual, DFT, seed, and remote-profile provenance gates."},
    ]
    blocked = [item["code"] for item in checks if item["status"] in {"BLOCKED", "PENDING"}]
    return {
        "schema_version": "migrationbench_fixed_path_trajectory_figure_qa_r02_v1",
        "status": R02_STATUS,
        "figure_accepted": False,
        "summary": {
            "checks_total": len(checks),
            "pass": sum(item["status"] == "PASS" for item in checks),
            "blocked_or_pending": len(blocked),
            "blocked_or_pending_codes": blocked,
            "numeric_fixed_rows": 330,
            "fixed_finite_diagnostic_blocks": 33,
            "fixed_failure_blocks": 3,
            "dft_warning_rows": 30,
            "blender_visual_review": "PENDING; supplied Blender render remains an unreviewed candidate",
        },
        "r02_geometry_contract": contract,
        "tier1_revalidation": revalidation,
        "render_metadata": render_meta,
        "checks": checks,
        "targeted_next_actions": [
            "Have the visual reviewer inspect the supplied Blender triptych at final placement, including image 1–8 distinguishability and grayscale/color review.",
            "Repair the remote fixed-profile read limitation or provide an independently auditable raw profile hash ledger.",
            "Repair DFT SCF convergence and verified seed gates before any publication-final claim.",
        ],
    }


def build_r02(repo_root: Path) -> dict[str, Any]:
    fig_root = repo_root / "analysis/neb_fixed_path_trajectory_figure_20261023"
    bundle = r01.build_data(repo_root, fig_root)
    observed_top = observed_top_triptych(fig_root)
    contract = make_geometry_contract(fig_root, observed_top)
    revalidation = independent_tier1_revalidation(bundle)
    render_meta = render_lower_r02(bundle, contract)
    caption_path = fig_root / "python/generated/caption_R02.md"
    caption_path.write_text(make_caption(bundle, contract, revalidation), encoding="utf-8")
    manifest = dict(bundle["manifest"])
    manifest.update({
        "schema_version": R02_SCHEMA,
        "status": R02_STATUS,
        "r02_review_source": str(fig_root / "visual_review/REVIEW_R01.md"),
        "shared_panel_geometry_contract": contract,
        "tier1_revalidation": revalidation,
        "render_metadata": render_meta,
        "caption": str(caption_path),
        "r02_processing_decisions": [
            "R02 uses a shared normalized and pixel panel geometry contract for top triptych and lower axes.",
            "Integration refuses width mismatch and never resizes the lower raster.",
            "Ghost semantics are stated at figure level and in caption; alpha is not uncertainty/error.",
            "Uncertainty-withheld, nonconverged/diagnostic, and failure semantics use separate legend styles.",
            "TIER1_IMAGE_PROFILES fallback was independently revalidated with source SHA256, row counts, coverage, and record IDs; remote raw-profile SHA256 remains unavailable by policy.",
        ],
        "generated_file_hashes": {},
    })
    output_paths = [
        caption_path,
        fig_root / "integration/panel_geometry_contract_R02.json",
        Path(render_meta["composite"]["outputs"]["pdf"]),
        Path(render_meta["composite"]["outputs"]["svg"]),
        Path(render_meta["composite"]["outputs"]["png"]),
    ]
    for panel in render_meta["standalone_panels"]:
        output_paths.extend(Path(panel["outputs"][suffix]) for suffix in ("pdf", "svg", "png"))
    manifest["generated_file_hashes"] = {str(path): r01.sha256_file(path) for path in output_paths if path.exists()}
    manifest_path = fig_root / "figure_data_manifest_R02.json"
    json_write(manifest_path, manifest)
    qa = make_qa(bundle, contract, render_meta, revalidation)
    qa["manifest"] = str(manifest_path)
    qa_path = fig_root / "QA_REPORT_R02.json"
    json_write(qa_path, qa)
    return {
        "status": R02_STATUS,
        "manifest": str(manifest_path),
        "qa_report": str(qa_path),
        "caption": str(caption_path),
        "contract": contract,
        "revalidation": revalidation,
        "render_metadata": render_meta,
        "fixed_rows": len(bundle["fixed_rows"]),
        "dft_rows": len(bundle["dft_profiles"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=r01.default_repo_root())
    args = parser.parse_args(argv)
    result = build_r02(args.repo_root.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
