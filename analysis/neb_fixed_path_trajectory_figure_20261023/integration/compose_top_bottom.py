#!/usr/bin/env python3
"""R02 strict composition of a Blender triptych and the lower NEB panel.

The top and lower rasters must have the same width under the shared geometry
contract.  This script never rescales the lower panel and never writes to the
Blender directory.  A successful composition remains an unreviewed Blender
candidate; visual acceptance is not automated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
FIG_ROOT = ROOT / "analysis/neb_fixed_path_trajectory_figure_20261023"
CONTRACT_PATH = FIG_ROOT / "integration/panel_geometry_contract_R02.json"
LOWER_PNG = FIG_ROOT / "python/generated/fixed_path_lower_panels_R02.png"
REQUIRED_TOP = ("within_gap.png", "gap2QL.png", "withinQL.png")
TOP_ALIASES = {
    "within_gap.png": ("within_gap.png", "cr_canonical_in_gap.png"),
    "gap2QL.png": ("gap2QL.png", "cr_canonical_gap_to_ql.png"),
    "withinQL.png": ("withinQL.png", "cr_canonical_within_ql.png"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_gap(top_dir: Path, missing: list[str], triptych: Path | None = None) -> int:
    payload = {
        "status": "BLOCKED_MISSING_BLENDER_PNG_INTERFACE_R02",
        "top_dir": str(top_dir),
        "missing": missing,
        "required": list(REQUIRED_TOP),
        "accepted_triptych_name": "triptych_top.png",
        "triptych_argument": str(triptych) if triptych else None,
        "lower_panel": str(LOWER_PNG),
        "lower_panel_available": LOWER_PNG.exists(),
        "next_action": "Provide a real triptych_top.png or the three real column PNGs, then rerun without changing the geometry contract.",
    }
    path = FIG_ROOT / "integration/INTEGRATION_GAP_R02.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 2


def load_contract() -> dict[str, object]:
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(f"missing shared panel geometry contract: {CONTRACT_PATH}")
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def resolve_top_source(top_dir: Path | None, triptych_top: Path | None) -> dict[str, object] | None:
    if triptych_top and triptych_top.exists():
        return {"mode": "triptych", "path": triptych_top}
    if top_dir:
        default_triptych = top_dir / "triptych_top.png"
        if default_triptych.exists():
            return {"mode": "triptych", "path": default_triptych}
        paths = []
        for required in REQUIRED_TOP:
            chosen = next((top_dir / alias for alias in TOP_ALIASES[required] if (top_dir / alias).exists()), None)
            if chosen is None:
                return None
            paths.append(chosen)
        return {"mode": "three_columns", "paths": paths}
    return None


def load_top_canvas(top_source: dict[str, object], contract: dict[str, object]) -> tuple[Image.Image, list[Path], dict[str, object]]:
    expected_width = int(contract["reference_width_px"])
    if top_source["mode"] == "triptych":
        triptych_path = Path(top_source["path"])
        image = Image.open(triptych_path).convert("RGB")
        if image.width != expected_width:
            raise ValueError(f"triptych width {image.width} != contract width {expected_width}; refusing rescale")
        if image.width % 3:
            raise ValueError(f"triptych width {image.width} is not divisible by three")
        return image, [triptych_path], {"mode": "triptych", "width_px": image.width, "height_px": image.height}

    paths = [Path(path) for path in top_source["paths"]]
    images = [Image.open(path).convert("RGB") for path in paths]
    expected_column_width = expected_width // 3
    if expected_width % 3:
        raise ValueError("contract width is not divisible by three")
    if any(image.size != (expected_column_width, images[0].height) for image in images):
        raise ValueError("three-column Blender PNGs must have identical contract dimensions; refusing rescale")
    canvas = Image.new("RGB", (expected_width, images[0].height), "white")
    for index, image in enumerate(images):
        canvas.paste(image, (index * expected_column_width, 0))
    return canvas, paths, {"mode": "three_columns", "width_px": canvas.width, "height_px": canvas.height}


def geometry_qa(contract: dict[str, object], top_width: int, lower_width: int) -> dict[str, object]:
    expected_width = int(contract["reference_width_px"])
    expected_columns = contract["columns"]
    top_boundaries = [round(top_width * index / 3) for index in range(4)]
    lower_boundaries = [round(lower_width * index / 3) for index in range(4)]
    contract_boundaries = [int(item) for item in contract["top_triptych"]["boundary_px"]]
    per_column = []
    for index, column in enumerate(expected_columns):
        per_column.append(
            {
                "path_id": column["path_id"],
                "top_x0_px": top_boundaries[index],
                "top_x1_px": top_boundaries[index + 1],
                "lower_x0_px": lower_boundaries[index],
                "lower_x1_px": lower_boundaries[index + 1],
                "contract_x0_px": column["x0_px"],
                "contract_x1_px": column["x1_px"],
                "top_lower_x0_error_px": lower_boundaries[index] - top_boundaries[index],
                "top_lower_x1_error_px": lower_boundaries[index + 1] - top_boundaries[index + 1],
                "top_contract_x0_error_px": top_boundaries[index] - int(column["x0_px"]),
                "top_contract_x1_error_px": top_boundaries[index + 1] - int(column["x1_px"]),
                "center_error_px": ((lower_boundaries[index] + lower_boundaries[index + 1]) // 2) - int(column["center_px"]),
            }
        )
    errors = [
        value
        for row in per_column
        for key, value in row.items()
        if key.endswith("error_px")
    ]
    passed = (
        top_width == expected_width
        and lower_width == expected_width
        and top_boundaries == contract_boundaries
        and lower_boundaries == contract_boundaries
        and max(abs(value) for value in errors) == 0
    )
    return {
        "status": "PASS" if passed else "BLOCKED_GEOMETRY_MISMATCH",
        "tolerance_px": int(contract["alignment_tolerance_px"]),
        "expected_width_px": expected_width,
        "top_width_px": top_width,
        "lower_width_px": lower_width,
        "top_boundaries_px": top_boundaries,
        "lower_boundaries_px": lower_boundaries,
        "contract_boundaries_px": contract_boundaries,
        "per_column": per_column,
        "max_abs_error_px": max(abs(value) for value in errors),
        "passed": passed,
    }


def save_composite_formats(canvas: Image.Image, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "fixed_path_trajectory_composite_R02"
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    svg = stem.with_suffix(".svg")
    canvas.save(png, dpi=(300, 300))
    width, height = canvas.size
    fig = plt.figure(figsize=(width / 300, height / 300), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.asarray(canvas), interpolation="none", origin="upper", aspect="auto")
    ax.set_axis_off()
    fig.savefig(pdf, dpi=300, facecolor="white", edgecolor="white")
    fig.savefig(svg, dpi=300, facecolor="white", edgecolor="white")
    plt.close(fig)
    with Image.open(png) as image:
        actual_size = list(image.size)
    return {
        "png": str(png),
        "pdf": str(pdf),
        "svg": str(svg),
        "png_size_px": actual_size,
        "sha256": {str(path): sha256(path) for path in (png, pdf, svg)},
        "formats_complete": all(path.exists() for path in (png, pdf, svg)),
        "natural_pixel_composition": True,
    }


def update_r02_qa(report: dict[str, object]) -> None:
    qa_path = FIG_ROOT / "QA_REPORT_R02.json"
    manifest_path = FIG_ROOT / "figure_data_manifest_R02.json"
    if qa_path.exists():
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        qa["integration_report"] = report
        for check in qa.get("checks", []):
            if check["code"] in {"VIS-01", "PUB-06"}:
                check["status"] = "PASS" if report["geometry_qa"]["passed"] else "BLOCKED"
                check["detail"] = "R02 top/lower shared geometry QA has zero-pixel error for every column; visual Blender review remains required." if report["geometry_qa"]["passed"] else "R02 strict geometry QA failed; no resized composite was accepted."
            if check["code"] == "PUB-07":
                check["status"] = "PASS" if report["output_formats"]["formats_complete"] else "BLOCKED"
                check["detail"] = "Composite PNG, PDF, and SVG exist with recorded SHA256 hashes." if report["output_formats"]["formats_complete"] else "Composite format set is incomplete."
        blocked = [item["code"] for item in qa.get("checks", []) if item["status"] in {"BLOCKED", "PENDING"}]
        qa["summary"]["pass"] = sum(item["status"] == "PASS" for item in qa.get("checks", []))
        qa["summary"]["blocked_or_pending"] = len(blocked)
        qa["summary"]["blocked_or_pending_codes"] = blocked
        qa["figure_accepted"] = False
        qa_path.write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["integration_report"] = report
        for path in report["output_formats"]["sha256"]:
            manifest.setdefault("generated_file_hashes", {})[path] = report["output_formats"]["sha256"][path]
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def compose(top_source: dict[str, object], output_dir: Path, contract: dict[str, object]) -> dict[str, object]:
    top_canvas, top_paths, top_meta = load_top_canvas(top_source, contract)
    lower = Image.open(LOWER_PNG).convert("RGB")
    qa = geometry_qa(contract, top_canvas.width, lower.width)
    if not qa["passed"]:
        report = {
            "status": "BLOCKED_GEOMETRY_MISMATCH_R02",
            "visual_review_required": True,
            "top_source_mode": top_source["mode"],
            "geometry_qa": qa,
            "top_pngs": [str(path) for path in top_paths],
            "lower_png": str(LOWER_PNG),
        }
        report_path = FIG_ROOT / "integration/INTEGRATION_REPORT_R02.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        update_r02_qa(report)
        print(json.dumps(report, indent=2))
        return report

    canvas = Image.new("RGB", (top_canvas.width, top_canvas.height + lower.height), "white")
    canvas.paste(top_canvas, (0, 0))
    canvas.paste(lower, (0, top_canvas.height))
    output_formats = save_composite_formats(canvas, output_dir)
    report = {
        "status": "PROVISIONAL_UNREVIEWED_BLENDER_CANDIDATE_R02",
        "visual_review_required": True,
        "top_source_mode": top_source["mode"],
        "top_meta": top_meta,
        "top_pngs": [str(path) for path in top_paths],
        "lower_png": str(LOWER_PNG),
        "output_formats": output_formats,
        "geometry_contract": str(CONTRACT_PATH),
        "geometry_contract_sha256": sha256(CONTRACT_PATH),
        "geometry_qa": qa,
        "lower_rescaled": False,
    }
    report_path = FIG_ROOT / "integration/INTEGRATION_REPORT_R02.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    update_r02_qa(report)
    print(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    global LOWER_PNG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-dir", type=Path, required=False)
    parser.add_argument("--triptych-top", type=Path, required=False)
    parser.add_argument("--lower-png", type=Path, default=LOWER_PNG)
    parser.add_argument("--output-dir", type=Path, default=FIG_ROOT / "integration/generated_R02")
    args = parser.parse_args(argv)
    LOWER_PNG = args.lower_png.expanduser().resolve()
    top_dir = args.top_dir.expanduser().resolve() if args.top_dir else None
    triptych_top = args.triptych_top.expanduser().resolve() if args.triptych_top else None
    if not LOWER_PNG.exists():
        return write_gap(top_dir or Path("."), [str(LOWER_PNG)], triptych_top)
    top_source = resolve_top_source(top_dir, triptych_top)
    if top_source is None:
        missing = []
        if top_dir:
            missing = [str(top_dir / alias[0]) for alias in TOP_ALIASES.values()]
            missing.append(str(top_dir / "triptych_top.png"))
        elif triptych_top:
            missing = [str(triptych_top)]
        else:
            missing = ["--top-dir or --triptych-top was not supplied"]
        return write_gap(top_dir or Path("."), missing, triptych_top)
    try:
        contract = load_contract()
        report = compose(top_source, args.output_dir.resolve(), contract)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        report = {"status": "BLOCKED_R02_COMPOSITION_ERROR", "error": str(error), "visual_review_required": True}
        path = FIG_ROOT / "integration/INTEGRATION_REPORT_R02.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 3
    return 0 if report["status"] == "PROVISIONAL_UNREVIEWED_BLENDER_CANDIDATE_R02" else 3


if __name__ == "__main__":
    raise SystemExit(main())
