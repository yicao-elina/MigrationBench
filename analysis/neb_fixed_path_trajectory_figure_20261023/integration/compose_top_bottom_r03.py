#!/usr/bin/env python3
"""R03 strict composition of Blender triptych PNGs and the R03 lower panel.

This additive wrapper accepts either a rerendered ``triptych_top.png`` or
three equal-sized column PNGs.  It enforces the shared R03 geometry contract,
never rescales either input, and leaves the Blender directory untouched.  A
successful run is still an unreviewed Blender candidate; this script never
sets visual acceptance automatically.
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

import compose_top_bottom as strict


ROOT = Path(__file__).resolve().parents[3]
FIG_ROOT = ROOT / "analysis/neb_fixed_path_trajectory_figure_20261023"
CONTRACT_PATH = FIG_ROOT / "integration/panel_geometry_contract_R03.json"
LOWER_PNG = FIG_ROOT / "python/generated/fixed_path_lower_panels_R03.png"
REQUIRED_TOP = strict.REQUIRED_TOP
TOP_ALIASES = strict.TOP_ALIASES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_gap(top_dir: Path, missing: list[str], triptych: Path | None = None) -> dict[str, object]:
    payload = {
        "status": "BLOCKED_MISSING_BLENDER_PNG_INTERFACE_R03",
        "visual_review_required": True,
        "top_dir": str(top_dir),
        "missing": missing,
        "required_columns": list(REQUIRED_TOP),
        "accepted_triptych_name": "triptych_top.png",
        "triptych_argument": str(triptych) if triptych else None,
        "lower_panel": str(LOWER_PNG),
        "lower_panel_available": LOWER_PNG.exists(),
        "next_action": "Provide a rerendered triptych_top.png or three equal-sized column PNGs and rerun this R03 wrapper.",
    }
    write_json(FIG_ROOT / "integration/INTEGRATION_GAP_R03.json", payload)
    return payload


def load_contract() -> dict[str, object]:
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(f"missing R03 shared panel geometry contract: {CONTRACT_PATH}")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract["path"] = str(CONTRACT_PATH)
    contract["sha256"] = sha256(CONTRACT_PATH)
    return contract


def save_composite_formats(canvas: Image.Image, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "fixed_path_trajectory_composite_R03"
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
    paths = (png, pdf, svg)
    return {
        "png": str(png),
        "pdf": str(pdf),
        "svg": str(svg),
        "png_size_px": actual_size,
        "sha256": {str(path): sha256(path) for path in paths},
        "formats_complete": all(path.exists() for path in paths),
        "natural_pixel_composition": True,
    }


def update_r03_metadata(report: dict[str, object]) -> None:
    qa_path = FIG_ROOT / "QA_REPORT_R03.json"
    manifest_path = FIG_ROOT / "figure_data_manifest_R03.json"
    if qa_path.exists():
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        qa["integration_report"] = report
        geometry_passed = bool(report.get("geometry_qa", {}).get("passed", False))
        formats_passed = bool(report.get("output_formats", {}).get("formats_complete", False))
        for check in qa.get("checks", []):
            if check["code"] in {"VIS-01", "PUB-06"}:
                check["status"] = "PASS" if geometry_passed else "BLOCKED"
                check["detail"] = (
                    "R03 top/lower shared geometry QA has zero-pixel error for every column; visual Blender review remains required."
                    if geometry_passed
                    else "R03 strict geometry QA failed; no resized composite was accepted."
                )
            if check["code"] == "PUB-07":
                check["status"] = "PASS" if formats_passed else "BLOCKED"
                check["detail"] = (
                    "R03 composite PNG, PDF, and SVG exist with recorded SHA256 hashes."
                    if formats_passed
                    else "R03 composite format set is incomplete."
                )
        blocked = [item["code"] for item in qa.get("checks", []) if item["status"] in {"BLOCKED", "PENDING"}]
        qa.setdefault("summary", {})["pass"] = sum(item["status"] == "PASS" for item in qa.get("checks", []))
        qa["summary"]["blocked_or_pending"] = len(blocked)
        qa["summary"]["blocked_or_pending_codes"] = blocked
        qa["figure_accepted"] = False
        qa["status"] = "PROVISIONAL_DIAGNOSTIC_CANDIDATE_R03"
        write_json(qa_path, qa)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["integration_report"] = report
        for path, digest in report.get("output_formats", {}).get("sha256", {}).items():
            manifest.setdefault("generated_file_hashes", {})[path] = digest
        write_json(manifest_path, manifest)


def compose(top_source: dict[str, object], output_dir: Path, contract: dict[str, object], lower_png: Path) -> dict[str, object]:
    top_canvas, top_paths, top_meta = strict.load_top_canvas(top_source, contract)
    lower = Image.open(lower_png).convert("RGB")
    geometry = strict.geometry_qa(contract, top_canvas.width, lower.width)
    if not geometry["passed"]:
        report = {
            "status": "BLOCKED_GEOMETRY_MISMATCH_R03",
            "visual_review_required": True,
            "top_source_mode": top_source["mode"],
            "geometry_qa": geometry,
            "top_pngs": [str(path) for path in top_paths],
            "lower_png": str(lower_png),
            "lower_rescaled": False,
        }
        write_json(FIG_ROOT / "integration/INTEGRATION_REPORT_R03.json", report)
        update_r03_metadata(report)
        return report

    canvas = Image.new("RGB", (top_canvas.width, top_canvas.height + lower.height), "white")
    canvas.paste(top_canvas, (0, 0))
    canvas.paste(lower, (0, top_canvas.height))
    output_formats = save_composite_formats(canvas, output_dir)
    report = {
        "status": "PROVISIONAL_UNREVIEWED_BLENDER_CANDIDATE_R03",
        "visual_review_required": True,
        "figure_accepted": False,
        "top_source_mode": top_source["mode"],
        "top_meta": top_meta,
        "top_pngs": [str(path) for path in top_paths],
        "lower_png": str(lower_png),
        "output_formats": output_formats,
        "geometry_contract": str(CONTRACT_PATH),
        "geometry_contract_sha256": sha256(CONTRACT_PATH),
        "geometry_qa": geometry,
        "lower_rescaled": False,
        "blender_candidate_review": "UNPASSED_UNTIL_VISUAL_REVIEW",
    }
    write_json(FIG_ROOT / "integration/INTEGRATION_REPORT_R03.json", report)
    update_r03_metadata(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-dir", type=Path)
    parser.add_argument("--triptych-top", type=Path)
    parser.add_argument("--lower-png", type=Path, default=LOWER_PNG)
    parser.add_argument("--output-dir", type=Path, default=FIG_ROOT / "integration/generated_R03")
    args = parser.parse_args(argv)

    lower_png = args.lower_png.expanduser().resolve()
    top_dir = args.top_dir.expanduser().resolve() if args.top_dir else None
    triptych_top = args.triptych_top.expanduser().resolve() if args.triptych_top else None
    if not lower_png.exists():
        report = write_gap(top_dir or Path("."), [str(lower_png)], triptych_top)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2
    top_source = strict.resolve_top_source(top_dir, triptych_top)
    if top_source is None:
        missing = []
        if top_dir:
            missing = [str(top_dir / "triptych_top.png")]
            missing.extend(str(top_dir / alias) for aliases in TOP_ALIASES.values() for alias in aliases)
        elif triptych_top:
            missing = [str(triptych_top)]
        else:
            missing = ["--top-dir or --triptych-top was not supplied"]
        report = write_gap(top_dir or Path("."), missing, triptych_top)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2
    try:
        contract = load_contract()
        report = compose(top_source, args.output_dir.expanduser().resolve(), contract, lower_png)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        report = {
            "status": "BLOCKED_R03_COMPOSITION_ERROR",
            "visual_review_required": True,
            "figure_accepted": False,
            "error": str(error),
            "lower_rescaled": False,
        }
        write_json(FIG_ROOT / "integration/INTEGRATION_REPORT_R03.json", report)
        update_r03_metadata(report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 3
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PROVISIONAL_UNREVIEWED_BLENDER_CANDIDATE_R03" else 3


if __name__ == "__main__":
    raise SystemExit(main())
