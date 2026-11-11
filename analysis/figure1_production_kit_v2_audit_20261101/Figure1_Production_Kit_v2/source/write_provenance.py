#!/usr/bin/env python3
"""Write a compact, source-backed provenance package for the Figure 1 preview."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROV = ROOT / "provenance"
PROV.mkdir(exist_ok=True)
ZIP_SOURCE = Path("/Users/alina/Downloads/Figure1_Production_Kit_v2.zip")
AIMD_SOURCE = Path(
    "/Users/alina/Project/migrationbench-paper-repro/analysis/"
    "task_conditioned_distribution_atlas_20261018/raw_cache/grouped_a2/valid.xyz"
)
STYLE_SOURCE = Path(
    "/Users/alina/.codex/skills/molecular-blender-render/scripts/cases/"
    "cr_migration_sb2te3/build_cr_migration_preview.py"
)


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_tsv(name: str, rows: list[dict[str, object]], fields: list[str]) -> None:
    with (PROV / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def git_snapshot() -> dict[str, object]:
    try:
        head = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            timeout=4,
            check=False,
        )
        status = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--short"],
            text=True,
            capture_output=True,
            timeout=4,
            check=False,
        )
        return {"head": head.stdout.strip() or None, "status_short": status.stdout.splitlines()[:200]}
    except Exception as exc:
        return {"head": None, "status_short": [], "error": str(exc)}


def main() -> None:
    render_manifest = json.loads((PROV / "actual_render_manifest.json").read_text())
    geometry_audit = json.loads((ROOT / "data" / "geometry_audit.json").read_text())
    asset_resolution = json.loads((ROOT / "data" / "asset_resolution.json").read_text())

    model_fields = [
        "canonical_model_id", "manuscript_label", "family", "checkpoint_path",
        "checkpoint_sha256", "training_seed", "status", "notes",
    ]
    write_tsv(
        "MODEL_REGISTRY.tsv",
        [
            {"canonical_model_id": "FM_prior", "manuscript_label": "Foundation prior", "family": "foundation", "checkpoint_path": "NOT_USED_IN_FIGURE1", "checkpoint_sha256": "", "training_seed": "", "status": "PROVENANCE_PARTIAL", "notes": "Panel b is a workflow schematic; no model output or checkpoint claim is plotted."},
            {"canonical_model_id": "S600", "manuscript_label": "S600", "family": "scratch", "checkpoint_path": "NOT_USED_IN_FIGURE1", "checkpoint_sha256": "", "training_seed": "123|234|345", "status": "PROVENANCE_PARTIAL", "notes": "Workflow cell only; seeds are encoded as vector symbols, not a checkpoint audit."},
            {"canonical_model_id": "SMT", "manuscript_label": "SMT", "family": "scratch", "checkpoint_path": "NOT_USED_IN_FIGURE1", "checkpoint_sha256": "", "training_seed": "123|234|345", "status": "PROVENANCE_PARTIAL", "notes": "Workflow cell only; seeds are encoded as vector symbols, not a checkpoint audit."},
            {"canonical_model_id": "FT600", "manuscript_label": "FT600", "family": "fine_tuning", "checkpoint_path": "NOT_USED_IN_FIGURE1", "checkpoint_sha256": "", "training_seed": "123|234|345", "status": "PROVENANCE_PARTIAL", "notes": "Workflow cell only; foundation-prior initialization is shown schematically."},
            {"canonical_model_id": "FTMT", "manuscript_label": "FTMT", "family": "fine_tuning", "checkpoint_path": "NOT_USED_IN_FIGURE1", "checkpoint_sha256": "", "training_seed": "123|234|345", "status": "PROVENANCE_PARTIAL", "notes": "Workflow cell only; Multi-T target adaptation is shown schematically."},
        ],
        model_fields,
    )

    claim_fields = [
        "claim_id", "panel_or_asset", "claim", "source_paths", "source_sha256", "status", "boundary_or_note",
    ]
    input_map = {
        key: value for key, value in render_manifest["sources"].items()
    }
    claim_rows = [
        {"claim_id": "F1-a", "panel_or_asset": "a / A01", "claim": "Two QLs and an in-gap Cr are rendered from supplied atom coordinates.", "source_paths": "inputs/within_gap__ideal2_cr_canonical_in_gap.extxyz; inputs/Sb2Te3.cif", "source_sha256": input_map["in_gap_extxyz"]["sha256"], "status": "VERIFIED_COORDINATE_RENDER", "boundary_or_note": "Host-host rods are display-neighbour cues; no bond-order claim."},
        {"claim_id": "F1-b", "panel_or_asset": "b / vector", "claim": "Target adaptation, four training combinations, seeds and independent zero-shot branch.", "source_paths": "source/build_figure1.py; Global_Visual_Tokens.json; Figure1_Caption_EN.txt", "source_sha256": sha256(ROOT / "source" / "build_figure1.py"), "status": "PROVENANCE_PARTIAL", "boundary_or_note": "No checkpoint, fit metric or model output is plotted in Figure 1."},
        {"claim_id": "F1-b-A07", "panel_or_asset": "b / A07", "claim": "Three target-ensemble structural snapshots use real 600 K grouped validation coordinates.", "source_paths": str(AIMD_SOURCE), "source_sha256": input_map["target_aimd_600K_grouped_valid"]["sha256"], "status": "VERIFIED_STRUCTURAL_SOURCE", "boundary_or_note": "Frames 0, 20 and 40 from an existing validation configuration pool; not a new MD run or independent replica."},
        {"claim_id": "F1-c-static", "panel_or_asset": "c / A02", "claim": "Local environment is drawn from a real two-Cr 600 K configuration.", "source_paths": str(AIMD_SOURCE), "source_sha256": input_map["target_aimd_600K_grouped_valid"]["sha256"], "status": "VERIFIED_STRUCTURAL_SOURCE", "boundary_or_note": "Pale green ring is a semantic selected-shell cue, not electron density or a coordination count."},
        {"claim_id": "F1-c-in-gap", "panel_or_asset": "c / A03", "claim": "In-gap scene is rendered from the supplied Cr-in-gap path.", "source_paths": "inputs/within_gap__ideal2_cr_canonical_in_gap.extxyz", "source_sha256": input_map["in_gap_extxyz"]["sha256"], "status": "VERIFIED_COORDINATE_RENDER", "boundary_or_note": "The arrow is vector schematic direction only; no fitted MEP is asserted."},
        {"claim_id": "F1-c-entry", "panel_or_asset": "c / A04", "claim": "Gap-to-QL entry scene is rendered from the supplied path.", "source_paths": "inputs/gap_to_ql__ideal1_cr_canonical_gap_to_ql.extxyz", "source_sha256": input_map["gap_to_ql_extxyz"]["sha256"], "status": "VERIFIED_COORDINATE_RENDER", "boundary_or_note": "The arrow is vector schematic direction only."},
        {"claim_id": "F1-c-within", "panel_or_asset": "c / A05", "claim": "Within-QL scene is rendered from the supplied path.", "source_paths": "inputs/within_ql__neb_3_sb2te3.extxyz", "source_sha256": input_map["within_ql_extxyz"]["sha256"], "status": "VERIFIED_COORDINATE_RENDER", "boundary_or_note": "The arrow is vector schematic direction only."},
        {"claim_id": "F1-c-slide", "panel_or_asset": "c / A06", "claim": "Collective sliding visual uses actual host coordinates plus one rigid upper-QL shift.", "source_paths": "inputs/within_gap__ideal2_cr_canonical_in_gap.extxyz; source/render_actual_assets.py", "source_sha256": input_map["in_gap_extxyz"]["sha256"], "status": "RECONSTRUCTION", "boundary_or_note": "No sampled sliding trajectory was found; this is not a kinetic or energy claim."},
        {"claim_id": "F1-d", "panel_or_asset": "d / vector", "claim": "Fixed-path and model-relaxed measurement operators and formulas are native vector layers.", "source_paths": "source/build_figure1.py; vector/Figure1_Vector_Overlay.svg", "source_sha256": sha256(ROOT / "source" / "build_figure1.py"), "status": "VERIFIED_VECTOR_DEFINITION", "boundary_or_note": "Schematic image paths; optimizer convergence and path validity remain separate checks."},
        {"claim_id": "F1-e", "panel_or_asset": "e / data", "claim": "Panel e contains 30 geometry rows computed from three supplied 10-frame EXTXYZ paths with consecutive-image MIC.", "source_paths": "inputs/within_gap__ideal2_cr_canonical_in_gap.extxyz; inputs/gap_to_ql__ideal1_cr_canonical_gap_to_ql.extxyz; inputs/within_ql__neb_3_sb2te3.extxyz; data/Cr_input_geometry.csv", "source_sha256": ";".join(input_map[key]["sha256"] for key in ("in_gap_extxyz", "gap_to_ql_extxyz", "within_ql_extxyz")), "status": "VERIFIED_GEOMETRY_CALCULATION", "boundary_or_note": "Normal is a×b; nominal entry origin is retained; no energy table or model-performance join."},
    ]
    write_tsv("CLAIM_EVIDENCE_MATRIX.tsv", claim_rows, claim_fields)

    failure_fields = ["item_id", "item", "state", "reason", "publication_use"]
    write_tsv(
        "FAILURE_EXCLUSION_REGISTRY.tsv",
        [
            {"item_id": "EX-A06", "item": "A06 collective sliding", "state": "RECONSTRUCTION", "reason": "No atomic sliding trajectory was found; upper QL is rigidly shifted +1.20 Å for a visual probe.", "publication_use": "Keep only with reconstruction label; do not present as sampled dynamics, MEP or barrier evidence."},
            {"item_id": "EX-B-MODEL", "item": "Panel b model cells", "state": "PROVENANCE_PARTIAL", "reason": "The workflow layer carries labels and seed symbols but no checkpoint paths or predictions.", "publication_use": "Workflow schematic only; not a model-comparison result."},
            {"item_id": "EX-A07-MD", "item": "A07 target snapshots", "state": "DESCRIPTIVE_STRUCTURAL_RENDER", "reason": "Existing 600 K grouped validation frames are reused as structural snapshots.", "publication_use": "Do not call them new MD, independent replicas or transport evidence."},
            {"item_id": "EX-CONCEPT", "item": "Original concept PNGs", "state": "REPLACED", "reason": "All seven production slots resolve to coordinate-based Blender renders in data/asset_resolution.json.", "publication_use": "Retain only as visual reference/history, not as coordinate evidence."},
            {"item_id": "EX-HD", "item": "High-resolution final render", "state": "PENDING_APPROVAL", "reason": "Only 1800 px-class preview assets have been rendered.", "publication_use": "Do not release HD RGB/RGBA output until the user explicitly approves the preview."},
        ],
        failure_fields,
    )

    env = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "blender": "/Applications/Blender.app/Contents/MacOS/Blender",
        "blender_version_from_render_manifest": render_manifest.get("blender_version"),
        "latex": "/Library/TeX/texbin/xelatex",
        "style_source": str(STYLE_SOURCE),
        "package_root": str(ROOT),
        "temporary_runtime_note": "Build used local Anaconda matplotlib plus a local PyMuPDF-only import path; runtime helper directories are not release assets.",
    }
    (PROV / "ENVIRONMENT_MANIFEST.json").write_text(json.dumps(env, indent=2) + "\n")

    release = {
        "schema_version": "1.0",
        "release_name": "Figure1_Production_Kit_v2_coordinate_preview_20261101",
        "status": "PREVIEW_ONLY",
        "approval_required_for_hd": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(ROOT),
        "user_supplied_zip": {"path": str(ZIP_SOURCE), "sha256": sha256(ZIP_SOURCE)},
        "render_style_source": {"path": str(STYLE_SOURCE), "sha256": sha256(STYLE_SOURCE)},
        "render_manifest": render_manifest,
        "geometry_audit": geometry_audit,
        "asset_resolution": asset_resolution,
        "commands": [
            "Blender 4.0.2 -b --python source/render_actual_assets.py -- --package-root <package> --aimd-source <valid.xyz>",
            "python source/compose_actual_assets.py",
            "python source/build_figure1.py",
            "python source/build_production_guide.py",
            "xelatex -interaction=nonstopmode -halt-on-error latex/Figure1_LaTeX_Assembly.tex",
        ],
        "outputs": {
            "main_png": rel(ROOT / "Figure1_Ideal_Visual.png"),
            "main_pdf": rel(ROOT / "Figure1_Ideal_Visual.pdf"),
            "main_svg": rel(ROOT / "Figure1_Ideal_Visual.svg"),
            "latex_pdf": rel(ROOT / "latex" / "Figure1_LaTeX_Assembly.pdf"),
            "guide_pdf": rel(ROOT / "Figure1_Production_Guide.pdf"),
            "geometry_csv": rel(ROOT / "data" / "Cr_input_geometry.csv"),
            "preview_manifest": rel(PROV / "actual_render_manifest.json"),
            "assembled_preview_zip": rel(ROOT.parent / "Figure1_Production_Kit_v2_coordinate_preview_20261101.zip"),
        },
        "output_sha256": {
            "main_png": sha256(ROOT / "Figure1_Ideal_Visual.png"),
            "main_pdf": sha256(ROOT / "Figure1_Ideal_Visual.pdf"),
            "main_svg": sha256(ROOT / "Figure1_Ideal_Visual.svg"),
            "latex_pdf": sha256(ROOT / "latex" / "Figure1_LaTeX_Assembly.pdf"),
            "guide_pdf": sha256(ROOT / "Figure1_Production_Guide.pdf"),
            "geometry_csv": sha256(ROOT / "data" / "Cr_input_geometry.csv"),
        },
        "git_snapshot": git_snapshot(),
    }
    (PROV / "FIGURE1_RELEASE_MANIFEST.json").write_text(json.dumps(release, indent=2) + "\n")

    gap_report = f"""# Figure 1 preview release and gap report

Generated: {release['created_utc']}

## Delivered in this preview

- A01–A07 resolve to coordinate-based Blender PNG assets using the frozen Cr/Sb2Te3 mother-scene style: scene-linear Te/Sb/Cr palette, 64×32 smooth spheres, low-elevation orthographic camera, four softboxes, Standard view and exposure 0.52.
- Panel e is rebuilt from 30 rows across three supplied 10-frame EXTXYZ paths with explicit consecutive-image MIC.
- Panels b and d remain native vector/text/formula layers; the LaTeX overlay compiles successfully.

## Scientific status boundaries

- `VERIFIED_COORDINATE_RENDER`: A01, A03, A04 and A05 use supplied EXTXYZ coordinates.
- `VERIFIED_STRUCTURAL_SOURCE`: A02 and A07 reuse real 600 K grouped validation configurations; they are descriptive snapshots, not a new MD campaign.
- `RECONSTRUCTION`: A06 applies a documented rigid +1.20 Å upper-QL display shift because no sampled atomic sliding trajectory was found.
- `PROVENANCE_PARTIAL`: Panel b labels workflow combinations and seeds, but this figure does not audit checkpoints or predictions.
- `VERIFIED_GEOMETRY_CALCULATION`: Panel e is the 30-point coordinate calculation; it is not joined to energy, force, OOD, or training-coverage claims.

## Pending gate

This is a reviewable preview only. No high-resolution RGB/RGBA Blender export has been made. Release HD assets only after explicit user approval of this visual preview.
"""
    (PROV / "GAP_REPORT.md").write_text(gap_report)
    print(json.dumps({"release_manifest": str(PROV / "FIGURE1_RELEASE_MANIFEST.json"), "status": "PREVIEW_ONLY"}, indent=2))


if __name__ == "__main__":
    main()
