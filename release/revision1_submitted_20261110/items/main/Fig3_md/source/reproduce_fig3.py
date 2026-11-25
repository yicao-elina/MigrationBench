#!/usr/bin/env python3
"""Figure 3 (a-f): 1 ns MD six-panel stability/structure figure with the shared palette.

Runs the Canva-bundle palette renderer (which wraps the archived MD15 R06G renderer and reads
the md15_analysis_r03 / md15_global_rdf_r06 / md15_partial_rdf_r05 tables) with its output
redirected to rerun_output/, then compares the regenerated PDF with the committed pre-Canva
panel whose sha256 is recorded in fig3_render_manifest.json. Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import REPO, Evidence, lint_script, sha256  # noqa: E402

BUNDLE = "reproducibility/canva_palette_vector_bundle_20261105"
RENDERER = f"{BUNDLE}/scripts/render_md_six_panel_palette_20261102.py"
SOURCE = f"{BUNDLE}/data/source_archive/analysis/md_1ns_updated_publication_figure_20261025"
WRAPPED = f"{SOURCE}/render_md15_figure_R02.py"
DATA_DIRS = [f"{SOURCE}/{name}" for name in ("md15_analysis_r03", "md15_global_rdf_r06", "md15_partial_rdf_r05")]
DATA_FILES = [f"{SOURCE}/MD15_STATUS_AUDIT_R02.tsv", f"{SOURCE}/md15_inputs.tsv"]
RENDER_MANIFEST = "paper/revision1/data_processed/current_release/fig3_render_manifest.json"
STEM = "md_six_panel_palette_updated"
REFERENCE = f"docs/reproducibility/canva_figure_provenance/palette_recolor_20261102_r02/figures/CANVA-P05/{STEM}.pdf"
OUT_ENV = "MD_SIX_PANEL_OUT"


def lint_renderers(ev: Evidence) -> None:
    findings = [f for script in (RENDERER, WRAPPED) for f in lint_script(ev.input(script))]
    ev.check("hardcode lint of renderer chain", not findings, findings or [RENDERER, WRAPPED])


def record_data(ev: Evidence) -> None:
    files = sorted(p for d in DATA_DIRS for p in (REPO / d).rglob("*") if p.is_file())
    for path in files:
        ev.input(path.relative_to(REPO).as_posix())
    for name in DATA_FILES:
        ev.input(name)
    ev.check("archived MD15 analysis tables present", bool(files), f"{len(files)} files under {DATA_DIRS}")


def reference_is_canva_input(ev: Evidence) -> None:
    recorded = json.loads(ev.input(RENDER_MANIFEST).read_text())["outputs"]["pdf"]["sha256"]
    actual = sha256(REPO / REFERENCE)
    ev.check(f"reference panel sha256 equals {RENDER_MANIFEST} outputs.pdf.sha256",
             recorded == actual, {"recorded": recorded, "reference": actual})


def main() -> int:
    ev = Evidence(__file__, "Fig3_a-f")
    lint_renderers(ev)
    record_data(ev)
    reference_is_canva_input(ev)
    out_dir = ev.output("fig3")
    result = subprocess.run([sys.executable, str(REPO / RENDERER)], cwd=REPO / BUNDLE, capture_output=True,
                            text=True, env={**os.environ, "MPLBACKEND": "Agg", OUT_ENV: str(out_dir)})
    if ev.check("renderer exits 0", result.returncode == 0, result.stderr[-2000:] or "ok"):
        ev.compare_pdf(out_dir / f"{STEM}.pdf", REFERENCE)
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
