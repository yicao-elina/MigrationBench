#!/usr/bin/env python3
"""Fig. S4: Cr interlayer sliding, fixed vs self-relaxed (script panel CANVA-P09).

The submitted SI PDF (fig_NEB_interlayer-sliding.pdf) is the P09 SVG with hand-edited legend
labels (FT-600K / FT-MultiT / Scratch-600K / Scratch-MultiT, DFT-NEB) exported through cairo;
that edit step was never scripted. The data panels are therefore checked against the committed
P09 render, and the difference to the final asset is recorded as a known, non-data edit.

Writes rerun_output/ and evidence/FigS4.json.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import REPO, Evidence, image_difference, lint_script, rasterize  # noqa: E402

BUNDLE = "reproducibility/canva_palette_vector_bundle_20261105"
SOURCE = f"{BUNDLE}/data/source_archive/analysis/cr_interlayer_sliding_grouped_20261025"
RENDERER = f"{BUNDLE}/scripts/render_neb_palette_20261102.py"
LOADERS = [f"{SOURCE}/scripts/build_cr_grouped_r04.py", f"{SOURCE}/make_cr_three_path_2x3_R05.py",
           f"{SOURCE}/make_cr_three_path_2x3_R02.py",
           f"{BUNDLE}/data/source_archive/analysis/foundation_interlayer_sliding_10model_20261024/scripts/"
           "plot_interlayer_grouped_profile_atlas_r01.py"]
SOURCE_TABLES = [f"{SOURCE}/source_tables/{name}" for name in (
    "fixed_profiles_R04.csv", "foundation_2x2_IMAGE_PROFILES.tsv", "foundation_2x2_RESULT_MATRIX.tsv",
    "foundation_2x2_MODEL_REGISTRY.tsv", "foundation_2x2_QE_REFERENCE.tsv")]
RAW_CAMPAIGNS = [f"{SOURCE}/raw/rockfish/{name}" for name in (
    "scratch_lmax1_interlayer_sliding_20261025_r01", "scratch_lmax1_g4_self_20261025_r02",
    "ft_interlayer_sliding_20261025_r01")]
REFERENCE_PANEL = f"{BUNDLE}/figures/CANVA-P09_cr_interlayer_sliding_palette_updated.pdf"
FINAL_ASSET = "paper/revision1/figures/main/SI/fig_NEB_interlayer-sliding.pdf"


def load_renderer(out_dir: Path):
    os.environ["NEB_PALETTE_OUT"] = str(out_dir)
    os.environ["CR_FIGURE_OUT"] = str(out_dir)  # R02/R05 loaders create their OUT at import
    spec = importlib.util.spec_from_file_location("render_neb_palette", REPO / RENDERER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record_inputs(ev: Evidence) -> None:
    for path in [RENDERER, *LOADERS, *SOURCE_TABLES, FINAL_ASSET]:
        ev.input(path)
    for campaign in RAW_CAMPAIGNS:
        for path in sorted((REPO / campaign).rglob("*")):
            if path.is_file() and path.suffix in {".json", ".csv", ".tsv"}:
                ev.input(path.relative_to(REPO).as_posix())


def main() -> int:
    ev = Evidence(__file__, "FigS4")
    record_inputs(ev)
    for script in [RENDERER, *LOADERS]:
        findings = lint_script(REPO / script)
        ev.check(f"hardcode lint {script}", not findings, findings or "no findings")

    stem = load_renderer(ev.out_dir).cr_sliding()
    produced = stem.with_suffix(".pdf")
    ev.compare_pdf(produced, REFERENCE_PANEL)

    final = image_difference(rasterize(produced, 100), rasterize(REPO / FINAL_ASSET, 100))
    ev.check("known difference vs final asset (not used for PASS)", True, {
        "final_asset": FINAL_ASSET, "metrics_at_100dpi": final,
        "reason": "final PDF is the P09 SVG hand-edited (legend renamed FT-600K/FT-MultiT/Scratch-600K/"
                  "Scratch-MultiT, 'DFT reference' -> 'DFT-NEB') and exported via cairo; the edit is not "
                  "scripted, so the page size and legend differ while the plotted data are those of P09"})
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
