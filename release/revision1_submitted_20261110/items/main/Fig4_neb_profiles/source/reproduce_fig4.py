#!/usr/bin/env python3
"""Fig. 4 a-f: Cr three-path fixed-path / self-relaxed NEB profiles (pre-Canva panel P06).

The main-text figure was composed in Canva from the script panel
CANVA-P06_neb_cr_r05_palette_updated.pdf (Canva added the row titles and the typo
"Self-relaxted"), so the reference is that panel, not the Canva export.  The copy actually
placed in Canva (full data, 2026-11-05) is kept in upstream/; the bundle's figures/ copy is a
stale render made while 8 Scratch-600K and 1 FT-600K self-relaxed run files were missing from
the bundle (those files were imported on 2026-11-10, see imports/IMPORT_MANIFEST.tsv).
The renderer reads every MLIP profile from the bundle's source archive and the DFT-NEB
reference from the committed QE neb.out files (src/migrationbench/dft_neb_reference.py).

Writes rerun_output/ and evidence/Fig4_a-f.json.
"""
from __future__ import annotations

import filecmp
import importlib.util
import os
import sys
from pathlib import Path

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import REPO, Evidence, image_difference, lint_script, rasterize, sha256  # noqa: E402

BUNDLE = "reproducibility/canva_palette_vector_bundle_20261105"
SOURCE = f"{BUNDLE}/data/source_archive/analysis/cr_interlayer_sliding_grouped_20261025"
RENDERER = f"{BUNDLE}/scripts/render_neb_palette_20261102.py"
LOADERS = [f"{SOURCE}/make_cr_three_path_2x3_R05.py", f"{SOURCE}/make_cr_three_path_2x3_R02.py",
           f"{SOURCE}/scripts/build_cr_grouped_r04.py",
           f"{BUNDLE}/data/source_archive/analysis/foundation_interlayer_sliding_10model_20261024/scripts/"
           "plot_interlayer_grouped_profile_atlas_r01.py",
           "src/migrationbench/dft_neb_reference.py"]
DFT_RAW = "data/raw_compact/cr81_dft_neb_reference_20261013/raw"
PROFILE_TABLES = [f"{SOURCE}/three_path_r02/source_tables_remote/FOUNDATION_SELF_IMAGE_PROFILES.tsv",
                  f"{SOURCE}/three_path_r02/source_tables_remote/LEGACY_TIER1_IMAGE_PROFILES.tsv",
                  f"{SOURCE}/three_path_r02/ft_self_remote/r02/FT_SELF_R02_COLLECT.tsv",
                  f"{SOURCE}/three_path_r02/ft_self_remote/r03/FT_SELF_R03_RETRY_COLLECT.tsv"]
ITEM = "release/revision1_submitted_20261110/items/main/Fig4_neb_profiles"
REFERENCE_PANEL = f"{ITEM}/upstream/CANVA-P06_neb_cr_r05_palette_updated_used_in_canva.pdf"
STALE_BUNDLE_PANEL = f"{BUNDLE}/figures/CANVA-P06_neb_cr_r05_palette_updated.pdf"
RUN_MANIFESTS = [f"{SOURCE}/three_path_r02/ft_self_remote", "endpoint_only_manifest.json",
                 f"{BUNDLE}/data/source_archive/analysis/task3_endpoint_only_neb/scratch_600k_migration_20261101/raw", "*_manifest.json"]
CURRENT_RELEASE = "paper/revision1/data_processed/current_release"
AUDITS = ["cr_three_path_plotted_profiles_R05.tsv", "cr_three_path_coverage_audit_R05.tsv"]
FINAL_ASSET = "paper/revision1/figures/main/fig4_NEB_new.pdf"


def load_renderer(out_dir: Path):
    os.environ["NEB_PALETTE_OUT"] = str(out_dir)
    os.environ["CR_FIGURE_OUT"] = str(out_dir)  # R02/R05 loaders create their OUT at import
    spec = importlib.util.spec_from_file_location("render_neb_palette", REPO / RENDERER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record_inputs(ev: Evidence) -> None:
    for path in [RENDERER, *LOADERS, *PROFILE_TABLES, FINAL_ASSET]:
        ev.input(path)
    for neb_out in sorted((REPO / DFT_RAW).glob("2L-Cu__neb-*/neb.out")):
        ev.input(neb_out.relative_to(REPO).as_posix())
    for root, pattern in zip(RUN_MANIFESTS[::2], RUN_MANIFESTS[1::2]):
        for manifest in sorted((REPO / root).rglob(pattern)):
            if "runs" in manifest.parts and manifest.parent.name.startswith("attempt_"):
                ev.input(manifest.relative_to(REPO).as_posix())
    for name in AUDITS:
        ev.input(f"{CURRENT_RELEASE}/{name}")


def main() -> int:
    ev = Evidence(__file__, "Fig4_a-f")
    record_inputs(ev)
    for script in [RENDERER, *LOADERS]:
        findings = lint_script(REPO / script)
        ev.check(f"hardcode lint {script}", not findings, findings or "no findings")

    renderer = load_renderer(ev.out_dir)
    stem = renderer.cr_three_path()
    ev.compare_pdf(stem.with_suffix(".pdf"), REFERENCE_PANEL)

    cr5 = renderer.CR5
    records = (cr5.load_current_foundation_self() + cr5.load_legacy_profiles()
               + cr5.load_current_ft_self() + cr5.load_current_scratch_600k())
    cr5.validate_scope(records)
    for produced in cr5.write_audits(records):
        reference = f"{CURRENT_RELEASE}/{produced.name}"
        ev.outputs[produced.relative_to(REPO).as_posix()] = sha256(produced)
        ev.check(f"{produced.name} equals {reference}", filecmp.cmp(produced, REPO / reference, shallow=False),
                 "byte-identical" if filecmp.cmp(produced, REPO / reference, shallow=False) else "differs")
    ev.input(STALE_BUNDLE_PANEL)
    stale = image_difference(rasterize(stem.with_suffix(".pdf"), 100), rasterize(REPO / STALE_BUNDLE_PANEL, 100))
    ev.check("known differences (not used for PASS)", True, {
        "final_asset": f"{FINAL_ASSET} adds Canva row titles ('Fixed path', 'Self-relaxted Activation Energy' "
                       "typo) and relayouts the legend; data panels are the P06 panel compared above",
        "stale_bundle_panel": {"path": STALE_BUNDLE_PANEL, "metrics_at_100dpi": stale,
                               "reason": "rendered before the missing Scratch-600K/FT-600K self-relaxed run "
                                         "files were in the bundle; not the panel used in Canva"}})
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
