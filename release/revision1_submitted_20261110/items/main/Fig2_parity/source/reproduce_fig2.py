#!/usr/bin/env python3
"""Figure 2 (a-h): energy/force train-test parity hexbins with the shared palette.

Runs the palette-recolor wrapper (which drives build_slide3_energy_force_train_test_separated.py
on the packaged Rockfish parity NPZ/CSV files) with --out-root in rerun_output/, then compares
the regenerated Slide-3 panel with the committed pre-Canva panel whose sha256 is recorded in
fig2_render_manifest.json. Writes rerun_output/ and evidence/.
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

PROVENANCE = "docs/reproducibility/canva_figure_provenance"
R03 = f"{PROVENANCE}/palette_recolor_20261102_r03"
RENDERER = f"{R03}/render_variant04_and_slide3_palette_20261102.py"
CHAIN = [f"{PROVENANCE}/scripts/{name}.py" for name in (
    "build_slide3_energy_force_train_test_separated", "build_slide3_nature_density",
    "build_slide3_energy_train_test_cluster_centered_heatmap", "_figure02_latest_seed",
    "_figure02_latest_layout", "_figure02_latest_panels", "_figure02_latest_base",
    "_figure02_latest_aggregate", "repro_paths")]
DATA_DIR = f"{PROVENANCE}/data/figure02"
RENDER_MANIFEST = "paper/revision1/data_processed/current_release/fig2_render_manifest.json"
STEM = "slide3_energy_force_train_test_separated_palette_updated"
REFERENCE = f"{R03}/figures/{STEM}.pdf"
MANIFEST_KEY = "slide3_pdf"


def lint_renderers(ev: Evidence) -> None:
    findings = [f for script in [RENDERER, *CHAIN] for f in lint_script(ev.input(script))]
    ev.check("hardcode lint of renderer chain", not findings, findings or [RENDERER, *CHAIN])


def record_data(ev: Evidence) -> None:
    files = sorted(p for p in (REPO / DATA_DIR).rglob("*") if p.is_file())
    for path in files:
        ev.input(path.relative_to(REPO).as_posix())
    ev.check("packaged parity data present", bool(files), f"{len(files)} files under {DATA_DIR}")


def reference_is_canva_input(ev: Evidence) -> None:
    manifest = json.loads(ev.input(RENDER_MANIFEST).read_text())
    recorded = manifest["output_hashes"][MANIFEST_KEY]
    actual = sha256(REPO / REFERENCE)
    ev.check(f"reference panel sha256 equals {RENDER_MANIFEST} output_hashes.{MANIFEST_KEY}",
             recorded == actual, {"recorded": recorded, "reference": actual})


def main() -> int:
    ev = Evidence(__file__, "Fig2_a-h")
    lint_renderers(ev)
    record_data(ev)
    reference_is_canva_input(ev)
    out_root = ev.output("fig2")
    result = subprocess.run([sys.executable, str(REPO / RENDERER), "--out-root", str(out_root)],
                            cwd=REPO / R03, capture_output=True, text=True,
                            env={**os.environ, "MPLBACKEND": "Agg"})
    if ev.check("renderer exits 0", result.returncode == 0, result.stderr[-2000:] or "ok"):
        ev.compare_pdf(out_root / "figures" / f"{STEM}.pdf", REFERENCE)
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
