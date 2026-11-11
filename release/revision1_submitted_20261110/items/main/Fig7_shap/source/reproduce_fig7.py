#!/usr/bin/env python3
"""Figure 7 (a-b): Top-20 SHAP strategy bars and transposed seed heatmap.

Re-plots the committed SHAP tables (shap_family_aggregate.csv, shap_seed_feature_importance.csv;
the SHAP campaign itself is not re-run) with the camera-ready renderer, output in rerun_output/,
and compares the composite and both panels with the committed pre-Canva PDFs whose sha256 are
recorded in fig7_render_manifest.json. Writes rerun_output/ and evidence/.
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

CAMERA = "release/revision1_camera_ready/reproducibility"
RENDERER = f"{CAMERA}/plot_shap_top20_bar_transposed_seed_heatmap_20261102.py"
CHAIN = [f"{CAMERA}/plot_shap_top20_bar_seed_heatmap_20261102.py", f"{CAMERA}/shap_feature_labels.py"]
ANALYSIS = "paper/revision1/data_processed/current_release"
TABLES = [f"{ANALYSIS}/shap_family_aggregate.csv", f"{ANALYSIS}/shap_seed_feature_importance.csv"]
RENDER_MANIFEST = f"{ANALYSIS}/fig7_render_manifest.json"
VARIANT = "top20_bar_transposed_seed_heatmap_20261102"
COMMITTED = f"analysis/shap_full_campaign_20261101/figures/variants_20261102/{VARIANT}"
PANELS = ("shap_top20_bar_transposed_seed_heatmap", "panel_a_top20_strategy_bars",
          "panel_b_top20_transposed_seed_heatmap")


def lint_renderers(ev: Evidence) -> None:
    scripts = [RENDERER, *(s for s in CHAIN if (REPO / s).exists())]
    findings = [f for script in scripts for f in lint_script(ev.input(script))]
    ev.check("hardcode lint of renderer chain", not findings, findings or scripts)


def check_references(ev: Evidence, manifest: dict) -> None:
    rows = {name: {"recorded": manifest["files"][f"{name}.pdf"]["sha256"],
                   "reference": sha256(REPO / COMMITTED / f"{name}.pdf")} for name in PANELS}
    renderer = {"recorded": manifest["renderer_script_sha256"], "current": sha256(REPO / RENDERER)}
    ev.check(f"reference panels sha256 equal {RENDER_MANIFEST} files", all(r["recorded"] == r["reference"]
             for r in rows.values()), rows)
    ev.check("renderer script sha256 equals the one recorded in the render manifest",
             renderer["recorded"] == renderer["current"], renderer)


def main() -> int:
    ev = Evidence(__file__, "Fig7_a-b")
    manifest = json.loads(ev.input(RENDER_MANIFEST).read_text())
    for table in TABLES:
        ev.input(table)
    lint_renderers(ev)
    check_references(ev, manifest)
    out_dir = ev.output("fig7")
    result = subprocess.run([sys.executable, str(REPO / RENDERER), "--analysis", str(REPO / ANALYSIS),
                             "--output", str(out_dir)], cwd=REPO / CAMERA, capture_output=True, text=True,
                            env={**os.environ, "MPLBACKEND": "Agg"})
    if ev.check("renderer exits 0", result.returncode == 0, result.stderr[-2000:] or "ok"):
        for name in PANELS:
            ev.compare_pdf(out_dir / VARIANT / f"{name}.pdf", f"{COMMITTED}/{name}.pdf")
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
