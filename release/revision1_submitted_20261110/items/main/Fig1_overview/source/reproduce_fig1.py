#!/usr/bin/env python3
"""Figure 1: regenerate the SOAP-PCA panel (b) and verify the Blender render set (a, c).

Panel b is re-rendered from the packaged task-map CSV by the original renderer with its
output redirected into rerun_output/, then compared with the committed pre-Canva panel.
Panels a and c are Blender renders; Blender is not run. Instead every render and .blend
listed in RENDER_SET_MANIFEST.json is hashed and compared with the recorded sha256 (files
absent from git are accepted only when EXTERNAL_STORE_MANIFEST.tsv records the same hash).
Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import REPO, Evidence, lint_script, sha256  # noqa: E402

PROVENANCE = "docs/reproducibility/canva_figure_provenance"
RENDERER = f"{PROVENANCE}/scripts/build_figure01a_clean_panel.py"
HELPERS = [f"{PROVENANCE}/scripts/build_atomic_migration_figures.py", f"{PROVENANCE}/scripts/repro_paths.py"]
POINTS = f"{PROVENANCE}/data/figure01/comprehensive_task_map_points.csv"
STYLE = f"{PROVENANCE}/data/style/FIG168_MODEL_STYLE_YELLOW_SCRATCH_R04.json"
REFERENCE = f"{PROVENANCE}/R06_clean_raw_pca/figures/figure01a_clean_raw_pca.pdf"
STEM = "figure01a_clean_raw_pca"

RENDER_SET = "analysis/cr_sb2te3_blender_render_set_20261105"
RENDER_MANIFEST = f"{RENDER_SET}/RENDER_SET_MANIFEST.json"
STORE_MANIFEST = "release/revision1_submitted_20261110/imports/EXTERNAL_STORE_MANIFEST.tsv"
FILE_KEYS = (("render", "render_sha256"), ("blend", "blend_sha256"))


def lint_renderers(ev: Evidence) -> None:
    findings = [f for script in [RENDERER, *HELPERS] for f in lint_script(ev.input(script))]
    ev.check("hardcode lint of renderer chain", not findings, findings or [RENDERER, *HELPERS])


def reproduce_panel_b(ev: Evidence) -> None:
    ev.input(POINTS)
    ev.input(STYLE)
    out_dir = ev.output("fig1b")
    out_dir.mkdir(exist_ok=True)
    result = subprocess.run([sys.executable, str(REPO / RENDERER), "--out-dir", str(out_dir)],
                            cwd=REPO / PROVENANCE / "scripts", capture_output=True, text=True,
                            env={**os.environ, "MPLBACKEND": "Agg"})
    if not ev.check("panel b renderer exits 0", result.returncode == 0,
                    result.stderr[-2000:] or "ok"):
        return
    ev.compare_pdf(out_dir / f"{STEM}.pdf", REFERENCE)


def store_hashes() -> dict[str, str]:
    path = REPO / STORE_MANIFEST
    with path.open(newline="") as handle:
        return {row["repo_path"]: row["sha256"] for row in csv.DictReader(handle, delimiter="\t")}


def verify_render_set(ev: Evidence) -> None:
    manifest = json.loads(ev.input(RENDER_MANIFEST).read_text())
    store = store_hashes()
    ev.input(STORE_MANIFEST)
    rows, failures = [], []
    for panel in manifest["panels"]:
        for key, hash_key in FILE_KEYS:
            repo_path = f"{RENDER_SET}/{panel[key]}"
            recorded = panel[hash_key]
            path = REPO / repo_path
            if path.exists():
                actual, where = sha256(path), "repo"
            else:
                actual, where = store.get(repo_path), "external store manifest"
            row = {"panel": panel["id"], "file": repo_path, "recorded": recorded, "found": actual, "where": where}
            rows.append(row)
            if actual != recorded:
                failures.append(row)
    ev.check("panels a/c: Blender renders and .blend sources match RENDER_SET_MANIFEST sha256 "
             "(Blender not re-run; final layout composed in Canva)", not failures and bool(rows),
             failures or rows)


def main() -> int:
    ev = Evidence(__file__, "Fig1_a-c")
    lint_renderers(ev)
    reproduce_panel_b(ev)
    verify_render_set(ev)
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
