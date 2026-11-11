#!/usr/bin/env python3
"""Figure 6 (a-d): t-SNE / PHATE / UMAP-density / silhouette representation analysis.

Full mode (phate and umap-learn importable): runs render_fig6_full_campaign_20261102.py on
fig6_current_input.npz, recomputing all embeddings, with output in rerun_output/.
Fallback mode (either package missing): re-renders from the committed embeddings npz
(--embeddings) and records the missing packages as a failed check, plus a t-SNE-only
recomputation against the stored t-SNE coordinates. Either way the regenerated 1x4 PDF is
compared with the committed pre-Canva panel whose sha256 is in fig6_render_manifest.json.
Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import REPO, Evidence, lint_script, sha256  # noqa: E402

LEGACY = "analysis/legacy_fig6_phate_reproduction_20261101"
RENDERER = f"{LEGACY}/render_fig6_full_campaign_20261102.py"
PREPARE = f"{LEGACY}/prepare_fig6_full_campaign_20261102.py"
FEATURES = "release/revision1_camera_ready/reproducibility/fig6_current_input.npz"
PREP_MANIFEST = f"{LEGACY}/updated_campaign_20261102/source/preparation_manifest.json"
COMMITTED = f"{LEGACY}/updated_campaign_20261102_umap_density"
EMBEDDINGS = f"{COMMITTED}/fig6_full_campaign_embeddings.npz"
STEM = "fig6_umap_density_full_campaign_15models_1x4"
REFERENCE = f"{COMMITTED}/figures/{STEM}.pdf"
RENDER_MANIFEST = "paper/revision1/data_processed/current_release/fig6_render_manifest.json"
EMBEDDING_PACKAGES = ("phate", "umap")
TSNE_ATOL = 1e-3  # coordinates are float32; anything larger is a different embedding


def lint_renderers(ev: Evidence) -> None:
    findings = [f for script in (RENDERER, PREPARE) for f in lint_script(ev.input(script))]
    ev.check("hardcode lint of renderer chain", not findings, findings or [RENDERER, PREPARE])


def check_input_matches_audit(ev: Evidence, manifest: dict) -> None:
    payload = np.load(ev.input(FEATURES), allow_pickle=False)
    labels = np.asarray(payload["model_ids"], dtype=str)
    audit = manifest["data_audit"]
    found = {"n_rows": int(payload["features"].shape[0]), "n_features": int(payload["features"].shape[1]),
             "rows_by_model": {m: int(np.sum(labels == m)) for m in audit["rows_by_model"]}}
    expected = {key: audit[key] for key in found}
    ev.check("feature input shape and per-model row counts equal fig6_render_manifest data_audit",
             found == expected, {"found": found, "manifest": expected})


def reference_is_canva_input(ev: Evidence, manifest: dict) -> None:
    recorded, actual = manifest["outputs"]["main_pdf"]["sha256"], sha256(REPO / REFERENCE)
    ev.check(f"reference panel sha256 equals {RENDER_MANIFEST} outputs.main_pdf",
             recorded == actual, {"recorded": recorded, "reference": actual})


def check_stored_embeddings(ev: Evidence, manifest: dict) -> None:
    """The stored embeddings are the ones the submitted panel was drawn from."""
    from sklearn.metrics import silhouette_score

    stored = np.load(ev.input(EMBEDDINGS), allow_pickle=False)
    score = float(silhouette_score(stored["tsne"], stored["labels"]))
    recorded = manifest["embedding"]["overall_tsne_silhouette"]
    ev.check("stored t-SNE embedding reproduces the recorded overall silhouette",
             abs(score - recorded) < 1e-6, {"recomputed": score, "manifest": recorded})


def check_tsne_recompute(ev: Evidence) -> None:
    from sklearn import __version__ as sklearn_version
    from sklearn.manifold import TSNE

    features = np.asarray(np.load(REPO / FEATURES)["features"], dtype=float)
    stored = np.load(REPO / EMBEDDINGS)["tsne"]
    fresh = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000).fit_transform(features)
    diff = float(np.max(np.abs(fresh - stored)))
    ev.check("t-SNE recomputed in this environment equals stored t-SNE coordinates", diff <= TSNE_ATOL,
             {"max_abs_diff": diff, "stored_max_abs": float(np.max(np.abs(stored))),
              "sklearn": sklearn_version, "note": "original environment versions were not recorded"})


def run_renderer(ev: Evidence, full: bool) -> None:
    out_dir = ev.output("fig6")
    command = [sys.executable, str(REPO / RENDERER), "--features", str(REPO / FEATURES), "--out-dir", str(out_dir)]
    if not full:
        command += ["--embeddings", str(REPO / EMBEDDINGS)]
    result = subprocess.run(command, cwd=REPO / LEGACY, capture_output=True, text=True,
                            env={**os.environ, "MPLBACKEND": "Agg"})
    mode = "full (recomputed embeddings)" if full else "re-render from stored embeddings"
    if ev.check(f"renderer exits 0 [{mode}]", result.returncode == 0, result.stderr[-2000:] or "ok"):
        ev.compare_pdf(out_dir / "figures" / f"{STEM}.pdf", REFERENCE)


def interpreter() -> str:
    path = Path(sys.executable)
    return path.relative_to(REPO).as_posix() if path.is_relative_to(REPO) else path.name


def main() -> int:
    ev = Evidence(__file__, "Fig6_a-d")
    manifest = json.loads(ev.input(RENDER_MANIFEST).read_text())
    ev.input(PREP_MANIFEST)
    lint_renderers(ev)
    check_input_matches_audit(ev, manifest)
    reference_is_canva_input(ev, manifest)
    missing = [name for name in EMBEDDING_PACKAGES if importlib.util.find_spec(name) is None]
    full = not missing
    ev.check("embedding packages available for full recomputation", full,
             "phate, umap importable" if full else
             f"missing from {interpreter()}: {missing} (not installed); PHATE/UMAP embeddings not recomputed")
    if not full:
        check_stored_embeddings(ev, manifest)
        check_tsne_recompute(ev)
    run_renderer(ev, full)
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
