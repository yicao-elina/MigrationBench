#!/usr/bin/env python3
"""Fig. S5: feature-space audit (a: original-space silhouettes; b: t-SNE perplexity sweep).

Rendered by the shared SI generator items/si/_shared_generators/audit_current_figures.py, which
also writes several SI tables. It is run unmodified with MB_AUDIT_OUT pointed at this item's
rerun_output/, so nothing committed is touched. The regenerated figure is compared with the
submitted SI asset (a direct script output) and the two metric tables behind it with the
committed current_release CSVs.

Writes rerun_output/ and evidence/FigS5.json.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import REPO, Evidence, lint_script, rel, sha256  # noqa: E402

GENERATOR = "release/revision1_submitted_20261110/items/si/_shared_generators/audit_current_figures.py"
FEATURES = "release/revision1_camera_ready/reproducibility/fig6_current_input.npz"
CURRENT_RELEASE = "paper/revision1/data_processed/current_release"
METRIC_TABLES = ["fig6_current_original_space_metrics.csv", "fig6_current_tsne_sensitivity.csv"]
FINAL_ASSET = "paper/revision1/figures/figS_current_feature_audit.pdf"
METRIC_TOLERANCE = 1e-9
SINGLE_THREAD = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def compare_csv(produced: Path, reference: Path) -> list[str]:
    with produced.open(newline="") as a, reference.open(newline="") as b:
        ours, theirs = list(csv.DictReader(a)), list(csv.DictReader(b))
    if len(ours) != len(theirs):
        return [f"{len(ours)} vs {len(theirs)} rows"]
    errors = []
    for i, (x, y) in enumerate(zip(ours, theirs), start=1):
        for key in y:
            try:
                same = abs(float(x[key]) - float(y[key])) <= METRIC_TOLERANCE
            except ValueError:
                same = x[key] == y[key]
            if not same:
                errors.append(f"row {i} {key}: {x[key]} vs {y[key]}")
    return errors


def main() -> int:
    ev = Evidence(__file__, "FigS5")
    for path in (GENERATOR, FEATURES, FINAL_ASSET, *(f"{CURRENT_RELEASE}/{n}" for n in METRIC_TABLES)):
        ev.input(path)
    findings = lint_script(REPO / GENERATOR)
    ev.check(f"hardcode lint {GENERATOR}", not findings, findings or "no findings")

    out = ev.output("audit_current_figures")
    run = subprocess.run([sys.executable, str(REPO / GENERATOR)], cwd=REPO, capture_output=True, text=True,
                         env={**os.environ, **SINGLE_THREAD, "MB_AUDIT_OUT": str(out)})
    if not ev.check("generator exits 0", run.returncode == 0, run.stderr[-2000:] or "ok"):
        return ev.write()
    for name in METRIC_TABLES:
        produced = out / "data_processed/current_release" / name
        ev.outputs[rel(produced)] = sha256(produced)
        errors = compare_csv(produced, REPO / CURRENT_RELEASE / name)
        ev.check(f"{name} equals {CURRENT_RELEASE}/{name} (tol {METRIC_TOLERANCE})", not errors, errors[:10] or "equal")
    ev.compare_pdf(out / "figures/figS_current_feature_audit.pdf", FINAL_ASSET)
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
