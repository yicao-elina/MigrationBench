#!/usr/bin/env python3
"""Fig. 5 a-e: NEB task matrix (design schematic, 120-record dot matrix, within-QL and sliding peaks).

The main-text figure was composed in Canva from the script page NEB_main_figure.pdf written by
release/revision1_camera_ready/reproducibility/render_fig5_corrected.py; that corrected script page
is committed as paper/revision1/figures/main/fig5_NEB_main_figure (2).pdf and is the reference for
panels a-e.  Panel a is also compared on its own (NEB_panel_a_design_v3, unchanged by the
correction).  Canva redrew panel a with icons, so the final asset differs there by design.

Writes rerun_output/ and evidence/Fig5_a-e.json.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import REPO, Evidence, lint_script, rel, sha256  # noqa: E402

RENDERER = "release/revision1_camera_ready/reproducibility/render_fig5_corrected.py"
CURRENT_RELEASE = "paper/revision1/data_processed/current_release"
INPUT = f"{CURRENT_RELEASE}/fig5_corrected_input.tsv"
CORRECTION_AUDIT = f"{CURRENT_RELEASE}/fig5_correction_audit.json"
RENDER_AUDIT = f"{CURRENT_RELEASE}/fig5_render_audit.json"
PLOTTED_VALUES = f"{CURRENT_RELEASE}/fig5_plotted_values.tsv"
SCRIPT_PAGE = "paper/revision1/figures/main/fig5_NEB_main_figure (2).pdf"
PANEL_A = "analysis/neb_dotmatrix_v3_latest_20261101/NEB_panel_a_design_v3.pdf"
FINAL_ASSET = "paper/revision1/figures/main/fig5_NEB_main_figure_new.pdf"
NUMERIC_TOLERANCE = 1e-12


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def same_value(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) <= NUMERIC_TOLERANCE
    except ValueError:
        return False


def compare_plotted(produced: Path, reference: Path) -> list[str]:
    """Cell-by-cell equality; numbers compare by value (the committed copy wrote '0.0' for '0.00')."""
    ours, theirs = read_rows(produced), read_rows(reference)
    if len(ours) != len(theirs) or (ours and ours[0].keys() != theirs[0].keys()):
        return [f"shape differs: {len(ours)} vs {len(theirs)} rows"]
    return [f"row {i} {key}: {a[key]!r} vs {b[key]!r}" for i, (a, b) in enumerate(zip(ours, theirs), start=1)
            for key in a if not same_value(a[key], b[key])]


def main() -> int:
    ev = Evidence(__file__, "Fig5_a-e")
    for path in (RENDERER, INPUT, CORRECTION_AUDIT, RENDER_AUDIT, PLOTTED_VALUES, SCRIPT_PAGE, PANEL_A, FINAL_ASSET):
        ev.input(path)
    findings = lint_script(REPO / RENDERER)
    ev.check(f"hardcode lint {RENDERER}", not findings, findings or "no findings")

    out = ev.output("NEB_Dotmatrix_v3")
    run = subprocess.run([sys.executable, str(REPO / RENDERER), "--input", str(REPO / INPUT),
                          "--correction-audit", str(REPO / CORRECTION_AUDIT), "--output", str(out)],
                         cwd=REPO, capture_output=True, text=True)
    if not ev.check("renderer exits 0", run.returncode == 0, run.stderr[-2000:] or run.stdout[-500:]):
        return ev.write()

    ev.compare_pdf(out / "NEB_main_figure.pdf", SCRIPT_PAGE)
    ev.compare_pdf(out / "NEB_panel_a_design_v3.pdf", PANEL_A)
    audit_same = (out / "audit.json").read_bytes() == (REPO / RENDER_AUDIT).read_bytes()
    ev.outputs[rel(out / "audit.json")] = sha256(out / "audit.json")
    ev.check(f"audit.json equals {RENDER_AUDIT}", audit_same, "byte-identical" if audit_same else "differs")
    errors = compare_plotted(out / "plotted_values.tsv", REPO / PLOTTED_VALUES)
    ev.outputs[rel(out / "plotted_values.tsv")] = sha256(out / "plotted_values.tsv")
    ev.check(f"plotted_values.tsv equals {PLOTTED_VALUES} cell by cell", not errors, errors[:10] or "120 rows equal")

    correction = json.loads((REPO / CORRECTION_AUDIT).read_text())
    caption = (out / "figure_caption.txt").read_text()
    ev.outputs[rel(out / "figure_caption.txt")] = sha256(out / "figure_caption.txt")
    counts = correction["revised_status_counts"]
    ev.check("generated caption carries the audited counts", all(str(n) in caption for n in counts.values() if n >= 10),
             {"caption": caption, "status_counts": counts})
    ev.check("known differences (not used for PASS)", True, {
        "caption_rounding": "the previous hand-typed caption said 2.3507667 eV; the value in the input and the "
                            "correction audit is 2.3507668938545976 eV, so the generated 7-decimal caption reads "
                            "2.3507669 eV (only auxiliary figure_caption.txt; not in the submitted manuscript)",
        "note_pdf": "NEB_dot_encoding_and_layout_v3.pdf (reportlab layout note) is skipped: reportlab not installed",
        "final_asset": f"{FINAL_ASSET} is the Canva composition: panel a redrawn with icons, panels b-e placed "
                       "from the script page compared above"})
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
