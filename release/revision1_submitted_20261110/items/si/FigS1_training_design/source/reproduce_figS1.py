#!/usr/bin/env python3
"""Fig. S1: training-route schematic drawn in diagrams.net (no data content).

The submitted PDF was exported by diagrams.net with "include a copy of my diagram", so the exact
editable diagram behind it is stored URL-encoded in the PDF Subject field.  This script
  * checks the submitted asset's sha256 against RESULTS_INDEX.tsv,
  * extracts the embedded diagram and checks it is byte-identical to the committed editable source
    fig_training_design_routes_submitted.drawio,
  * reports how the older editable file method-P02_palette_updated_pre_vector_export.pdf.drawio
    (2026-11-06) differs from the submitted diagram (2026-11-08 edit), and
  * if a diagrams.net CLI is available (DRAWIO_BIN or `drawio` on PATH), re-exports the diagram
    and records the pixel difference (informational: GUI and CLI exports lay out text differently).

Writes rerun_output/ and evidence/FigS1.json.
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import PACKAGE, REPO, Evidence, image_difference, rasterize, rel, sha256  # noqa: E402

ITEM = "release/revision1_submitted_20261110/items/si/FigS1_training_design/source"
SUBMITTED_SOURCE = f"{ITEM}/fig_training_design_routes_submitted.drawio"
OLDER_SOURCE = f"{ITEM}/method-P02_palette_updated_pre_vector_export.pdf.drawio"
ASSET = "paper/revision1/figures/main/SI/fig_training_design_routes.pdf"
ITEM_ID = "Fig. S1"
EXPORT_TIMEOUT_S = 180


def embedded_diagram(pdf: Path) -> str:
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    subject = next(line.split(":", 1)[1].strip() for line in info.splitlines() if line.startswith("Subject:"))
    return urllib.parse.unquote(subject)


def cells(xml_text: str) -> dict[str, tuple]:
    out = {}
    for cell in ET.fromstring(xml_text).iter("mxCell"):
        geometry = cell.find("mxGeometry")
        points = tuple(tuple(sorted(p.attrib.items())) for p in geometry.iter("mxPoint")) if geometry is not None else ()
        out[cell.get("id")] = (tuple(sorted(cell.attrib.items())),
                               tuple(sorted(geometry.attrib.items())) if geometry is not None else (), points)
    return out


def indexed_sha(item_id: str) -> str:
    """Full asset_sha256, or the 'sha xxxxxxxx...' prefix SI rows carry in final_asset."""
    with (PACKAGE / "RESULTS_INDEX.tsv").open(newline="") as handle:
        row = next(r for r in csv.DictReader(handle, delimiter="\t") if r["item_id"] == item_id)
    return row["asset_sha256"] or re.search(r"sha ([0-9a-f]{8,})", row["final_asset"]).group(1)


def drawio_export(ev: Evidence, diagram: Path) -> None:
    binary = os.environ.get("DRAWIO_BIN") or shutil.which("drawio")
    if not binary:
        ev.check("diagrams.net re-export (informational)", True, "no diagrams.net CLI found; export not re-run")
        return
    target = ev.output("fig_training_design_routes_cli_export.pdf")
    run = subprocess.run([binary, "--export", "--format", "pdf", "--crop", "--output", str(target), str(diagram)],
                         capture_output=True, text=True, timeout=EXPORT_TIMEOUT_S)
    details = {"returncode": run.returncode}
    if run.returncode == 0 and target.exists():
        ev.outputs[rel(target)] = sha256(target)
        details["metrics_at_100dpi"] = image_difference(rasterize(target, 100), rasterize(REPO / ASSET, 100))
        details["note"] = ("CLI export of the identical diagram; differs from the GUI export in page crop and "
                           "text scaling, so it is not used for PASS")
    ev.check("diagrams.net re-export (informational)", True, details)


def main() -> int:
    ev = Evidence(__file__, "FigS1")
    asset = ev.input(ASSET)
    expected = indexed_sha(ITEM_ID)
    ev.check("submitted asset sha256 matches RESULTS_INDEX", sha256(asset).startswith(expected),
             {"sha256": sha256(asset), "indexed": expected})

    embedded = embedded_diagram(asset)
    extracted = ev.output("embedded_in_submitted_pdf.drawio")
    extracted.write_text(embedded)
    ev.outputs[rel(extracted)] = sha256(extracted)
    source = ev.input(SUBMITTED_SOURCE)
    ev.check(f"editable source {SUBMITTED_SOURCE} equals the diagram embedded in the asset",
             source.read_text() == embedded, {"embedded_sha256": sha256(extracted), "source_sha256": sha256(source)})

    older, current = cells(ev.input(OLDER_SOURCE).read_text()), cells(embedded)
    changed = sorted(k for k in current.keys() & older.keys() if current[k] != older[k])
    ev.check("known difference: older editable file vs submitted diagram (not used for PASS)", True, {
        "cells_submitted": len(current), "cells_older": len(older),
        "only_in_submitted": sorted(current.keys() - older.keys()),
        "only_in_older": sorted(older.keys() - current.keys()), "changed_cells": changed})
    drawio_export(ev, source)
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
