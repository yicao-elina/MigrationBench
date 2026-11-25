#!/usr/bin/env python3
"""Hi-res figure uploads (submission pp. 56-62) = the final Fig. 1-7 assets in the repo.

For each main figure: the final asset still has the sha256 recorded in sources/main_figures.tsv, and
the words on its hi-res page in the submitted PDF cover the words in that asset (Wiley
re-rendered the uploads, so the files are not byte-identical; the text layer is the evidence).
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import PACKAGE, Evidence, rel, sha256  # noqa: E402

SUBMITTED = "release/revision1_submitted_20261110/submitted/MigrationBench_R1_submitted_20261110.pdf"
INDEX = "release/revision1_submitted_20261110/sources/main_figures.tsv"  # stable survey source of RESULTS_INDEX
FIGURES = [f"Fig{n}" for n in range(1, 8)]
MIN_COVERAGE = 0.95  # fraction of the asset's distinct words found on the uploaded page
WORD = re.compile(r"[A-Za-z]{3,}")


def words(pdf: Path, page: int | None = None) -> set[str]:
    pages = ["-f", str(page), "-l", str(page)] if page else []
    text = subprocess.run(["pdftotext", *pages, str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    return {w.lower() for w in WORD.findall(text)}


def page_count(pdf: Path) -> int:
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    return int(re.search(r"Pages:\s+(\d+)", info).group(1))


def main() -> int:
    ev = Evidence(__file__, "HiResFigureFiles")
    submitted = ev.input(SUBMITTED)
    with ev.input(INDEX).open(newline="") as handle:
        assets = {}
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["item_id"] in FIGURES and row["asset_sha256"]:
                assets.setdefault(row["item_id"], (row["final_asset"], row["asset_sha256"]))
    total = page_count(submitted)
    first_upload = total - len(FIGURES)  # the uploads are the last pages, after the graphical abstract
    for offset, figure in enumerate(FIGURES, start=1):
        asset, recorded = assets[figure]
        path = ev.input(asset)
        ev.check(f"{figure} asset sha256 = recorded", sha256(path) == recorded, rel(path))
        own, page = words(path), words(submitted, first_upload + offset)
        coverage = len(own & page) / len(own) if own else 1.0
        ev.check(f"{figure} words on submitted page {first_upload + offset}", coverage >= MIN_COVERAGE,
                 {"asset_words": len(own), "coverage": round(coverage, 4)})
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
