#!/usr/bin/env python3
"""Export Cr fixed-path NEB profiles from the MigrationBench asset registry to JSON.

The registry SQLite is the entry point: the chart's data-file locator and
status are read from it (read-only), the linked TSV is loaded, and the result
is written to public/data/ for Remotion. Status/caveat fields are carried
through unchanged so the video can label them.

Usage:
    python3 scripts/export_neb_profiles.py [--registry PATH] [--root PATH]
"""

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

# Repository root (this file lives at <repo>/video/migration_story/scripts/).
DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = (
    DEFAULT_ROOT
    / "analysis/migrationbench_asset_registry_20261025/migrationbench_asset_registry.sqlite"
)
CHART_ID = "chart_cross_element_neb_4x3_20261024"
FILE_ID = "cr_profiles"
OUT_PATH = Path(__file__).resolve().parent.parent / "public/data/cr_neb_profiles.json"

PATH_ORDER = ["within_gap", "gap_to_ql", "within_ql"]
SERIES_ORDER = [
    "qe_reference",
    "foundation_omat0_medium",
    "ft600k_seed123",
    "ftmultit_seed234",
    "from_scratch_seed123",
]


def read_registry(registry: Path) -> dict:
    uri = f"file:{registry}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        chart = conn.execute(
            "SELECT chart_id, title, chart_status, notes FROM charts WHERE chart_id = ?",
            (CHART_ID,),
        ).fetchone()
        data_file = conn.execute(
            "SELECT local_path FROM chart_data_files WHERE chart_id = ? AND file_id = ?",
            (CHART_ID, FILE_ID),
        ).fetchone()
    if chart is None or data_file is None:
        sys.exit(f"Registry has no {CHART_ID}/{FILE_ID} entry: {registry}")
    return {
        "chart_id": chart[0],
        "title": chart[1],
        "chart_status": chart[2],
        "notes": chart[3],
        "local_path": data_file[0],
    }


def load_rows(tsv: Path) -> list[dict]:
    with tsv.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def build_paths(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(lambda: defaultdict(list))
    labels, series_meta = {}, {}
    for row in rows:
        grouped[row["path_key"]][row["series_key"]].append(row)
        labels[row["path_key"]] = row["path_label"]
        series_meta[row["series_key"]] = {
            "label": row["series_label"],
            "kind": row["kind"],
            "status": row["status"],
        }

    paths = []
    for path_key in PATH_ORDER:
        series = []
        for series_key in SERIES_ORDER:
            pts = sorted(grouped[path_key][series_key], key=lambda r: int(r["image_index"]))
            if not pts:
                continue
            points = [
                {"s": float(p["s"]), "e": float(p["relative_energy_eV"])} for p in pts
            ]
            series.append(
                {
                    "key": series_key,
                    **series_meta[series_key],
                    "points": points,
                    "barrier_eV": max(pt["e"] for pt in points),
                }
            )
        ref = next((s for s in series if s["key"] == "qe_reference"), None)
        for s in series:
            s["barrier_error_eV"] = (
                None if ref is None else s["barrier_eV"] - ref["barrier_eV"]
            )
        paths.append({"key": path_key, "label": labels[path_key], "series": series})
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    meta = read_registry(args.registry)
    tsv = args.root / meta["local_path"]
    if not tsv.is_file():
        sys.exit(f"Linked data file missing: {tsv}")

    rows = [r for r in load_rows(tsv) if r["element"] == "Cr"]
    payload = {
        "provenance": {
            **meta,
            "registry": str(args.registry),
            "source_tsv": str(tsv),
            "source_sha256": hashlib.sha256(tsv.read_bytes()).hexdigest(),
            "exported_on": date.today().isoformat(),
            "caveat": "QE reference COMPLETED_BUT_SCF_WARNING; MLIP rows are fixed-path "
            "single-point diagnostics (FINITE_DIAGNOSTIC), not relaxed barriers.",
        },
        "paths": build_paths(rows),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_PATH} ({len(rows)} rows, sha256 {payload['provenance']['source_sha256'][:12]})")
    for p in payload["paths"]:
        print(p["label"], [(s["key"], round(s["barrier_eV"], 3)) for s in p["series"]])


if __name__ == "__main__":
    main()
