#!/usr/bin/env python3
"""Build the dated MigrationBench figure-registry snapshot for the R05 bundle.

The script copies a materialized, SQL-first base snapshot, applies the
idempotent migration, runs SQLite integrity checks, and records hashes/counts
in a JSON manifest. It never edits the File Provider-backed registry in
analysis/ directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    here = Path(__file__).resolve()
    registry_dir = here.parents[1]
    repo_root = here.parents[4]
    default_base = registry_dir / "migrationbench_asset_registry_20261101_base_20261028.sqlite"
    default_sql = registry_dir / "updates/20261101_cr_scratch600k_R05.sql"
    default_output = registry_dir / "migrationbench_asset_registry_20261101.sqlite"
    default_manifest = registry_dir / "registry_manifest_20261101.json"

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=default_base)
    parser.add_argument("--migration", type=Path, default=default_sql)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base = args.base.resolve()
    migration = args.migration.resolve()
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    for path in (base, migration):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; use --overwrite")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    shutil.copy2(base, output)

    with sqlite3.connect(output) as con:
        con.execute("PRAGMA foreign_keys = ON")
        con.executescript(migration.read_text(encoding="utf-8"))
        foreign_keys = con.execute("PRAGMA foreign_key_check").fetchall()
        chart_id = "chart_cr_three_path_scratch600k_r05_20261101"
        chart = con.execute(
            "SELECT chart_id, chart_status, asset_path FROM charts WHERE chart_id=?",
            (chart_id,),
        ).fetchone()
        counts = {}
        for table in (
            "charts",
            "chart_panels",
            "chart_data_files",
            "chart_links",
            "chart_provenance",
            "campaigns",
            "datasets",
        ):
            counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        r05_counts = {
            "chart_panels": con.execute(
                "SELECT COUNT(*) FROM chart_panels WHERE chart_id=?", (chart_id,)
            ).fetchone()[0],
            "chart_data_files": con.execute(
                "SELECT COUNT(*) FROM chart_data_files WHERE chart_id=?", (chart_id,)
            ).fetchone()[0],
            "chart_links": con.execute(
                "SELECT COUNT(*) FROM chart_links WHERE chart_id=?", (chart_id,)
            ).fetchone()[0],
        }

    if foreign_keys:
        raise RuntimeError(f"foreign_key_check failed: {foreign_keys}")
    if not chart or chart[1] != "USER_CERTIFIED_RELEASE":
        raise RuntimeError(f"R05 chart row missing or unexpected: {chart}")
    if r05_counts != {"chart_panels": 6, "chart_data_files": 4, "chart_links": 5}:
        raise RuntimeError(f"R05 linkage counts failed: {r05_counts}")

    manifest = {
        "registry_name": "MigrationBench Asset Registry",
        "snapshot": "2026-11-01",
        "base_snapshot": str(base),
        "base_sha256": sha256(base),
        "migration_sql": str(migration),
        "migration_sql_sha256": sha256(migration),
        "output": str(output),
        "output_sha256": sha256(output),
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "r05_chart_id": chart[0],
        "r05_chart_status": chart[1],
        "r05_asset_path": chart[2],
        "integrity": {"foreign_key_check": "PASS"},
        "table_counts": counts,
        "r05_linkage_counts": r05_counts,
        "source_boundary": "The current analysis registry is a File Provider dataless placeholder; this dated snapshot is based on the materialized 2026-10-28 local SQL-first archive and does not overwrite it.",
        "repository_root": str(repo_root),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
