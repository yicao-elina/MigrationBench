#!/usr/bin/env python3
"""Check that the SQLite asset registry is complete and clean, then compact it.

Checks (exit 1 if any fails):
  1. PRAGMA integrity_check and foreign_key_check pass;
  2. every rev1_* table has a primary key and its row count equals its source TSV;
  3. no cell anywhere holds an absolute /Users/ path (rev1_path_roots excepted);
  4. no table has fully duplicated rows; every view runs.
With --compact: ANALYZE + VACUUM, refresh the JSON and CSV mirrors, and log the result in audit_log.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sqlite3
import sys
from datetime import date

from path_roots import PACKAGE, REPO

REGISTRY = REPO / "analysis/migrationbench_asset_registry_20261025"
DB = REGISTRY / "migrationbench_asset_registry.sqlite"
TSV_BACKED = {
    "rev1_results_index": "RESULTS_INDEX.tsv", "rev1_storage_locations": "STORAGE_LOCATIONS.tsv",
    "rev1_model_registry": "MODEL_REGISTRY.tsv", "rev1_rockfish_migration_plan": "ROCKFISH_MIGRATION_PLAN.tsv",
    "rev1_import_plan": "imports/import_plan.tsv", "rev1_imported_files": "imports/IMPORT_MANIFEST.tsv",
    "rev1_external_store": "imports/EXTERNAL_STORE_MANIFEST.tsv",
    "rev1_archived_untracked_raw": "imports/ARCHIVED_UNTRACKED_RAW.tsv",
    "rev1_downloads_archive": "imports/DOWNLOADS_ARCHIVE.tsv", "rev1_icloud_dataless": "imports/ICLOUD_DATALESS.tsv",
    "rev1_deferred_large_files": "imports/DEFERRED_LARGE_FILES.tsv",
    "rev1_import_conflicts": "imports/IMPORT_CONFLICTS.tsv",
    "rev1_item_calc_layers": "sources/item_calc_layers.tsv",
    "rev1_reproducibility": "REPRODUCIBILITY.tsv",
    "rev1_reproduce_exceptions": "sources/reproduce_exceptions.tsv",
    "rev1_known_discrepancies": "sources/known_discrepancies.tsv",
}
DERIVED = ["rev1_path_roots", "rev1_items", "rev1_pipeline_stages", "rev1_known_issues"]


def tsv_rows(rel: str) -> int:
    with (PACKAGE / rel).open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle, delimiter="\t"))


def objects(conn, kind: str) -> list[str]:
    return [n for (n,) in conn.execute("SELECT name FROM sqlite_master WHERE type=? AND name NOT LIKE 'sqlite_%' "
                                       "ORDER BY name", (kind,))]


def run_checks(conn: sqlite3.Connection) -> list[str]:
    errors = []
    if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        errors.append("integrity_check failed")
    errors += [f"foreign key violation: {row}" for row in conn.execute("PRAGMA foreign_key_check")]
    tables = objects(conn, "table")
    for name in [*TSV_BACKED, *DERIVED]:
        if name not in tables:
            errors.append(f"missing table {name}")
            continue
        if not any(col[5] for col in conn.execute(f'PRAGMA table_info("{name}")')):
            errors.append(f"{name}: no primary key")
        count = conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        expected = tsv_rows(TSV_BACKED[name]) if name in TSV_BACKED else None
        if count == 0 or (expected is not None and count != expected):
            errors.append(f"{name}: {count} rows, expected {expected if expected is not None else '> 0'}")
    for name in tables:
        columns = [col[1] for col in conn.execute(f'PRAGMA table_info("{name}")')]
        quoted = ", ".join(f'"{c}"' for c in columns)
        dupes = conn.execute(f'SELECT COUNT(*) FROM (SELECT 1 FROM "{name}" GROUP BY {quoted} HAVING COUNT(*) > 1)')
        if dupes.fetchone()[0]:
            errors.append(f"{name}: duplicated rows")
        for column in columns:
            if name == "rev1_path_roots" and column == "absolute_root":
                continue
            hits = conn.execute(f'SELECT COUNT(*) FROM "{name}" WHERE CAST("{column}" AS TEXT) LIKE \'%/Users/%\'')
            if hits.fetchone()[0]:
                errors.append(f"{name}.{column}: absolute /Users/ path")
    for view in objects(conn, "view"):
        try:
            conn.execute(f'SELECT COUNT(*) FROM "{view}"').fetchone()
        except sqlite3.Error as exc:
            errors.append(f"view {view} fails: {exc}")
    return errors


def refresh_mirrors(conn: sqlite3.Connection) -> list[str]:
    """Rewrite the CSV mirrors that exist next to the DB, and the JSON snapshot."""
    written = []
    for name in objects(conn, "table"):
        path = REGISTRY / f"{name}.csv"
        if not path.exists():
            continue
        cursor = conn.execute(f'SELECT * FROM "{name}"')
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow([col[0] for col in cursor.description])
            writer.writerows(cursor)
        written.append(path.name)
    spec = importlib.util.spec_from_file_location("export_registry_data", REGISTRY / "export_registry_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.export_registry(REGISTRY)
    return written + ["registry_data.json"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compact", action="store_true", help="ANALYZE + VACUUM, refresh mirrors, log the check")
    args = parser.parse_args()
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys = ON")
    errors = run_checks(conn)
    tables, views = objects(conn, "table"), objects(conn, "view")
    print(f"{len(tables)} tables, {len(views)} views checked")
    for error in errors:
        print(f"FAIL {error}")
    if args.compact and not errors:
        conn.execute("INSERT INTO audit_log (checked_at, check_name, result, details) VALUES (?, ?, ?, ?)",
                     (date.today().isoformat(), "check_sql", "PASS",
                      f"{len(tables)} tables, {len(views)} views; integrity, keys, row counts, portable paths"))
        conn.commit()
        conn.execute("ANALYZE")
        conn.execute("VACUUM")
        print("refreshed:", ", ".join(refresh_mirrors(conn)))
    conn.close()
    print("SQL OK" if not errors else f"{len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
