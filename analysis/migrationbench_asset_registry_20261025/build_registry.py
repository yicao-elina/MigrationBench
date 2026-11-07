#!/usr/bin/env python3
"""Rebuild the local MigrationBench registry from its versioned data bundle.

This local builder intentionally reads only files in this directory.  It does
not reach into the Google-Drive-mounted workspace, so a dehydrated cloud file
cannot replace the validated local registry.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "registry_data.json"
SCHEMA_PATH = ROOT / "schema.sql"
DB_PATH = ROOT / "migrationbench_asset_registry.sqlite"
REPORT_PATH = ROOT / "data_quality_report.json"

TABLES = (
    "sources",
    "model_families",
    "model_groups",
    "models",
    "model_group_members",
    "model_name_map",
    "model_aliases",
    "tasks",
    "task_groups",
    "task_group_members",
    "task_aliases",
    "task_group_aliases",
    "systems",
    "campaigns",
    "datasets",
    "coverage",
    "assets",
    "charts",
    "chart_panels",
    "chart_links",
    "chart_data_files",
    "chart_provenance",
    "gaps",
)

CSV_TABLES = TABLES
ID_TABLES = {"model_aliases", "task_aliases", "task_group_aliases"}


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def _insert_rows(conn: sqlite3.Connection, table: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    columns = _columns(conn, table)
    next_id = 1
    for row in rows:
        payload = dict(row)
        if table in ID_TABLES and "alias_id" in columns and "alias_id" not in payload:
            payload["alias_id"] = next_id
            next_id += 1
        present = [column for column in columns if column in payload]
        values = [payload[column] for column in present]
        marks = ", ".join("?" for _ in present)
        names = ", ".join(present)
        conn.execute(f"INSERT INTO {table} ({names}) VALUES ({marks})", values)


def _write_csv(conn: sqlite3.Connection, table: str) -> None:
    path = ROOT / f"{table}.csv"
    rows = conn.execute(f"SELECT * FROM {table}").fetchall()
    headers = [column[0] for column in conn.execute(f"PRAGMA table_info({table})")]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    with tempfile.NamedTemporaryFile(prefix="migrationbench_registry_", suffix=".sqlite", dir=ROOT, delete=False) as handle:
        temporary_db = Path(handle.name)
    try:
        conn = sqlite3.connect(temporary_db)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(schema)
        for table in TABLES:
            _insert_rows(conn, table, list(data.get(table, [])))
        for key, value in dict(data.get("metadata", {})).items():
            conn.execute("INSERT INTO metadata (key, value) VALUES (?, ?)", (key, str(value)))
        checked_at = datetime.now(timezone.utc).isoformat()
        for check in data.get("quality_checks", []):
            conn.execute(
                "INSERT INTO audit_log (checked_at, check_name, result, details) VALUES (?, ?, ?, ?)",
                (checked_at, check["check_name"], check["result"], check["details"]),
            )
        conn.commit()
        conn.execute("PRAGMA foreign_key_check").fetchall()
        for table in CSV_TABLES:
            _write_csv(conn, table)
        counts = {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in TABLES}
        counts.update({"audit_log": conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]})
        quality = {
            "built_at": checked_at,
            "source_of_truth": str(ROOT),
            "counts": counts,
            "checks": data.get("quality_checks", []),
        }
        REPORT_PATH.write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        conn.close()
        os.replace(temporary_db, DB_PATH)
    finally:
        if temporary_db.exists():
            temporary_db.unlink()
    print(json.dumps({"database": str(DB_PATH), "counts": counts, "source_of_truth": str(ROOT)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
