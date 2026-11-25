#!/usr/bin/env python3
"""Export the local SQLite source of truth to the portable JSON snapshot."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "migrationbench_asset_registry.sqlite"
DATA_PATH = ROOT / "registry_data.json"
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


def export_registry(root: Path = ROOT) -> dict[str, object]:
    db_path = root / DB_PATH.name
    data_path = root / DATA_PATH.name
    report_path = root / REPORT_PATH.name
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    payload: dict[str, object] = {}
    for table in TABLES:
        payload[table] = [dict(row) for row in conn.execute(f"SELECT * FROM {table}")]
    payload["metadata"] = {
        row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM metadata ORDER BY key")
    }
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    payload["quality_checks"] = report.get("checks", [])
    conn.close()

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=root, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, data_path)
    return {"path": str(data_path), "counts": {key: len(value) for key, value in payload.items() if isinstance(value, list)}}


if __name__ == "__main__":
    print(json.dumps(export_registry(), ensure_ascii=False, indent=2))
