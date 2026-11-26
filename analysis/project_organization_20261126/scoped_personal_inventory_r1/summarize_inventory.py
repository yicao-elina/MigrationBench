#!/usr/bin/env python3
"""Run the registered QPR summaries read-only against the archived inventory."""

from __future__ import print_function

import argparse
import heapq
import hashlib
import json
import os
import sqlite3
from datetime import datetime


HERE = os.path.dirname(os.path.abspath(__file__))
QUERY_FILE = os.path.join(HERE, "queries.sql")


def load_queries():
    queries = {}
    current = None
    pending = []
    for line in open(QUERY_FILE, "r"):
        stripped = line.strip()
        if stripped.startswith("-- QPR"):
            if current is not None and pending:
                queries[current] = " ".join(pending).rstrip(";")
            current = stripped.split(":", 1)[0].split()[1]
            pending = []
        elif not stripped or stripped.startswith("--"):
            continue
        else:
            pending.append(stripped)
            if stripped.endswith(";"):
                queries[current] = " ".join(pending).rstrip(";")
                current = None
                pending = []
    if current is not None and pending:
        queries[current] = " ".join(pending).rstrip(";")
    return queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("database")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    database = os.path.abspath(args.database)
    uri = "file:{}?mode=ro".format(database.replace("?", "%3f").replace("#", "%23"))
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    results = {}
    try:
        queries = load_queries()
        for query_id in sorted(queries):
            cursor = connection.execute(queries[query_id])
            columns = [item[0] for item in cursor.description]
            if query_id == "QPR04":
                groups = {}
                for row in cursor:
                    root_id = row["root_id"]
                    bucket = groups.setdefault(root_id, {"group_count": 0, "extra_hardlink_paths": 0,
                                                         "allocated_path_sum_bytes": 0, "top_groups": []})
                    item = {key: row[key] for key in columns}
                    bucket["group_count"] += 1
                    bucket["extra_hardlink_paths"] += int(row["path_count"]) - 1
                    bucket["allocated_path_sum_bytes"] += int(row["allocated_path_sum_bytes"] or 0)
                    bucket["top_groups"].append(item)
                    bucket["top_groups"] = heapq.nlargest(20, bucket["top_groups"],
                                                          key=lambda x: x["allocated_path_sum_bytes"])
                results[query_id] = groups
            else:
                results[query_id] = [{key: row[key] for key in columns} for row in cursor.fetchall()]
        quick = connection.execute("PRAGMA quick_check").fetchone()[0]
        foreign_keys = len(connection.execute("PRAGMA foreign_key_check").fetchall())
        snapshot = dict(connection.execute("SELECT snapshot_id,started_utc,finished_utc,completion_state,host_name,scanner_version,scope_note,excluded_paths_json FROM inventory_snapshot").fetchone())
        if quick != "ok" or foreign_keys:
            raise RuntimeError("integrity check failed")
    finally:
        connection.close()
    digest = hashlib.sha256()
    with open(database, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    payload = {
        "summary_captured_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "database_path": database,
        "database_bytes": os.path.getsize(database),
        "database_sha256": digest.hexdigest(),
        "query_only": True,
        "sqlite_version": sqlite3.sqlite_version,
        "quick_check": quick,
        "foreign_key_violations": foreign_keys,
        "snapshot": snapshot,
        "queries": results,
    }
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(args.output, 0o600)
    print(json.dumps({"summary": args.output, "roots": results.get("QPR01", []),
                      "project_folders": len(results.get("QPR02", [])),
                      "largest_files": len(results.get("QPR03", [])),
                      "hardlink_groups": sum(row["group_count"] for row in results.get("QPR04", {}).values()),
                      "errors": len(results.get("QPR05", [])),
                      "folder_depth_rows": len(results.get("QPR06", [])),
                      "global_unique_inode": results.get("QPR07", []),
                      "quick_check": quick, "foreign_key_violations": foreign_keys},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
