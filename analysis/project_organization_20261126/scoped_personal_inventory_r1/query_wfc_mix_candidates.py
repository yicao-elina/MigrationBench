#!/usr/bin/env python3
"""Read-only filename recheck for WFC/mixing candidates in the scoped inventory."""

from __future__ import print_function

import argparse
import json
import os
import sqlite3
from datetime import datetime


QUERY = """SELECT root_id,relative_path,logical_bytes,allocated_bytes,
                  mtime_ns,mode_octal
           FROM inventory_entry
           WHERE entry_type='FILE'
             AND (lower(relative_path) LIKE '%.wfc%'
                  OR lower(relative_path) LIKE '%.mix%')
           ORDER BY logical_bytes DESC,relative_path"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    database = os.path.abspath(args.database)
    connection = sqlite3.connect("file:{}?mode=ro".format(database), uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    try:
        rows = [dict(row) for row in connection.execute(QUERY)
                if ".wfc" in os.path.basename(row["relative_path"]).lower()
                or ".mix" in os.path.basename(row["relative_path"]).lower()]
    finally:
        connection.close()
    summary = {}
    for row in rows:
        root_id = row["root_id"]
        bucket = summary.setdefault(root_id, {"candidate_files": 0, "nonempty_files": 0,
                                              "logical_bytes": 0, "allocated_bytes": 0})
        bucket["candidate_files"] += 1
        bucket["nonempty_files"] += int(row["logical_bytes"] > 0)
        bucket["logical_bytes"] += row["logical_bytes"]
        bucket["allocated_bytes"] += row["allocated_bytes"] or 0
    payload = {
        "captured_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "database_path": database,
        "pattern": "basename contains .wfc or .mix (case-insensitive); metadata only",
        "query_only": True,
        "candidate_counts": summary,
        "rows": rows,
    }
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(args.output, 0o600)
    print(json.dumps({"candidate_counts": summary, "rows": len(rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
