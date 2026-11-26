#!/usr/bin/env python3
"""Read-only exact QE restart filename recheck against the R1 inventory."""
import argparse
import json
import os
import re
import sqlite3
from datetime import datetime

RESTART_BASENAME = re.compile(r"(?:\.wfc\d+|\.mix\d+|wfc[^/]*\.dat)$", re.IGNORECASE)
PATH_QUERY = """SELECT root_id,relative_path,logical_bytes,allocated_bytes,
                         mode_octal,mtime_ns
                  FROM inventory_entry
                  WHERE entry_type='FILE'
                    AND (lower(relative_path) LIKE '%.wfc%'
                         OR lower(relative_path) LIKE '%.mix%'
                         OR lower(relative_path) LIKE '%/wfc%.dat')
                  ORDER BY root_id,relative_path"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    database = os.path.abspath(args.database)
    con = sqlite3.connect("file:{}?mode=ro".format(database), uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    try:
        candidates = [dict(row) for row in con.execute(PATH_QUERY)
                      if RESTART_BASENAME.search(row["relative_path"].rsplit("/", 1)[-1])]
    finally:
        con.close()
    payload = {
        "captured_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "database_path": database,
        "query_only": True,
        "basename_patterns": ["*.wfcN", "wfc*.dat", "*.mixN"],
        "candidate_count": len(candidates),
        "nonempty_candidate_count": sum(row["logical_bytes"] > 0 for row in candidates),
        "logical_bytes": sum(row["logical_bytes"] for row in candidates),
        "rows": candidates,
    }
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(args.output, 0o600)
    print(json.dumps({key: payload[key] for key in (
        "captured_utc", "candidate_count", "nonempty_candidate_count", "logical_bytes")},
        indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
