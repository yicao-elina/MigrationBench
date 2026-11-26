#!/usr/bin/env python3
"""Run registered read-only queries against the Scratch16 hierarchy snapshot."""

import argparse
import csv
import json
import os
import sqlite3
import sys
from urllib.parse import quote


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "scratch16_folder_hierarchy_r1.sqlite")
QUERY_FILE = os.path.join(HERE, "queries.sql")
QUERY_IDS = {"QH01": 0, "QH02": 1, "QH03": 2, "QH04": 3, "QH05": 4}


def queries():
    raw = open(QUERY_FILE, "r").read()
    blocks = []
    current = []
    for line in raw.splitlines():
        if line.startswith("-- QH"):
            if current:
                blocks.append("\n".join(current).strip())
                current = []
        elif not line.startswith("--"):
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default=DEFAULT_DB)
    parser.add_argument("--query-id", choices=sorted(QUERY_IDS), required=True)
    parser.add_argument("--parameter", help="relative project path for QH04")
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    args = parser.parse_args()
    sql = queries()[QUERY_IDS[args.query_id]]
    if args.query_id == "QH04" and not args.parameter:
        parser.error("QH04 requires --parameter with a relative top-level folder")
    uri = "file:{}?mode=ro".format(quote(os.path.abspath(args.database), safe="/"))
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        cursor = connection.execute(sql, (args.parameter,) if args.query_id == "QH04" else ())
        columns = [item[0] for item in cursor.description]
        rows = cursor.fetchall()
        if args.format == "json":
            print(json.dumps({"query_id": args.query_id, "query_only": True,
                              "columns": columns,
                              "rows": [dict(zip(columns, row)) for row in rows]}, indent=2))
        else:
            writer = csv.writer(sys.stdout)
            writer.writerow(columns)
            writer.writerows(rows)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
