#!/usr/bin/env python3
"""Build the pinned Scratch16 depth-3 directory metadata snapshot."""

import argparse
import csv
import hashlib
import os
import sqlite3
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(HERE, "schema.sql")
EXPECTED_ROOT = "/scratch16/pclancy3/yi"
SNAPSHOT_ID = "scratch16_depth3_20261126_r1"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parent_for(path):
    if path == ".":
        return None
    parent = os.path.dirname(path.rstrip("/"))
    return parent if parent else "."


def validate_path(path, depth):
    if path.startswith("/") or ".." in path.split("/"):
        raise ValueError("non-relative or parent-traversal path: " + path)
    actual_depth = 0 if path == "." else len(path.split("/"))
    if actual_depth != depth or depth > 3:
        raise ValueError("path depth mismatch: " + path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_tsv")
    parser.add_argument("classification_tsv")
    parser.add_argument("output_sqlite")
    parser.add_argument("--captured-utc", default="2026-11-26")
    args = parser.parse_args()

    tsv_hash = sha256_file(args.input_tsv)
    classification_hash = sha256_file(args.classification_tsv)
    records = []
    with open(args.input_tsv, "r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = ["relative_path", "depth", "direct_child_directories",
                    "direct_regular_files", "direct_symlinks", "direct_other",
                    "direct_logical_bytes", "scan_error"]
        if reader.fieldnames != expected:
            raise ValueError("unexpected inventory columns")
        for row in reader:
            path = row["relative_path"]
            depth = int(row["depth"])
            validate_path(path, depth)
            records.append((path, parent_for(path), depth,
                            int(row["direct_child_directories"]),
                            int(row["direct_regular_files"]),
                            int(row["direct_symlinks"]), int(row["direct_other"]),
                            int(row["direct_logical_bytes"]), row["scan_error"]))

    paths = {row[0] for row in records}
    if "." not in paths or len(paths) != len(records):
        raise ValueError("root row missing or duplicate path present")
    for row in records:
        if row[1] is not None and row[1] not in paths:
            raise ValueError("parent row missing for " + row[0])
    errors = sum(bool(row[8]) for row in records)
    classifications = []
    with open(args.classification_tsv, "r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = ["name", "entry_type", "logical_bytes", "allocated_kib_du",
                    "mtime_utc", "uid", "gid", "mode_octal", "symlink_target",
                    "name_based_project_group", "classification_basis", "classification_note"]
        if reader.fieldnames != expected:
            raise ValueError("unexpected classification columns")
        for row in reader:
            if row["entry_type"] == "d":
                validate_path(row["name"], 1)
                classifications.append((row["name"], row["name_based_project_group"],
                                        row["classification_basis"], row["classification_note"]))
    if len(classifications) != 222 or {row[0] for row in classifications} != {
            row[0] for row in records if row[2] == 1}:
        raise ValueError("classification set does not exactly match depth-1 directories")

    if os.path.exists(args.output_sqlite):
        os.unlink(args.output_sqlite)
    connection = sqlite3.connect(args.output_sqlite)
    try:
        connection.executescript(open(SCHEMA, "r").read())
        connection.executemany(
            "INSERT INTO directory_entry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            sorted(records, key=lambda row: (row[2], row[0])))
        connection.executemany(
            "INSERT INTO directory_project_classification VALUES (?, ?, ?, ?)",
            sorted(classifications))
        connection.execute(
            "INSERT INTO inventory_snapshot VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (SNAPSHOT_ID, args.captured_utc, EXPECTED_ROOT, 3, len(records),
             tsv_hash, classification_hash, errors,
             "Depth-limited metadata only; direct logical bytes sum immediate regular-file and symlink sizes, not recursive storage or quota."))
        connection.commit()
        connection.execute("PRAGMA optimize")
        connection.commit()
    finally:
        connection.close()
    print("rows={} scan_errors={} tsv_sha256={}".format(len(records), errors, tsv_hash))


if __name__ == "__main__":
    main()
