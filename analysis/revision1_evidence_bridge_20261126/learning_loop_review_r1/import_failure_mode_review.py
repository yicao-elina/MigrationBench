#!/usr/bin/env python3
"""Build a pinned SQLite review index for the failure-mode audit register."""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "FAILURE_MODE_REGISTER.tsv"
EXPECTED_SHA256 = "e0024cba419441891c93eb8806f8363e2cdcc85edc7bcb5547d15df1f7b45fa7"
EXPECTED_COLUMNS = (
    "failure_mode_id", "evidence_family", "observed_signal", "queryable_sources", "lineage_status",
    "repeatability_assessment", "next_round_decision_and_gain", "recommended_closure",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-db", required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    source, output = Path(args.input_tsv), Path(args.output_db)
    digest = sha256_file(source)
    if digest != EXPECTED_SHA256:
        raise ValueError("failure register SHA mismatch: {}".format(digest))
    with open(str(source), "r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError("failure register header differs from pinned eight-column contract")
        source_rows = list(reader)
    if len(source_rows) != 7 or len({r["failure_mode_id"] for r in source_rows}) != 7:
        raise ValueError("expected 7 unique audit-summary rows")
    if output.exists() and not args.replace:
        raise IOError("output exists; choose another path or pass --replace")
    if output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "snapshot_id": "failure_mode_review_20261126_r1",
        "capture_date_utc": "2026-11-26",
        "evidence_grain": "one row per evidence family in a read-only audit register; not one row per observed event/run",
        "source_register_tsv_sha256": digest,
        "source_register_tsv_bytes": str(source.stat().st_size),
        "source_audit_readme_sha256": "e2aa70eec10d38b237177872088e753d954f85c2a78dc861b052e6f21be222b2",
        "catalog_commit": "0e3afed611ad2e502280346c5e5b1f72ce3e7e56",
        "row_count": str(len(source_rows)),
        "scope_warning": "summaries do not establish detector calibration, a decision-to-training link, or active-learning benefit",
        "schema_version": "1",
    }
    connection = sqlite3.connect(str(output))
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.executescript((HERE / "schema.sql").read_text())
        connection.executemany("INSERT INTO snapshot_metadata VALUES (?,?)", sorted(metadata.items()))
        for r in source_rows:
            values = tuple(r[c] for c in EXPECTED_COLUMNS) + ("AUDIT_SUMMARY_NOT_FAILURE_EVENT", "NOT_ESTABLISHED_IN_AUDIT", json.dumps(r, sort_keys=True, ensure_ascii=False))
            connection.execute("INSERT INTO failure_mode_review VALUES ("+",".join("?" for _ in values)+")", values)
        connection.commit()
        connection.execute("VACUUM")
        check = connection.execute("PRAGMA quick_check").fetchone()[0]
        count = connection.execute("SELECT count(*) FROM failure_mode_review").fetchone()[0]
        if check != "ok" or count != 7:
            raise ValueError("validation failed: quick_check={}, rows={}".format(check, count))
    finally:
        connection.close()
    print(json.dumps({"output_path":str(output),"bytes":output.stat().st_size,"sha256":sha256_file(output),"source_tsv_sha256":digest,"rows":7,"quick_check":"ok","foreign_key_violations":0},indent=2,sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
