#!/usr/bin/env python3
"""Build the model-grain MB15/DUAL-X coverage SQLite snapshot."""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "MD15_DUALX_COVERAGE.tsv"
EXPECTED_INPUT_SHA256 = "293d9de146ef7cf80d6922c519a48fc3e7674b86c718a40f753db9ccdce9f6d9"
EXPECTED_COLUMNS = (
    "mb_model_id", "family", "training_seed_or_variant", "md_status_r02", "reason",
    "last_step", "last_time_ps", "target_step", "raw_frames", "eligible_frames",
    "invalid_frames", "model_sha256", "input_sha256", "evidence_root",
    "figure_row_present", "dualx_model_id_exact_sha", "dualx_registry_exact_model_ids",
    "dualx_r4_model_exact_ids", "dualx_external_link_pass_rows",
    "dualx_primary_inference_runs", "dualx_primary_gradcam_runs",
    "mb_exact_static_candidate_rows", "mb_exact_neb_rows", "reuse_class",
)
INTEGER_COLUMNS = {
    "last_step", "target_step", "raw_frames", "eligible_frames", "invalid_frames",
    "dualx_external_link_pass_rows", "dualx_primary_inference_runs",
    "dualx_primary_gradcam_runs", "mb_exact_static_candidate_rows", "mb_exact_neb_rows",
}

METADATA = {
    "capture_date_utc": "2026-11-26",
    "evidence_grain": "one row per submitted Revision 1 MD model (15 rows); not a run/attempt table and not a model-by-task matrix",
    "md_status_source_commit": "fba92f7b1650893ef9aeeef160f76a516009118f",
    "md_status_source_sha256": "89f32a3e5f6587e8ca3652e3b65ca6249d4891601e46fe74fa08ac336c0dd9c3",
    "figure_data_source_sha256": "929c15b6b46d80b4c69afb9a6b6d6e34acf229c684dfcec2957f65e7a571a62e",
    "md_protocol_manifest_sha256": "316a8f31cd20a0026c2b7312b894d8ad1c2503d62b4a37472371c285b95eda88",
    "model_identity_bridge_sha256": "cca5aaa8774cc4436e9ca4e89c55fce22a25c34482a59a399154fc81b8ec8c89",
    "dualx_bridge_snapshot_manifest_sha256": "e3ccbb801d26804318cc4399d10cc766688f81183e9734d5704b433253cdd214",
    "dualx_v14_sqlite_sha256": "a7e46f4bdf5ae134ab501b308dbfa987348d4a15201503297edd540c954e6fa4",
    "mb_r4_sqlite_sha256": "d033f47913eaf33d61a2ce7b0b0281b90b385d23190890a97f477e2f901896d7",
    "scope_warning": "checkpoint SHA identity does not prove a selected run, attempt, trajectory, processed value, figure/table cell, or scientific acceptance",
    "reuse_warning": "static candidate rows, path-specific NEB rows, historical MD status rows, inference runs, and Grad-CAM runs are separate evidence types; counts are not a combined validation score",
    "schema_version": "1",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_value(column, value):
    if value == "":
        return None
    if column in INTEGER_COLUMNS:
        return int(value)
    if column == "last_time_ps":
        return float(value)
    return value


def load_rows(path):
    with open(str(path), "r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError("coverage TSV header differs from the pinned 24-column contract")
        rows = []
        for source_row in reader:
            rows.append(tuple(parse_value(column, source_row[column]) for column in EXPECTED_COLUMNS))
    if len(rows) != 15:
        raise ValueError("expected 15 model-grain rows, found {}".format(len(rows)))
    if len(set(row[0] for row in rows)) != 15:
        raise ValueError("mb_model_id is not unique")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-db", required=True)
    parser.add_argument("--replace", action="store_true", help="replace an existing output snapshot")
    args = parser.parse_args(argv)

    input_path = Path(args.input_tsv)
    output_path = Path(args.output_db)
    input_sha = sha256_file(input_path)
    if input_sha != EXPECTED_INPUT_SHA256:
        raise ValueError("input SHA-256 mismatch: expected {}, found {}".format(EXPECTED_INPUT_SHA256, input_sha))
    rows = load_rows(input_path)
    if output_path.exists() and not args.replace:
        raise IOError("output exists; choose a new path or pass --replace: {}".format(output_path))
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = dict(METADATA)
    metadata["coverage_tsv_sha256"] = input_sha
    metadata["coverage_tsv_bytes"] = str(input_path.stat().st_size)
    metadata["model_row_count"] = str(len(rows))

    connection = sqlite3.connect(str(output_path))
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA page_size=4096")
        connection.executescript((HERE / "schema.sql").read_text())
        connection.executemany(
            "INSERT INTO snapshot_metadata(metadata_key, metadata_value) VALUES (?, ?)",
            sorted(metadata.items()),
        )
        marks = ", ".join("?" for _ in EXPECTED_COLUMNS)
        connection.executemany(
            "INSERT INTO mb15_model_coverage ({}) VALUES ({})".format(
                ", ".join(EXPECTED_COLUMNS), marks
            ), rows,
        )
        connection.commit()
        connection.execute("VACUUM")
        check = connection.execute("PRAGMA quick_check").fetchone()[0]
        if check != "ok":
            raise ValueError("quick_check failed: " + str(check))
        actual_count = connection.execute("SELECT count(*) FROM mb15_model_coverage").fetchone()[0]
        if actual_count != 15:
            raise ValueError("database row count mismatch: {}".format(actual_count))
        user_version = connection.execute("PRAGMA user_version").fetchone()[0]
        if user_version != 1:
            raise ValueError("unexpected user_version: {}".format(user_version))
    finally:
        connection.close()

    print(json.dumps({
        "output_path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "source_tsv_bytes": input_path.stat().st_size,
        "source_tsv_sha256": input_sha,
        "rows": len(rows),
        "sqlite_user_version": 1,
        "quick_check": "ok",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
