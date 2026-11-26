#!/usr/bin/env python3
"""Build a pinned SQLite view of the submitted Revision 1 Figure 5 data."""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE.parents[2] / "paper/revision1/data_processed/current_release/fig5_corrected_input.tsv"
EXPECTED_SHA256 = "ee9d43cbb4f6141fb50730a073b25d0a71f6963379ab54614ebe4b17a5019ead"
EXPECTED_COLUMNS = (
    "row_id", "row_label", "model_kind", "training_seed", "column_id", "column_label", "task_block",
    "metric_type", "raw_value_eV", "predicted_value_eV", "error_eV", "absolute_error_eV", "n_available",
    "n_expected", "coverage_fraction", "aggregation", "status_summary", "source_artifact", "model_sha256",
    "scientific_status", "publication_status", "reference_value_eV", "error_definition", "color_value_eV",
    "color_vmin_eV", "color_vmax_eV", "color_transform", "color_scale", "family", "strategy",
    "dft_reference_eV", "source_status", "matrix_status", "dft_reference_display_eV", "correction_record_id",
)
TEXT_FIELDS = set(EXPECTED_COLUMNS) - {
    "training_seed", "raw_value_eV", "predicted_value_eV", "error_eV", "absolute_error_eV", "n_available",
    "n_expected", "coverage_fraction", "reference_value_eV", "color_value_eV", "color_vmin_eV",
    "color_vmax_eV", "dft_reference_eV", "dft_reference_display_eV",
}
INTEGER_FIELDS = {"n_available", "n_expected"}
REAL_FIELDS = {"training_seed", "raw_value_eV", "predicted_value_eV", "error_eV", "absolute_error_eV", "coverage_fraction", "reference_value_eV", "color_value_eV", "color_vmin_eV", "color_vmax_eV", "dft_reference_eV", "dft_reference_display_eV"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse(row):
    values = []
    for key in EXPECTED_COLUMNS:
        value = row[key]
        if value == "":
            values.append(None)
        elif key in INTEGER_FIELDS:
            values.append(int(value))
        elif key in REAL_FIELDS:
            values.append(float(value))
        else:
            values.append(value)
    values.append(json.dumps(row, sort_keys=True, ensure_ascii=False))
    return tuple(values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-db", required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    source = Path(args.input_tsv)
    output = Path(args.output_db)
    digest = sha256_file(source)
    if digest != EXPECTED_SHA256:
        raise ValueError("Figure 5 TSV SHA mismatch: {}".format(digest))
    with open(str(source), "r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError("Figure 5 TSV header differs from pinned 35-column contract")
        source_rows = list(reader)
    rows = [parse(row) for row in source_rows]
    model_count = len({row[0] for row in rows})
    task_count = len({row[4] for row in rows})
    if len(rows) != 120 or model_count != 15 or task_count != 8 or len({(row[0], row[4]) for row in rows}) != 120:
        raise ValueError("expected complete 15×8 Figure 5 source cells; found rows={}, models={}, tasks={}".format(len(rows), model_count, task_count))
    if output.exists() and not args.replace:
        raise IOError("output exists; choose another path or pass --replace")
    if output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "snapshot_id": "mb15_fig5_matrix_20261126_r1",
        "capture_date_utc": "2026-11-26",
        "evidence_grain": "15 submitted model rows × 8 Figure 5 task cells = 120 source records; not a complete model-by-task evaluation matrix",
        "source_tsv_path": "paper/revision1/data_processed/current_release/fig5_corrected_input.tsv",
        "source_tsv_sha256": digest,
        "source_tsv_bytes": str(source.stat().st_size),
        "source_commit": "8e50d5adb56087ce388a1aa593150d14fd70c66f",
        "model_count": str(model_count),
        "task_count": str(task_count),
        "row_count": str(len(rows)),
        "status_warning": "publication and scientific statuses are copied from source; nonconverged and diagnostic/supporting rows are retained",
        "reuse_warning": "model SHA establishes checkpoint identity only; rows do not expose definitive raw attempt/job/trajectory links",
        "schema_version": "1",
    }
    connection = sqlite3.connect(str(output))
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.executescript((HERE / "schema.sql").read_text())
        connection.executemany("INSERT INTO snapshot_metadata VALUES (?,?)", sorted(metadata.items()))
        marks = ",".join("?" for _ in range(len(EXPECTED_COLUMNS)+1))
        connection.executemany("INSERT INTO fig5_matrix_cell VALUES ("+marks+")", rows)
        connection.commit()
        connection.execute("VACUUM")
        check = connection.execute("PRAGMA quick_check").fetchone()[0]
        count = connection.execute("SELECT count(*) FROM fig5_matrix_cell").fetchone()[0]
        fk = connection.execute("PRAGMA foreign_key_check").fetchall()
        if check != "ok" or count != 120 or fk:
            raise ValueError("validation failed: quick_check={}, rows={}, foreign_key_violations={}".format(check,count,len(fk)))
        statuses = connection.execute("SELECT publication_status,count(*) FROM fig5_matrix_cell GROUP BY publication_status ORDER BY publication_status").fetchall()
    finally:
        connection.close()
    print(json.dumps({"output_path":str(output),"bytes":output.stat().st_size,"sha256":sha256_file(output),"source_tsv_sha256":digest,"models":model_count,"tasks":task_count,"rows":120,"publication_status_counts":statuses,"quick_check":"ok","foreign_key_violations":0},indent=2,sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
