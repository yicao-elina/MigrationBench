#!/usr/bin/env python3
"""Build an immutable, source-pinned SQLite copy of the R1 claim inventory."""
from __future__ import print_function
import argparse
import csv
import hashlib
import json
import os
import sqlite3
import stat
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PACKAGE = REPO / "analysis" / "manuscript_claim_evidence_matrix_revision1_20261125"
BRIDGE = HERE.parent
TSV = PACKAGE / "MIGRATIONBENCH_R1_CLAIM_EVIDENCE_MATRIX_20261125.tsv"
PIN_MANIFEST = PACKAGE / "manifest.json"
SCHEMA = HERE / "schema.sql"
GENERATOR = PACKAGE / "build_matrix.py"
COLUMNS = [
    "matrix_id", "claim_type", "source_document", "source_location", "source_object_id",
    "assertion_or_panel", "exact_source_quote_or_value", "numeric_tokens", "referenced_labels",
    "sql_item_id", "sql_panel", "registry_content", "artifact_pointer", "artifact_sha256",
    "generator_or_query", "registry_status", "reproduction_status", "lineage_status", "gap_reason",
    "source_sha256",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_equal(label, path, expected):
    actual = sha256(path)
    if actual.lower() != str(expected).lower():
        raise ValueError("{} SHA-256 mismatch: expected {}, found {}".format(label, expected, actual))
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-path", required=True, help="read-only registry SQLite file matching manifest pin")
    parser.add_argument("--output", required=True, help="new SQLite snapshot path; refuses to replace an existing file")
    parser.add_argument("--archive-path", help="stable deployment path recorded in the manifest and database metadata")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    registry = Path(args.registry_path).expanduser().resolve()
    archive_path = args.archive_path or str(output)
    if output.exists():
        raise ValueError("refusing to replace existing output: {}".format(output))
    if not registry.is_file():
        raise IOError("registry is not a regular file: {}".format(registry))

    pinned = json.loads(PIN_MANIFEST.read_text(encoding="utf-8"))
    input_hashes = pinned["pinned_inputs_sha256"]
    for relative, expected in input_hashes.items():
        path = registry if relative.endswith("migrationbench_asset_registry.sqlite") else REPO / relative
        if not path.is_file():
            raise IOError("pinned input is missing: {} (override registry with --registry-path)".format(path))
        check_equal(relative, path, expected)
    output_hash = check_equal("claim matrix TSV", TSV, pinned["output_sha256"])
    generator_hash = check_equal("claim matrix generator", GENERATOR, pinned["generator_sha256"])
    schema_hash = check_equal("claim matrix schema", PACKAGE / "schema.json", pinned["schema_sha256"])
    local_schema_hash = sha256(SCHEMA)

    con = sqlite3.connect(str(registry.resolve().as_uri()) + "?mode=ro", uri=True)
    try:
        check = con.execute("PRAGMA quick_check").fetchone()[0]
        violations = con.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        con.close()
    if check != "ok" or violations:
        raise ValueError("registry integrity failure: quick_check={}, foreign_key_violations={}".format(check, len(violations)))

    with open(str(TSV), "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != COLUMNS:
            raise ValueError("TSV columns differ from pinned schema")
        rows = list(reader)
    if len(rows) != int(pinned["row_count"]):
        raise ValueError("row count mismatch: expected {}, found {}".format(pinned["row_count"], len(rows)))
    if len(set(row["matrix_id"] for row in rows)) != int(pinned["unique_matrix_ids"]):
        raise ValueError("matrix IDs are not unique")
    type_counts = Counter(row["claim_type"] for row in rows)
    status_counts = Counter(row["lineage_status"] for row in rows)
    if dict(type_counts) != pinned["claim_type_counts"] or dict(status_counts) != pinned["lineage_status_counts"]:
        raise ValueError("claim type or lineage-status counts differ from pinned manifest")

    output.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(output))
    try:
        db.executescript(SCHEMA.read_text(encoding="utf-8"))
        metadata = {
            "package": pinned["package"],
            "source_generation_commit": pinned["source_commit"],
            "submitted_source_baseline": pinned["submitted_source_baseline"],
            "claim_matrix_sha256": output_hash,
            "claim_matrix_rows": str(len(rows)),
            "claim_type_counts_json": json.dumps(dict(sorted(type_counts.items())), sort_keys=True),
            "lineage_status_counts_json": json.dumps(dict(sorted(status_counts.items())), sort_keys=True),
            "generator_path": str((PACKAGE / pinned["generator"]).relative_to(REPO)),
            "generator_sha256": generator_hash,
            "source_schema_path": str((PACKAGE / pinned["schema"]).relative_to(REPO)),
            "source_schema_sha256": schema_hash,
            "sqlite_schema_sha256": local_schema_hash,
            "registry_path_at_build": str(BRIDGE / "FEDERATED_SOURCES.json"),
            "registry_source_uri": "/scratch16/pclancy3/yi/MigrationBench_data_archive/provenance_catalog_snapshots/20261124/provenance_catalog_20261124.sqlite",
            "registry_sha256": input_hashes[next(k for k in input_hashes if k.endswith("migrationbench_asset_registry.sqlite"))],
            "registry_quick_check": check,
            "registry_foreign_key_violations": str(len(violations)),
            "scope": "639 source-index rows across assertion segments, figure/table objects, numeric cells, and reviewer questions; not scientific validation",
            "grain_boundary": "one source inventory row; prose segmentation is lexical; source IDs remain local",
            "status_semantics": "preserves TRACEABLE/PARTIAL/MISSING from the committed inventory; TRACEABLE is not scientific acceptance",
        }
        db.executemany("INSERT INTO snapshot_metadata(metadata_key,metadata_value) VALUES(?,?)", sorted(metadata.items()))
        placeholders = ",".join("?" for _ in COLUMNS)
        db.executemany(
            "INSERT INTO claim_evidence_row({}) VALUES({})".format(",".join(COLUMNS), placeholders),
            [tuple(row[column] for column in COLUMNS) for row in rows],
        )
        db.commit()
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("new snapshot quick_check failed")
        if db.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("new snapshot has foreign-key violations")
        if db.execute("SELECT COUNT(*) FROM claim_evidence_row").fetchone()[0] != len(rows):
            raise ValueError("new snapshot row count failed")
        db.close()
    except Exception:
        db.close()
        try:
            output.unlink()
        except OSError:
            pass
        raise
    os.chmod(str(output), stat.S_IRUSR | stat.S_IWUSR)
    snapshot = {
        "snapshot": "MigrationBench Revision 1 claim evidence matrix SQLite sidecar",
        "user_version": 1,
        "source_generation_commit": pinned["source_commit"],
        "submitted_source_baseline": pinned["submitted_source_baseline"],
        "database_path": archive_path,
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "quick_check": "ok",
        "foreign_key_violations": 0,
        "row_count": len(rows),
        "claim_type_counts": dict(sorted(type_counts.items())),
        "lineage_status_counts": dict(sorted(status_counts.items())),
        "source_tsv_sha256": output_hash,
        "generator_sha256": generator_hash,
        "source_schema_sha256": schema_hash,
        "sqlite_schema_sha256": local_schema_hash,
        "registry_sha256": input_hashes[next(k for k in input_hashes if k.endswith("migrationbench_asset_registry.sqlite"))],
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "This is artifact-lineage indexing, not scientific validation.",
            "Source-native IDs are not central-catalog foreign keys.",
            "The source commit and submitted source baseline remain distinct historical pins.",
        ],
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(str(manifest_path), stat.S_IRUSR | stat.S_IWUSR)
    print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (IOError, OSError, sqlite3.Error, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.stderr.write("claim matrix import failed: {}\n".format(exc))
        sys.exit(2)
