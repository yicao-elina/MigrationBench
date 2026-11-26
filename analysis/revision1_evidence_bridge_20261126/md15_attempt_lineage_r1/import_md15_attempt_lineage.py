#!/usr/bin/env python3
"""Build a normalized SQLite snapshot from the pinned MD15 lineage audit TSV."""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "MD15_ATTEMPT_LINEAGE_R01.tsv"
EXPECTED_SHA256 = "b2dc4b6ae0936b3b338ae1e212a37510239e43e58250c8a33a39174516f446cc"
R02_SHA256 = "89f32a3e5f6587e8ca3652e3b65ca6249d4891601e46fe74fa08ac336c0dd9c3"
EXPECTED_COLUMNS = (
    "model_id", "family", "display_label", "training_seed_or_variant", "last_step", "last_time_ps",
    "target_step", "status", "reason", "raw_frames", "eligible_frames", "invalid_frames",
    "finite_temperature", "finite_pressure", "metrics_core_finite", "model_sha256", "input_sha256",
    "evidence_root", "registry_checkpoint_path", "checkpoint_current_sha256", "checkpoint_size_bytes",
    "checkpoint_hash_match", "candidate_run_dir", "run_dir_exists", "smoke_dir",
    "trajectory_candidates_path_and_bytes", "metrics_candidate_path", "metrics_bytes", "run_status_record",
    "continuation_ledger_evidence", "slurm_job_evidence", "attempt_id_evidence", "raw_trajectory_sha256",
    "missing_links",
)
INTEGER_FIELDS = {"last_step", "target_step", "raw_frames", "eligible_frames", "invalid_frames", "checkpoint_size_bytes", "metrics_bytes"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nullable_int(value):
    return None if value in (None, "", "NOT_CHECKED", "NOT_APPLICABLE") else int(value)


def state(value):
    if value in ("", "NOT_APPLICABLE"):
        return "NOT_APPLICABLE"
    if value in ("NOT_CHECKED",):
        return "NOT_CHECKED"
    if value in ("NO", "NOT_FOUND", "MISSING"):
        return "ABSENT"
    return "PRESENT"


def add_artifact(rows, model_id, role, path, observed_state, size=None, digest=None, match="NOT_CHECKED", note="direct audit TSV"):
    if not path or path in ("NOT_APPLICABLE", "NOT_CHECKED"):
        return
    rows.append((model_id, role, path, observed_state, size, digest, match, note))


def load_rows(path):
    with open(str(path), "r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError("input columns differ from pinned 34-column contract")
        rows = list(reader)
    if len(rows) != 15 or len({row["model_id"] for row in rows}) != 15:
        raise ValueError("expected exactly 15 unique model rows")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-db", required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    source = Path(args.input_tsv)
    output = Path(args.output_db)
    source_hash = sha256_file(source)
    if source_hash != EXPECTED_SHA256:
        raise ValueError("source SHA-256 mismatch: {}".format(source_hash))
    rows = load_rows(source)
    if output.exists() and not args.replace:
        raise IOError("output exists; choose a new path or pass --replace")
    if output.exists():
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "snapshot_id": "md15_attempt_lineage_20261126_r1",
        "capture_date_utc": "2026-11-26",
        "evidence_grain": "one audit row per submitted R02 model; candidate job IDs are not confirmed attempt/run links",
        "source_attempt_audit_tsv_sha256": source_hash,
        "source_attempt_audit_tsv_bytes": str(source.stat().st_size),
        "source_r02_status_tsv_sha256": R02_SHA256,
        "source_audit_readme_sha256": "05277984e17773bdefa631f0fcbbbaca633129ab09de8de8df5a12c768408149",
        "source_remote_stat_hash_receipt_sha256": "a96c4a60836a29d42bdfffdb7b2e9f13d9f582053d801fdbfa3715ab3565095e",
        "model_row_count": str(len(rows)),
        "scope": "read-only scan limited to paths under /scratch16/pclancy3/yi; the MPA-0 checkpoint under /data was not checked",
        "limitations": "no attempt UUIDs; candidate scheduler IDs are not definitive attempt links; raw trajectory SHA-256 is absent from producer manifests",
        "schema_version": "1",
    }
    connection = sqlite3.connect(str(output))
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.executescript((HERE / "schema.sql").read_text())
        connection.executemany("INSERT INTO snapshot_metadata VALUES (?, ?)", sorted(metadata.items()))
        for row in rows:
            model_id = row["model_id"]
            raw_json = json.dumps(row, sort_keys=True, ensure_ascii=False)
            model_values = (
                model_id, row["family"], row["display_label"], row["training_seed_or_variant"],
                nullable_int(row["last_step"]), float(row["last_time_ps"]) if row["last_time_ps"] else None,
                nullable_int(row["target_step"]), row["status"], row["reason"] or None,
                nullable_int(row["raw_frames"]), nullable_int(row["eligible_frames"]), nullable_int(row["invalid_frames"]),
                row["model_sha256"], row["input_sha256"], row["candidate_run_dir"] or None,
                row["run_dir_exists"], row["smoke_dir"] or None, row["missing_links"], raw_json,
            )
            connection.execute(
                "INSERT INTO model_attempt_audit VALUES (" + ",".join("?" for _ in model_values) + ")", model_values)

            artifacts = []
            add_artifact(artifacts, model_id, "registered_full_run_dir", row["candidate_run_dir"], state(row["run_dir_exists"]), note="path registration/presence from scoped audit")
            add_artifact(artifacts, model_id, "smoke_dir", row["smoke_dir"], "PRESENT" if row["smoke_dir"] not in ("", "NOT_APPLICABLE", "NOT_CHECKED") else state(row["smoke_dir"]))
            add_artifact(artifacts, model_id, "checkpoint", row["registry_checkpoint_path"], state(row["checkpoint_hash_match"]), nullable_int(row["checkpoint_size_bytes"]), row["checkpoint_current_sha256"] if re.fullmatch(r"[0-9a-f]{64}", row["checkpoint_current_sha256"]) else None, row["checkpoint_hash_match"], "manifest/registry identity check; MPA-0 outside inspected scope")
            add_artifact(artifacts, model_id, "metrics_csv", row["metrics_candidate_path"], "PRESENT" if row["metrics_candidate_path"] else "ABSENT", nullable_int(row["metrics_bytes"]))
            add_artifact(artifacts, model_id, "status_json", row["run_status_record"], "PRESENT" if row["run_status_record"] else "ABSENT")
            continuation = row["continuation_ledger_evidence"]
            add_artifact(artifacts, model_id, "continuation_ledger", continuation, "PRESENT" if continuation and continuation not in ("NOT_FOUND", "NOT_APPLICABLE") else ("NOT_APPLICABLE" if continuation == "NOT_APPLICABLE" else "ABSENT"))
            for raw in re.split(r"\s*\|\s*", row["trajectory_candidates_path_and_bytes"]):
                match = re.match(r"(.+?)\s+\[bytes=(\d+)\]$", raw.strip())
                if match:
                    trajectory_path, size = match.group(1), int(match.group(2))
                    add_artifact(artifacts, model_id, "trajectory_segment", trajectory_path, "PRESENT", size, note="size from direct stat; trajectory SHA not recorded")
                    segment_match = re.search(r"segment_from_(\d+)\.extxyz$", trajectory_path)
                    connection.execute("INSERT INTO trajectory_segment(model_id,evidence_path,parsed_start_step,reported_bytes,sha256_status) VALUES (?,?,?,?,?)", (model_id, trajectory_path, int(segment_match.group(1)) if segment_match else None, size, "NOT_RECORDED_IN_PRODUCER_MANIFEST"))
            connection.executemany("INSERT INTO evidence_artifact(model_id,evidence_role,evidence_path,observed_state,reported_bytes,observed_sha256,hash_match,source_note) VALUES (?,?,?,?,?,?,?,?)", artifacts)

            job_text = row["slurm_job_evidence"]
            for job_id in sorted(set(re.findall(r"(?<!\d)(\d{7,})(?!\d)", job_text))):
                connection.execute("INSERT INTO scheduler_job_candidate(model_id,job_id,role_note,linkage_status,source_evidence) VALUES (?,?,?,?,?)", (model_id, job_id, "role detail preserved in raw source row; do not treat as definitive attempt join", "CANDIDATE_FROM_SCOPED_LOG", job_text))
            for gap in re.split(r"\s*;\s*", row["missing_links"].strip()):
                if gap:
                    connection.execute("INSERT OR IGNORE INTO missing_link(model_id,gap_text) VALUES (?,?)", (model_id, gap))
        connection.commit()
        connection.execute("VACUUM")
        check = connection.execute("PRAGMA quick_check").fetchone()[0]
        fk = connection.execute("PRAGMA foreign_key_check").fetchall()
        count = connection.execute("SELECT count(*) FROM model_attempt_audit").fetchone()[0]
        if check != "ok" or fk or count != 15:
            raise ValueError("snapshot validation failed: quick_check={}, fk={}, model_rows={}".format(check, len(fk), count))
    finally:
        connection.close()
    print(json.dumps({"output_path": str(output), "bytes": output.stat().st_size, "sha256": sha256_file(output), "source_tsv_sha256": source_hash, "model_rows": 15, "quick_check": "ok", "foreign_key_violations": 0}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
