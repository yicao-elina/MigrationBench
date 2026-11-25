#!/usr/bin/env python3
"""Normalize legacy WSe2 and post-hoc Conda cleanup evidence into migration 006."""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
WSE2 = AUDIT / "phase2_scratch4" / "wse2_wfc_delete_receipt_20261122.jsonl"
CONDA = AUDIT / "phase2_data" / "conda_index_cache_removed_manifest_20261122.tsv"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    events = [json.loads(row) for row in WSE2.read_text(encoding="utf-8").splitlines() if row.strip()]
    preflight = next(row for row in events if row.get("event") == "preflight_passed")
    deleted = [row for row in events if row.get("event") == "deleted_verified"]
    complete = next(row for row in events if row.get("event") == "complete")
    if preflight.get("queue_overlaps") != 0 or len(deleted) != 8 or complete.get("file_count") != 8:
        raise SystemExit("WSe2 evidence is not a complete zero-overlap eight-file receipt")
    if any(row.get("verified_missing") is not True for row in deleted):
        raise SystemExit("WSe2 receipt contains an unverified path")
    wse2_logical = sum(int(row["size_bytes"]) for row in deleted)
    if wse2_logical != int(complete["bytes"]):
        raise SystemExit("WSe2 per-file logical sizes differ from completion summary")

    with CONDA.open(encoding="utf-8", newline="") as stream:
        conda_rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(conda_rows) != 42 or any(row["sha256_predelete"] != "NOT_CAPTURED" for row in conda_rows):
        raise SystemExit("Conda evidence is not the expected 42-row post-hoc manifest")
    conda_logical = sum(int(row["size_bytes"]) for row in conda_rows)
    conda_allocated = sum(int(row["allocated_bytes"]) for row in conda_rows)
    observed = "2026-11-22T22:54:57Z"

    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=120000")
    try:
        con.execute("BEGIN IMMEDIATE")
        wse2_hash = digest(WSE2)
        wse2_key = "wse2_wfc_execution_receipt_20261122"
        con.execute(
            """INSERT OR IGNORE INTO registry_source
               (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note)
               VALUES(?,?,'CLEANUP_EXECUTION_RECEIPT_JSONL',?,?,?,1,?)""",
            (wse2_key, str(WSE2.resolve()), wse2_hash, complete["utc"], complete["utc"],
             "Guarded WSe2 WFC cleanup; includes zero-overlap queue preflight, eight pre-delete hashes, and eight verified-missing results."),
        )
        wse2_source_id = con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (wse2_key,)).fetchone()[0]

        for row in deleted:
            prefix = "/scratch4/pclancy3/yi/"
            if not row["path"].startswith(prefix):
                raise SystemExit("WSe2 receipt path outside reviewed user subtree")
            relative = row["path"][len(prefix):]
            inventory = con.execute(
                """SELECT logical_bytes FROM inventory_observation io
                   JOIN storage_path sp USING(path_id)
                   WHERE sp.root_id='RF-SCRATCH4-YI' AND sp.relative_path=?
                     AND io.snapshot_id='RF-SCRATCH4-YI-20261122-R01'""",
                (relative,),
            ).fetchone()
            if not inventory or inventory[0] != int(row["size_bytes"]):
                raise SystemExit("WSe2 allocation source inventory mismatch for " + relative)
        wse2_quota_before = con.execute(
            "SELECT used_bytes,files_used FROM quota_observation WHERE quota_observation_id=3"
        ).fetchone()
        wse2_quota_after = con.execute(
            "SELECT used_bytes,files_used FROM quota_observation WHERE quota_observation_id=6"
        ).fetchone()
        wse2_allocated = int(wse2_quota_before[0]) - int(wse2_quota_after[0])
        if wse2_allocated != 5045174272 or wse2_quota_before[1] - wse2_quota_after[1] != 8:
            raise SystemExit(f"Unexpected WSe2 quota bracket: allocated_delta={wse2_allocated}")

        conda_source_id = con.execute(
            "SELECT source_id FROM registry_source WHERE source_key='rockfish_conda_index_cache_posthoc_manifest_20261122'"
        ).fetchone()[0]
        conda_group_before = con.execute(
            """SELECT quota_observation_id FROM quota_observation
               WHERE raw_output_path LIKE '%rockfish_quota_after_wse2_20261122.txt'
                 AND root_id='RF-DATA-FS' AND quota_subject_kind='GROUP' AND quota_subject='pclancy3'"""
        ).fetchone()[0]
        conda_group_after = con.execute(
            """SELECT quota_observation_id FROM quota_observation
               WHERE raw_output_path LIKE '%rockfish_quota_pre_cleanup_batch1_20261122.txt'
                 AND root_id='RF-DATA-FS' AND quota_subject_kind='GROUP' AND quota_subject='pclancy3'"""
        ).fetchone()[0]

        records = [
            ("RF-S4-WSE2-WFC-CLEANUP-20261122", wse2_source_id, complete["utc"], "COMPLETED", 8,
             wse2_logical, wse2_allocated, 3, 6,
             "Legacy guarded receipt normalized: per-file hashes and missing-path checks passed; zero live queue overlaps. The file manifest lacks per-file allocated blocks, so actual allocated bytes are the bracketing group-quota delta; group file count fell by exactly eight."),
            ("RF-DATA-CONDA-INDEX-CACHE-20261122", conda_source_id, observed, "PARTIAL", 42,
             conda_logical, conda_allocated, conda_group_before, conda_group_after,
             "Post-hoc cleanup normalized as PARTIAL. Conda 4.8.3 removed the index cache during a command labelled dry-run; R01 metadata and later absence are recorded, but no pre-delete hashes, live queue preflight, or contemporaneous quota bracket exist. Quota IDs 5 and 9 are broad observations and do not causally attribute the interval delta."),
        ]
        for row in records:
            if con.execute("SELECT 1 FROM cleanup_execution_receipt WHERE cleanup_batch_id=?", (row[0],)).fetchone():
                continue
            con.execute(
                """INSERT INTO cleanup_execution_receipt
                   (cleanup_batch_id,receipt_source_id,observed_utc,execution_state,deleted_file_count,
                    actual_logical_bytes,actual_allocated_bytes,quota_before_observation_id,quota_after_observation_id,notes)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                row,
            )
        if con.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Foreign-key check failed")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    with sqlite3.connect(DB) as verify:
        print("integrity_check=" + verify.execute("PRAGMA integrity_check").fetchone()[0])
        print("foreign_key_violations=" + str(len(verify.execute("PRAGMA foreign_key_check").fetchall())))
        for row in verify.execute(
            """SELECT cleanup_batch_id,execution_state,deleted_file_count,actual_logical_bytes,
                      actual_allocated_bytes,quota_before_observation_id,quota_after_observation_id
               FROM cleanup_execution_receipt WHERE cleanup_batch_id IN
                 ('RF-S4-WSE2-WFC-CLEANUP-20261122','RF-DATA-CONDA-INDEX-CACHE-20261122')
               ORDER BY cleanup_batch_id"""
        ):
            print("|".join(str(value) if value is not None else "NULL" for value in row))


if __name__ == "__main__":
    main()
