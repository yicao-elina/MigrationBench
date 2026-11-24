#!/usr/bin/env python3
"""Record the index-cache removal observed during the old Conda dry-run attempt."""

import csv
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
DATA = AUDIT / "phase2_data"
DB = AUDIT / "provenance_catalog.sqlite"
MANIFEST = DATA / "conda_index_cache_removed_manifest_20261122.tsv"
COMMAND_EVIDENCE = DATA / "conda_index_clean_dryrun_20261122.txt"
VERIFY_EVIDENCE = DATA / "conda_index_cache_removed_verification_20261122.txt"
BATCH = "RF-DATA-CONDA-INDEX-CACHE-20261122"
OBSERVED_UTC = "2026-11-22T19:31:47Z"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def register_source(con, key, path, kind, note):
    absolute = str(path.resolve())
    digest = sha256(path)
    old = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (key,)).fetchone()
    if old and old != (absolute, digest):
        raise RuntimeError("source changed: " + key)
    con.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note)
           VALUES (?,?,?,?,?,?,1,?)""",
        (key, absolute, kind, digest, OBSERVED_UTC, OBSERVED_UTC, note),
    )
    return digest


def main():
    rows = list(csv.DictReader(MANIFEST.open(newline=""), delimiter="\t"))
    if len(rows) != 42 or any(r["uid"] != "3901" or r["gid"] != "3001" or r["sha256_predelete"] != "NOT_CAPTURED" for r in rows):
        raise SystemExit("manifest count, ownership, or hash-scope guard failed")
    logical = sum(int(r["size_bytes"]) for r in rows)
    allocated = sum(int(r["allocated_bytes"]) for r in rows)
    verification = VERIFY_EVIDENCE.read_text()
    if "cache_dir_exists=0" not in verification:
        raise SystemExit("post-command path verification is not affirmative")
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    try:
        con.execute("BEGIN IMMEDIATE")
        if con.execute("SELECT 1 FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH,)).fetchone():
            raise SystemExit("batch already exists")
        register_source(con, "rockfish_conda_index_cache_posthoc_manifest_20261122", MANIFEST,
                        "POSTHOC_CLEANUP_PATH_MANIFEST", "Pre-delete metadata is from the complete R01 inventory; pre-delete content hashes were not captured.")
        register_source(con, "rockfish_conda_index_cache_clean_attempt_20261122", COMMAND_EVIDENCE,
                        "CONDA_CLEAN_DRYRUN_TRANSCRIPT", "Conda 4.8.3 `conda clean -i -d -vvv` reported dry-run mode but emitted an rm_rf trace; current path verification shows the cache directory was removed.")
        register_source(con, "rockfish_conda_index_cache_postdelete_check_20261122", VERIFY_EVIDENCE,
                        "POST_DELETE_PATH_VERIFICATION", "At the capture time, the index-cache directory did not exist; all descendants in the manifest are therefore absent.")
        con.execute(
            """INSERT INTO cleanup_batch
               (cleanup_batch_id,created_utc,source_root_id,policy,rationale,status,authorization_note,expected_bytes,actual_bytes,completed_utc)
               VALUES (?,?,?,?,?,'COMPLETED',?,?,?,?)""",
            (BATCH, OBSERVED_UTC, "RF-DATA-YI", "REMOVE_REGENERABLE_CONDA_INDEX_CACHE",
             "Conda 4.8.3 index-cache directory disappeared during a command documented as dry-run. The 42 prior file paths and metadata are reconstructed from R01; no content hash was captured before removal.",
             "User authorized systematic cleanup of regenerable caches. This batch is logged post hoc because the installed cleaner removed files despite `-d/--dry-run`; no broader Conda cache deletion was executed.",
             logical, allocated, OBSERVED_UTC),
        )
        con.execute("INSERT INTO artifact(project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes) VALUES(NULL,?,'CLEANUP_RECEIPT','verified post hoc Conda index-cache removal manifest','USER_CONFIRMED',?,?, 'VERIFIED','PERMANENT_SOURCE',?)",
                    ("RF-DATA-CONDA-INDEX-CACHE-RECEIPT-20261122", sha256(MANIFEST), MANIFEST.stat().st_size,
                     "Lists 42 former index-cache file paths with R01 metadata, owner IDs, and explicit missing pre-delete content hashes."))
        receipt_artifact = con.execute("SELECT last_insert_rowid()").fetchone()[0]
        local_root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id='LOCAL-MAIN'").fetchone()[0]
        rel_receipt = str(MANIFEST.resolve().relative_to(Path(local_root)))
        con.execute("INSERT INTO storage_path(root_id,relative_path,first_seen_utc) VALUES('LOCAL-MAIN',?,?)", (rel_receipt, OBSERVED_UTC))
        receipt_path = con.execute("SELECT last_insert_rowid()").fetchone()[0]
        con.execute("INSERT INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)", (receipt_artifact, receipt_path, OBSERVED_UTC, OBSERVED_UTC))

        for item in rows:
            absolute = item["path"]
            prefix = "/data/pclancy3/yi/"
            if not absolute.startswith(prefix):
                raise SystemExit("candidate path outside registered data subtree")
            relative = absolute[len(prefix):]
            row = con.execute(
                """SELECT a.artifact_id, a.owner_status, sp.path_id, io.logical_bytes, io.allocated_bytes, io.mtime_ns,
                          io.uid, io.gid, io.link_count
                   FROM storage_path sp JOIN artifact_location l USING(path_id) JOIN artifact a USING(artifact_id)
                   JOIN inventory_observation io USING(path_id)
                   WHERE sp.root_id='RF-DATA-YI' AND sp.relative_path=?
                     AND io.snapshot_id='RF-DATA-YI-20261122-R01'""",
                (relative,),
            ).fetchone()
            if row is None or (row["logical_bytes"], row["allocated_bytes"], row["mtime_ns"]) != (int(item["size_bytes"]), int(item["allocated_bytes"]), int(item["mtime_ns"])):
                raise SystemExit("candidate does not match R01 metadata: " + relative)
            owner = con.execute(
                """SELECT uid,gid,link_count FROM inventory_observation io JOIN storage_path sp USING(path_id)
                   WHERE sp.root_id='RF-DATA-YI' AND sp.relative_path=?
                     AND io.snapshot_id='RF-DATA-PCLANCY3-UID3901-20261122-R01'""",
                (relative,),
            ).fetchone()
            if owner is None or (owner["uid"], owner["gid"], owner["link_count"]) != (3901, 3001, 1):
                raise SystemExit("owner/link-count guard failed: " + relative)
            con.execute("UPDATE artifact SET owner_status='USER_CONFIRMED',retention_class='CACHE',notes=notes || ' Classified as regenerable Conda index cache; removed during conda clean dry-run anomaly.' WHERE artifact_id=?", (row["artifact_id"],))
            con.execute(
                """INSERT INTO cleanup_item
                   (cleanup_batch_id,artifact_id,disposition,decision_reason,owner_clearance,pre_delete_sha256,
                    pre_delete_size_bytes,pre_delete_mtime_ns,deletion_state,deleted_utc,actual_bytes,notes)
                   VALUES (?,?,'DELETE',?,'CLEARED',NULL,?,?, 'VERIFIED',NULL,?,?)""",
                (BATCH, row["artifact_id"], "Regenerable Conda index-cache entry; directory absent after cleaner invocation.",
                 int(item["size_bytes"]), int(item["mtime_ns"]), int(item["allocated_bytes"]),
                 "Post hoc record. Pre-delete hash and exact unlink time were not captured; owner/link/size/mtime are from R01 and current directory absence was verified."),
            )
            con.execute("UPDATE artifact_location SET availability='REMOVED',last_verified_utc=? WHERE artifact_id=? AND path_id=?", (OBSERVED_UTC, row["artifact_id"], row["path_id"]))
        con.execute("UPDATE cleanup_batch SET evidence_artifact_id=? WHERE cleanup_batch_id=?", (receipt_artifact, BATCH))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    with sqlite3.connect(DB) as check:
        print("integrity_check=" + check.execute("PRAGMA integrity_check").fetchone()[0])
        print("foreign_key_violations=" + str(len(check.execute("PRAGMA foreign_key_check").fetchall())))
        print("batch=" + repr(check.execute("SELECT status,expected_bytes,actual_bytes,completed_utc,evidence_artifact_id FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH,)).fetchone()))
        print("items=" + repr(check.execute("SELECT count(*),sum(pre_delete_size_bytes),sum(actual_bytes),sum(pre_delete_sha256 IS NOT NULL) FROM cleanup_item WHERE cleanup_batch_id=?", (BATCH,)).fetchone()))


if __name__ == "__main__":
    main()
