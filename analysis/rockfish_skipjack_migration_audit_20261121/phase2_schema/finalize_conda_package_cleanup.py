#!/usr/bin/env python3
"""Close the synthetic cache-tree location and connect execution evidence to its recipe."""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
DATA = AUDIT / "phase2_data"
DB = AUDIT / "provenance_catalog.sqlite"
BATCH = "RF-DATA-CONDA-PKG-LIBSTDCXX-UNUSED-20261122-R01"
RUN = "RF-CONDAPKG-LIBSTDCXX-14_2-H8F9B012-REGEN-RECIPE-20261122"
POSTCHECK = DATA / "conda_package_cleanup_postcheck_20261122.json"
LOCAL_ROOT = "LOCAL-MAIN"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    post = json.loads(POSTCHECK.read_text(encoding="utf-8"))
    if post.get("candidate_exists") or post.get("remaining_matching_entries"):
        raise RuntimeError("Remote post-check found a remaining package cache path")
    observed = post["utc"].replace("+00:00", "Z")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=120000")
    try:
        con.execute("BEGIN IMMEDIATE")
        batch = con.execute("SELECT status FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH,)).fetchone()
        receipt = con.execute(
            """SELECT cer.execution_state,s.source_id FROM cleanup_execution_receipt cer
               JOIN registry_source s ON s.source_id=cer.receipt_source_id
               WHERE cer.cleanup_batch_id=? AND s.source_key=?""",
            (BATCH, "cleanup_execution_receipt_" + BATCH),
        ).fetchone()
        if batch != ("COMPLETED",) or not receipt or receipt[0] != "COMPLETED":
            raise RuntimeError("Cleanup receipt is not complete in the catalog")
        verified = con.execute(
            """SELECT count(*),sum(pre_delete_size_bytes),sum(deletion_state='VERIFIED')
               FROM cleanup_item WHERE cleanup_batch_id=? AND disposition='DELETE'""", (BATCH,)
        ).fetchone()
        if verified != (90, 14223129, 90):
            raise RuntimeError("Catalog deletion items are not all verified")

        root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (LOCAL_ROOT,)).fetchone()[0]
        rel = POSTCHECK.resolve().relative_to(Path(root)).as_posix()
        digest = sha(POSTCHECK)
        source_key = "conda_package_cleanup_postcheck_20261122"
        prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (source_key,)).fetchone()
        if prior and tuple(prior) != (str(POSTCHECK.resolve()), digest):
            raise RuntimeError("Post-check source key points at changed evidence")
        con.execute(
            """INSERT OR IGNORE INTO registry_source
               (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note)
               VALUES(?,?,'CONDA_CACHE_POST_DELETE_CHECK',?,?,?,1,?)""",
            (source_key, str(POSTCHECK.resolve()), digest, now, observed,
             "Rockfish confirmed the exact candidate directory is absent and no package-name prefix remains in its cache parent."),
        )
        source_id = con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (source_key,)).fetchone()[0]
        con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (LOCAL_ROOT, rel, now))
        path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (LOCAL_ROOT, rel)).fetchone()[0]
        artifact_key = "CONDAPKG-EVIDENCE:" + source_key
        con.execute(
            """INSERT OR IGNORE INTO artifact
               (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
                scientific_status,retention_class,notes)
               VALUES('HPC_ENVIRONMENTS',?,'CONDA_CACHE_POSTCHECK','Remote path absence receipt',
                      'USER_CONFIRMED',?,?,'LINEAGE_SUPPORTED','PERMANENT_SOURCE',?)""",
            (artifact_key, digest, POSTCHECK.stat().st_size, "Exact post-delete absence check for batch " + BATCH),
        )
        post_artifact, post_hash = con.execute("SELECT artifact_id,sha256 FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
        if post_hash != digest:
            raise RuntimeError("Post-check artifact hash mismatch")
        con.execute(
            """INSERT OR IGNORE INTO artifact_location
               (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
               VALUES(?,?,'PRESENT',?,?)""", (post_artifact, path_id, observed, observed)
        )

        cache = con.execute(
            """SELECT a.artifact_id,al.path_id,al.availability
               FROM artifact a JOIN artifact_location al USING(artifact_id)
               WHERE a.artifact_key='CONDAPKG-CACHE-TREE:libstdcxx-14.2.0-h8f9b012_2'"""
        ).fetchone()
        if not cache or cache[2] != "PRESENT":
            raise RuntimeError("Expected the historical cache-tree location to be present before finalization")
        con.execute("UPDATE artifact_location SET availability='REMOVED',last_verified_utc=? WHERE artifact_id=? AND path_id=?", (observed, cache[0], cache[1]))

        receipt_artifact = con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", ("CLEANUP-EXECUTION-RECEIPT:" + BATCH,)).fetchone()[0]
        for artifact_id, role in ((receipt_artifact, "CLEANUP_EXECUTION_RECEIPT"), (post_artifact, "POST_DELETE_ABSENCE_CHECK")):
            con.execute(
                """INSERT OR IGNORE INTO run_artifact
                   (run_id,artifact_id,role,direction,required_for_reproduction,notes)
                   VALUES(?,?,?,'PROVENANCE',0,?)""",
                (RUN, artifact_id, role, "Cleanup evidence attached to the pinned Conda cache-regeneration record."),
            )
        con.execute(
            """INSERT OR IGNORE INTO derivation_edge
               (parent_artifact_id,child_artifact_id,run_id,operation,tool_name,script_artifact_id,
                parameter_json,provenance_status,notes)
               VALUES(?,? ,?,'CONDA_PACKAGE_EXTRACT','conda',?,?,'LINEAGE_SUPPORTED',?)""",
            (con.execute("SELECT artifact_id FROM artifact WHERE artifact_key='CONDAPKG-REMOTE:libstdcxx-14.2.0-h8f9b012_2'").fetchone()[0],
             cache[0], RUN,
             con.execute("SELECT script_artifact_id FROM reproduction_recipe WHERE run_id=? AND recipe_version='2026-11-22-r01'", (RUN,)).fetchone()[0],
             json.dumps({"name": "libstdcxx", "version": "14.2.0", "build": "h8f9b012_2", "channel": "conda-forge/linux-64"}, sort_keys=True),
             "Cached package metadata records the pinned archive URL and SHA-256. Archive bytes were not independently downloaded for a bytewise extraction comparison; the recipe is a future regeneration method."),
        )
        if con.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Foreign-key check failed")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    print(json.dumps({"batch": BATCH, "status": "COMPLETED", "cache_tree_location": "REMOVED",
                      "postcheck_source": source_key, "postcheck_sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
