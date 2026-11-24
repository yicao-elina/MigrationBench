#!/usr/bin/env python3
"""Link duplicate decisions to the immutable hash and inode evidence sources."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
LINEAGE = AUDIT / "phase2_scratch4" / "wse2_run_lineage.tsv"
LOCAL_ROOT = "LOCAL-MAIN"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    manifest_time = "2026-11-22T18:21:53Z"
    digest = sha256(LINEAGE)
    mappings: list[tuple[str, str, str]] = []

    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=120000")
    try:
        con.execute("BEGIN IMMEDIATE")
        if not con.execute("SELECT 1 FROM schema_migration WHERE version=8").fetchone():
            raise RuntimeError("Migration 008 must be applied before linking duplicate evidence")

        prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key='wse2_run_lineage_hashes_20261122'").fetchone()
        if prior and prior != (str(LINEAGE.resolve()), digest):
            raise RuntimeError("WSe2 hash-lineage source key points to changed content")
        con.execute(
            """INSERT OR IGNORE INTO registry_source
               (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note)
               VALUES(?,?,'WSE2_XML_HASH_LINEAGE',?,?,?,?,?)""",
            ("wse2_run_lineage_hashes_20261122", str(LINEAGE.resolve()), digest, now, manifest_time, 1,
             "Per-run WSe2 lineage ledger records the same SHA-256 for the top-level XML and QE save XML duplicate pairs."),
        )
        lineage_source = con.execute("SELECT source_id FROM registry_source WHERE source_key='wse2_run_lineage_hashes_20261122'").fetchone()[0]
        project = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (LOCAL_ROOT,)).fetchone()[0]
        relative = LINEAGE.resolve().relative_to(Path(project)).as_posix()
        con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (LOCAL_ROOT, relative, now))
        path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (LOCAL_ROOT, relative)).fetchone()[0]
        artifact_key = "WSE2-XML-LINEAGE:" + LINEAGE.name
        con.execute(
            """INSERT OR IGNORE INTO artifact
               (artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
               VALUES(?,'HASH_LINEAGE_LEDGER','WSe2 output XML duplicate hash evidence','USER_CONFIRMED',?,?,'VERIFIED','PERMANENT_SOURCE',?)""",
            (artifact_key, digest, LINEAGE.stat().st_size, "Per-run XML path/hash pairs used for WSe2 duplicate-group decisions."),
        )
        artifact_id, artifact_hash = con.execute("SELECT artifact_id,sha256 FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
        if artifact_hash != digest:
            raise RuntimeError("WSe2 hash-lineage artifact mismatch")
        con.execute(
            """INSERT OR IGNORE INTO artifact_location
               (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
               VALUES(?,?,'PRESENT',?,?)""",
            (artifact_id, path_id, now, now),
        )

        source_keys = {
            "wse2": lineage_source,
            "former": con.execute("SELECT source_id FROM registry_source WHERE source_key='former_collab_identity_hashes_20261122'").fetchone()[0],
            "victor_hash": con.execute("SELECT source_id FROM registry_source WHERE source_key='for_victor_live_duplicate_pairs_20261122'").fetchone()[0],
            "victor_manifest": con.execute("SELECT source_id FROM registry_source WHERE source_key='for_victor_cleanup_manifest_20261122'").fetchone()[0],
            "large_hash": con.execute("SELECT source_id FROM registry_source WHERE source_key='data_yi_gt500mb_hash_receipt_20261122'").fetchone()[0],
        }
        groups = con.execute("SELECT duplicate_group_id,detection_run_id FROM duplicate_group ORDER BY duplicate_group_id").fetchall()
        if len(groups) != 79:
            raise RuntimeError(f"Expected 79 duplicate groups, found {len(groups)}")
        for group_id, detection in groups:
            if detection.startswith("RF-SCRATCH4-WSE2-"):
                rows = [(source_keys["wse2"], "HASH_COMPARISON", "Per-run WSe2 ledger contains matching hashes for the paired QE XML paths.")]
            elif detection == "former-collab-dedup-20261122":
                rows = [(source_keys["former"], "HASH_COMPARISON", "Former-collaborator exact-identity audit records each byte-identical path pair.")]
            elif detection == "for-victor-cyqe-example-duplicate-check-20261122":
                rows = [
                    (source_keys["victor_hash"], "HASH_COMPARISON", "Live SHA-256 comparison of each Victor duplicate and its retained canonical file."),
                    (source_keys["victor_manifest"], "CLEANUP_MANIFEST", "Exact file-level removal manifest for verified Victor copies."),
                ]
            elif detection == "RF-DATA-YI-LARGE-HASH-20261122-R01":
                role = "INODE_EVIDENCE" if group_id == 79 else "HASH_COMPARISON"
                rows = [(source_keys["large_hash"], role, "Large-file hash receipt records content hashes, device/inode, and link count for the duplicate set.")]
            else:
                raise RuntimeError("No evidence mapping for duplicate detection run " + detection)
            for source_id, role, note in rows:
                con.execute(
                    """INSERT OR IGNORE INTO duplicate_group_evidence
                       (duplicate_group_id,source_id,evidence_role,notes) VALUES(?,?,?,?)""",
                    (group_id, source_id, role, note),
                )
                mappings.append((str(group_id), str(source_id), role))

        if con.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Foreign-key check failed")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    with sqlite3.connect(DB) as verify:
        print("duplicate_groups=" + str(verify.execute("SELECT count(DISTINCT duplicate_group_id) FROM duplicate_group_evidence").fetchone()[0]))
        print("evidence_links=" + str(verify.execute("SELECT count(*) FROM duplicate_group_evidence").fetchone()[0]))
        print("foreign_key_violations=" + str(len(verify.execute("PRAGMA foreign_key_check").fetchall())))


if __name__ == "__main__":
    main()
