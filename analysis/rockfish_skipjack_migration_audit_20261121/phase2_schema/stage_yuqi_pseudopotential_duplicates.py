#!/usr/bin/env python3
"""Register canonical pseudo files and stage 65 verified duplicate-copy removals."""

import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
DATA = AUDIT / "phase2_data"
DB = AUDIT / "provenance_catalog.sqlite"
IDENTITY = DATA / "former_collab_file_identity_20261122.tsv"
SEARCH = DATA / "former_collab_inventory_candidate_search_extended_20261122.tsv"
HOME_LEDGER = DATA / "home_pseudo_inventory_20261122.jsonl"
BATCH = "RF-DATA-FOR-YUQI-PSEUDO-DUPLICATES-20261122"
OBSERVED = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def register_source(con, key, path, kind, note):
    digest = sha256(path)
    absolute = str(path.resolve())
    prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (key,)).fetchone()
    if prior and prior != (absolute, digest):
        raise RuntimeError("source key collision: " + key)
    con.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note)
           VALUES (?,?,?,?,?,?,1,?)""",
        (key, absolute, kind, digest, OBSERVED, OBSERVED, note),
    )
    return digest


def main():
    identity_rows = list(csv.DictReader(IDENTITY.open(newline=""), delimiter="\t"))
    targets = [r for r in identity_rows if r["path"].startswith("/data/pclancy3/for-yuqi/pseudo/")]
    home = [json.loads(line) for line in HOME_LEDGER.read_text().splitlines() if line.strip()]
    home_entries = {Path(r["path"]).name: r for r in home if r.get("record") == "file"}
    end = next(r for r in home if r.get("record") == "scan_end")
    if len(targets) != 65 or end["file_count"] != 66 or end.get("directory_count") != 1:
        raise SystemExit("expected 65 former-collaborator copies and 66 canonical home files")
    if any(r.get("record") == "excluded" for r in home):
        raise SystemExit("home pseudo subtree contains unexpected non-regular entries")
    for row in targets:
        name = Path(row["path"]).name
        canonical = home_entries.get(name)
        if not canonical or row["exact_match"] != "True" or row["sha256"] != row["canonical_sha256"] or canonical["sha256"] != row["sha256"] or int(canonical["size_bytes"]) != int(row["size_bytes"]):
            raise SystemExit("canonical duplicate identity mismatch: " + name)
        if row["nlink"] != "1" or row["uid"] != "3901" or row["gid"] != "3001":
            raise SystemExit("target ownership/link-count guard failed: " + name)
    logical = sum(int(r["size_bytes"]) for r in targets)
    allocated = sum(int(r["allocated_bytes"]) for r in targets)
    identity_hash = sha256(IDENTITY)
    search_rows = list(csv.DictReader(SEARCH.open(newline=""), delimiter="\t"))
    if any("for-yuqi" in str(r).lower() or "for-victor" in str(r).lower() for r in search_rows):
        raise SystemExit("reference search TSV contains a positive path hit; review before staging")
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    try:
        con.execute("BEGIN IMMEDIATE")
        if con.execute("SELECT 1 FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH,)).fetchone():
            raise SystemExit("cleanup batch already exists")
        register_source(con, "former_collab_identity_hashes_20261122", IDENTITY,
                        "TSV_SHA256_DUPLICATE_IDENTITY_AUDIT", "65/65 UPF copies match same-basename `/home/ycao73/pseudo` content by SHA-256 and size.")
        register_source(con, "former_collab_bounded_reference_search_20261122", SEARCH,
                        "BOUNDED_SOURCE_REFERENCE_SEARCH", "Searched 5,656 inventoried source/config/manifest candidates; zero for-yuqi/for-victor literal path hits. Scope limits in accompanying review note.")
        register_source(con, "home_pseudo_inventory_20261122", HOME_LEDGER,
                        "JSONL_HOME_PSEUDOPOTENTIAL_INVENTORY", "Direct-child metadata and SHA-256 inventory of all 66 files under `/home/ycao73/pseudo`.")
        con.execute(
            """INSERT INTO inventory_snapshot
               (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
                files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
                manifest_path,manifest_sha256,notes)
                   VALUES (?,?,?,?,?,'inventory_home_pseudo.py','PARTIAL',?,?,0,0,?,?,?,?,?)""",
            ("RF-HOME-YI-PSEUDO-20261122-R01", "RF-HOME-YI",
             datetime.fromtimestamp(next(x["started_epoch"] for x in home if x["record"] == "scan_start"), timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
             datetime.fromtimestamp(end["finished_epoch"], timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
             "Direct-child os.scandir/lstat plus full-file SHA-256; nonrecursive pseudo subtree scope", end["file_count"], end["directory_count"], end["logical_bytes"],
             end["allocated_bytes"], str(HOME_LEDGER.resolve()), sha256(HOME_LEDGER),
             "PARTIAL recursive coverage: all 66 direct files were hashed and the `PAW` child directory was identified but not traversed. This is a scoped child-directory snapshot, not the full home root."),
        )
        for row in (r for r in home if r.get("record") == "directory"):
            name = Path(row["path"]).name
            rel = "pseudo/" + name
            con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES('RF-HOME-YI',?,?)", (rel, OBSERVED))
            path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id='RF-HOME-YI' AND relative_path=?", (rel,)).fetchone()[0]
            con.execute(
                """INSERT INTO inventory_observation
                   (snapshot_id,path_id,entry_type,uid,gid,mode_octal,allocated_bytes,mtime_ns,inode,device,hash_scope,observation_status,link_count)
                   VALUES ('RF-HOME-YI-PSEUDO-20261122-R01',?,'DIRECTORY',?,?,?,?,?,?,?,'NONE','PRESENT',?)""",
                (path_id,row["uid"],row["gid"],row["mode_octal"],row["allocated_bytes"],row["mtime_ns"],str(row["inode"]),str(row["device"]),row["nlink"]),
            )
        for row in (r for r in home if r.get("record") == "file"):
            name = Path(row["path"]).name
            rel = "pseudo/" + name
            con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES('RF-HOME-YI',?,?)", (rel, OBSERVED))
            path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id='RF-HOME-YI' AND relative_path=?", (rel,)).fetchone()[0]
            con.execute(
                """INSERT INTO inventory_observation
                   (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,mtime_ns,inode,device,
                    sha256,hash_scope,observation_status,link_count)
                   VALUES ('RF-HOME-YI-PSEUDO-20261122-R01',?, 'FILE',?,?,?,?,?,?,?,?,?,'FULL_FILE','PRESENT',?)""",
                (path_id, row["uid"], row["gid"], row["mode_octal"], row["size_bytes"], row["allocated_bytes"], row["mtime_ns"],
                 str(row["inode"]), str(row["device"]), row["sha256"], row["nlink"]),
            )
            key = "RF-HOME-YI:" + rel
            con.execute(
                """INSERT OR IGNORE INTO artifact
                   (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
                   VALUES ('QE_PSEUDOPOTENTIALS',?,'FILE','QE pseudopotential','USER_CONFIRMED',?,?,'VERIFIED','REPRODUCIBLE_INPUT','Canonical home pseudopotential; SHA-256 verified.')""",
                (key, row["sha256"], row["size_bytes"]),
            )
            art_id = con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (key,)).fetchone()[0]
            con.execute("UPDATE artifact SET sha256=?,byte_size=?,owner_status='USER_CONFIRMED',retention_class='REPRODUCIBLE_INPUT',semantic_role='QE pseudopotential',scientific_status='VERIFIED' WHERE artifact_id=?", (row["sha256"], row["size_bytes"], art_id))
            con.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES (?,?,'PRESENT',?,?)", (art_id, path_id, OBSERVED, OBSERVED))
        con.execute(
            """INSERT INTO cleanup_batch
               (cleanup_batch_id,created_utc,source_root_id,policy,rationale,status,authorization_note,expected_bytes,actual_bytes,evidence_artifact_id)
               VALUES (?,?,?,?,?,'EVIDENCE_READY',?,?,NULL,NULL)""",
            (BATCH, OBSERVED, "RF-DATA-FOR-YUQI", "DELETE_SHA256_VERIFIED_DUPLICATE_PSEUDOPOTENTIALS",
             "65 user-owned UPFs are byte-exact duplicates of canonical files under `/home/ycao73/pseudo`; bounded source/config/manifest search found no literal references to the old paths.",
             "Explicit user instruction: for-yuqi is handled by the same cleanup pipeline as other data; its former-collaborator status is not a blanket retention exception. Final live hash/owner/link/queue preflight is still required.",
             logical),
        )
        for row in targets:
            name = Path(row["path"]).name
            target_rel = "pseudo/" + name
            canonical_rel = "pseudo/" + name
            target = con.execute("SELECT a.artifact_id,sp.path_id FROM artifact a JOIN artifact_location l USING(artifact_id) JOIN storage_path sp USING(path_id) WHERE a.artifact_key=?", ("RF-DATA-FOR-YUQI:" + target_rel,)).fetchone()
            canonical = con.execute("SELECT a.artifact_id FROM artifact a JOIN artifact_location l USING(artifact_id) JOIN storage_path sp USING(path_id) WHERE a.artifact_key=?", ("RF-HOME-YI:" + canonical_rel,)).fetchone()
            if target is None or canonical is None:
                raise SystemExit("canonical or duplicate artifact is not registered: " + name)
            actual = con.execute("SELECT sha256,byte_size,owner_status FROM artifact WHERE artifact_id=?", (target["artifact_id"],)).fetchone()
            if actual["sha256"] not in (None, row["sha256"]) or actual["byte_size"] != int(row["size_bytes"]):
                raise SystemExit("target artifact differs from identity evidence: " + name)
            con.execute("UPDATE artifact SET sha256=?,owner_status='USER_CONFIRMED',retention_class='DUPLICATE_COPY',scientific_status='VERIFIED',semantic_role='verified duplicate QE pseudopotential copy',notes=notes || ' Exact SHA-256 duplicate of canonical home path; staged for copy deletion.' WHERE artifact_id=?", (row["sha256"], target["artifact_id"]))
            con.execute("UPDATE artifact SET sha256=?,owner_status='USER_CONFIRMED',retention_class='REPRODUCIBLE_INPUT',scientific_status='VERIFIED' WHERE artifact_id=?", (row["canonical_sha256"], canonical["artifact_id"]))
            con.execute("INSERT INTO duplicate_group(detection_run_id,match_method,verification_status,disposition,rationale) VALUES('former-collab-dedup-20261122','SHA256','VERIFIED_IDENTICAL','DELETE_COPY',?)", ("Same basename and size; SHA-256 matches at both paths. Canonical path retained in /home; former-collaborator copy is user-owned and has nlink=1.",))
            group_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
            con.execute("INSERT INTO duplicate_member(duplicate_group_id,artifact_id,canonical_member) VALUES(?,?,1)", (group_id, canonical["artifact_id"]))
            con.execute("INSERT INTO duplicate_member(duplicate_group_id,artifact_id,canonical_member) VALUES(?,?,0)", (group_id, target["artifact_id"]))
            con.execute(
                """INSERT INTO cleanup_item
                   (cleanup_batch_id,artifact_id,disposition,decision_reason,owner_clearance,pre_delete_sha256,
                    pre_delete_size_bytes,pre_delete_mtime_ns,deletion_state,notes)
                   VALUES (?,?,'DELETE',?,'CLEARED',?,?,?,'EVIDENCE_READY',?)""",
                (BATCH, target["artifact_id"], "Byte-exact duplicate of registered canonical QE pseudopotential under /home/ycao73/pseudo.",
                 row["sha256"], int(row["size_bytes"]), int(row["mtime_ns"]),
                 "Awaiting final live SHA-256, metadata, queue, and path-absence checks."),
            )
        # Register the identity manifest as the staged batch evidence artifact.
        manifest_key = "RF-DATA-FOR-YUQI-PSEUDO-IDENTITY-MANIFEST-20261122"
        con.execute("INSERT INTO artifact(project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes) VALUES ('QE_PSEUDOPOTENTIALS',?,'AUDIT_MANIFEST','duplicate identity and canonical path ledger','USER_CONFIRMED',?,?, 'VERIFIED','PERMANENT_SOURCE','Hashes and canonical counterpart paths for former-collaborator pseudopotential duplicates.')", (manifest_key, identity_hash, IDENTITY.stat().st_size))
        evidence_artifact = con.execute("SELECT last_insert_rowid()").fetchone()[0]
        local_root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id='LOCAL-MAIN'").fetchone()[0]
        relative = str(IDENTITY.resolve().relative_to(Path(local_root)))
        con.execute("INSERT INTO storage_path(root_id,relative_path,first_seen_utc) VALUES('LOCAL-MAIN',?,?)", (relative, OBSERVED))
        path_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
        con.execute("INSERT INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)", (evidence_artifact, path_id, OBSERVED, OBSERVED))
        con.execute("UPDATE cleanup_batch SET evidence_artifact_id=? WHERE cleanup_batch_id=?", (evidence_artifact, BATCH))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    with sqlite3.connect(DB) as check:
        print("integrity_check=" + check.execute("PRAGMA integrity_check").fetchone()[0])
        print("foreign_key_violations=" + str(len(check.execute("PRAGMA foreign_key_check").fetchall())))
        print("batch=" + repr(check.execute("SELECT status,expected_bytes,actual_bytes,evidence_artifact_id FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH,)).fetchone()))
        print("items=" + repr(check.execute("SELECT count(*),sum(pre_delete_size_bytes),sum(pre_delete_sha256 IS NOT NULL) FROM cleanup_item WHERE cleanup_batch_id=?", (BATCH,)).fetchone()))
        print("duplicate_groups=" + str(check.execute("SELECT count(*) FROM duplicate_group WHERE detection_run_id='former-collab-dedup-20261122'").fetchone()[0]))


if __name__ == "__main__":
    main()
