#!/usr/bin/env python3
"""Import R04's full-hash duplicate review into the provenance catalog.

The importer is deliberately evidence-only: it does not modify Rockfish files,
remove paths, or change any retention decision. It can be rerun safely.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


AUDIT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB = AUDIT_DIR / "provenance_catalog.sqlite"
DEFAULT_GROUPS = (
    AUDIT_DIR
    / "phase2_data"
    / "yi_full_inventory_20261123_r04"
    / "large_duplicate_content_hash_groups_r04_20261123.tsv"
)
ROOT_ID = "RF-DATA-YI"
SCOPED_PATH = "/data/pclancy3/yi"
SOURCE_KEY = "R04-YI-LARGE-EXACT-DUPLICATE-HASH-REVIEW-20261123"
DETECTION_PREFIX = "RF-DATA-YI-LARGE-DUP-R04-20261123"
CAPTURED_UTC = "2026-11-23T11:04:43Z"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_groups(path: Path) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            paths = json.loads(row["paths_json"])
            if len(paths) != int(row["paths"]):
                raise ValueError(f"{row['group_id']}: paths count mismatch")
            if int(row["unique_inodes"]) != len(paths):
                raise ValueError(f"{row['group_id']}: expected distinct inodes")
            if len(set(paths)) != len(paths):
                raise ValueError(f"{row['group_id']}: duplicate path entry")
            groups.append({**row, "paths": paths})
    return groups


def append_note(existing: str, addition: str) -> str:
    if addition in existing:
        return existing
    return (existing.rstrip() + " " + addition).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--captured-utc", default=CAPTURED_UTC)
    args = parser.parse_args()

    db_path = args.db.resolve()
    groups_path = args.groups.resolve()
    source_hash = sha256_file(groups_path)
    groups = read_groups(groups_path)
    if len(groups) != 16:
        raise ValueError(f"Expected 16 verified groups, found {len(groups)}")

    con = sqlite3.connect(db_path, timeout=60)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    try:
        con.execute("BEGIN IMMEDIATE")
        root = con.execute(
            "SELECT scoped_path FROM storage_root WHERE root_id=?", (ROOT_ID,)
        ).fetchone()
        if not root or root["scoped_path"] != SCOPED_PATH:
            raise ValueError(f"Unexpected or missing root mapping: {root}")

        existing_source = con.execute(
            "SELECT source_id,source_path,sha256 FROM registry_source WHERE source_key=?",
            (SOURCE_KEY,),
        ).fetchone()
        source_path = str(groups_path)
        if existing_source:
            if existing_source["sha256"] != source_hash:
                raise ValueError("Registered TSV hash differs from current source")
            if existing_source["source_path"] != source_path:
                raise ValueError("Registered TSV path differs from current source")
            source_id = int(existing_source["source_id"])
        else:
            cur = con.execute(
                """INSERT INTO registry_source
                   (source_key,source_path,registry_kind,sha256,captured_utc,
                    as_of_utc,immutable_snapshot,authority_note,notes)
                   VALUES (?,?,?,?,?,?,1,?,?)""",
                (
                    SOURCE_KEY,
                    source_path,
                    "HASH_REVIEW_TSV",
                    source_hash,
                    datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
                        "+00:00", "Z"
                    ),
                    args.captured_utc,
                    "R04 metadata-guarded full-file SHA-256 review; source records exact paths, sizes, allocations, hashes, and duplicate groups.",
                    "Companion report: phase2_data/yi_full_inventory_20261123_r04/LARGE_DUPLICATE_CONTENT_HASH_REVIEW_R04_20261123.md. This source does not authorize deleting or redirecting any path.",
                ),
            )
            source_id = int(cur.lastrowid)

        existing_group_ids = {
            row["detection_run_id"]: int(row["duplicate_group_id"])
            for row in con.execute(
                "SELECT duplicate_group_id,detection_run_id FROM duplicate_group"
            )
        }
        inserted_groups = 0
        inserted_members = 0
        linked_evidence = 0
        updated_artifacts = 0
        path_count = 0
        for group in groups:
            group_name = str(group["group_id"])
            digest = str(group["sha256"])
            byte_size = int(group["logical_bytes_each"])
            alloc_each = int(group["allocated_bytes_each"])
            paths = list(group["paths"])
            group_key = f"{DETECTION_PREFIX}:{group_name}"
            rationale = (
                f"Exact SHA-256 verified on {args.captured_utc}; {len(paths)} paths, "
                f"{byte_size} logical bytes each ({alloc_each} allocated bytes each), "
                f"digest {digest}. Disposition remains PENDING: matching bytes alone do "
                f"not establish that either path can be removed or redirected. "
            )
            if group_name in {"DUP-01", "DUP-02"}:
                rationale += (
                    "The 39 identical test.xyz/valid.xyz paths represent the same seed-42 "
                    "evaluation split; the base dataset copy is recorded as canonical. "
                    "Consolidation is held because current variant builders and sanitizers "
                    "write split files in place, which can mutate a shared hardlink or "
                    "symlink target; update writers to atomic replacement/detachment first."
                )
            elif group_name == "DUP-10":
                rationale += (
                    "This is a Git pack with external hardlinks; handle only through Git "
                    "object-database maintenance and resolve every inode path before any cleanup."
                )
            else:
                rationale += (
                    "Project-specific producer/consumer role, canonical path, and run/artifact "
                    "lineage are unresolved; retain both pending that review."
                )

            member_rows: list[tuple[int, bool]] = []
            for path in paths:
                if not path.startswith(SCOPED_PATH + "/"):
                    raise ValueError(f"Path outside expected root: {path}")
                relative = path[len(SCOPED_PATH) + 1 :]
                path_row = con.execute(
                    "SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?",
                    (ROOT_ID, relative),
                ).fetchone()
                if not path_row:
                    raise ValueError(f"Path missing from catalog: {path}")
                locations = con.execute(
                    """SELECT a.artifact_id,a.sha256,a.byte_size,a.notes
                       FROM artifact_location l JOIN artifact a USING(artifact_id)
                       WHERE l.path_id=? AND l.availability='PRESENT'""",
                    (path_row["path_id"],),
                ).fetchall()
                if len(locations) != 1:
                    raise ValueError(
                        f"Expected one present artifact location for {path}, got {len(locations)}"
                    )
                artifact = locations[0]
                if artifact["sha256"] not in (None, digest):
                    raise ValueError(f"SHA-256 conflict for catalog artifact: {path}")
                if artifact["byte_size"] not in (None, byte_size):
                    raise ValueError(f"Byte-size conflict for catalog artifact: {path}")
                note = append_note(
                    artifact["notes"],
                    f"Exact SHA-256 recorded from R04 duplicate review ({group_name}, {args.captured_utc}); deletion/repointing not approved.",
                )
                con.execute(
                    "UPDATE artifact SET sha256=?,byte_size=COALESCE(byte_size,?),notes=? WHERE artifact_id=?",
                    (digest, byte_size, note, artifact["artifact_id"]),
                )
                con.execute(
                    """UPDATE artifact_location SET
                         last_verified_utc=CASE WHEN last_verified_utc IS NULL OR last_verified_utc<? THEN ? ELSE last_verified_utc END
                       WHERE artifact_id=? AND path_id=?""",
                    (
                        args.captured_utc,
                        args.captured_utc,
                        artifact["artifact_id"],
                        path_row["path_id"],
                    ),
                )
                updated_artifacts += 1
                is_canonical = group_name in {"DUP-01", "DUP-02"} and path == paths[0]
                member_rows.append((int(artifact["artifact_id"]), is_canonical))
                path_count += 1

            if group_key in existing_group_ids:
                group_id = existing_group_ids[group_key]
                existing = con.execute(
                    "SELECT * FROM duplicate_group WHERE duplicate_group_id=?", (group_id,)
                ).fetchone()
                if not (
                    existing["match_method"] == "SHA256"
                    and existing["verification_status"] == "VERIFIED_IDENTICAL"
                    and existing["disposition"] == "PENDING"
                ):
                    raise ValueError(f"Existing group has conflicting status: {group_key}")
            else:
                cur = con.execute(
                    """INSERT INTO duplicate_group
                       (detection_run_id,match_method,verification_status,disposition,rationale)
                       VALUES (?,'SHA256','VERIFIED_IDENTICAL','PENDING',?)""",
                    (group_key, rationale),
                )
                group_id = int(cur.lastrowid)
                inserted_groups += 1

            actual_members = {
                int(row["artifact_id"]): int(row["canonical_member"])
                for row in con.execute(
                    "SELECT artifact_id,canonical_member FROM duplicate_member WHERE duplicate_group_id=?",
                    (group_id,),
                )
            }
            expected_members = {artifact_id: int(canonical) for artifact_id, canonical in member_rows}
            if actual_members and actual_members != expected_members:
                raise ValueError(f"Existing members differ for {group_key}")
            if not actual_members:
                con.executemany(
                    "INSERT INTO duplicate_member(duplicate_group_id,artifact_id,canonical_member) VALUES (?,?,?)",
                    [(group_id, aid, int(canonical)) for aid, canonical in member_rows],
                )
                inserted_members += len(member_rows)

            evidence_exists = con.execute(
                "SELECT 1 FROM duplicate_group_evidence WHERE duplicate_group_id=? AND source_id=? AND evidence_role='HASH_COMPARISON'",
                (group_id, source_id),
            ).fetchone()
            if not evidence_exists:
                con.execute(
                    """INSERT INTO duplicate_group_evidence
                       (duplicate_group_id,source_id,evidence_role,notes)
                       VALUES (?,?,'HASH_COMPARISON',?)""",
                    (
                        group_id,
                        source_id,
                        f"Source TSV SHA-256={source_hash}; {group_name}; full-file SHA-256={digest}; metadata-guarded stable read.",
                    ),
                )
                linked_evidence += 1

        fk_errors = list(con.execute("PRAGMA foreign_key_check"))
        if fk_errors:
            raise RuntimeError(f"Foreign-key violations: {len(fk_errors)}")
        con.commit()

        imported = con.execute(
            """SELECT COUNT(*) AS groups,
                      SUM((SELECT COUNT(*) FROM duplicate_member m WHERE m.duplicate_group_id=g.duplicate_group_id)) AS members,
                      SUM((SELECT COUNT(*) FROM duplicate_group_evidence e WHERE e.duplicate_group_id=g.duplicate_group_id AND e.source_id=?)) AS evidence_links
               FROM duplicate_group g WHERE detection_run_id LIKE ?""",
            (source_id, DETECTION_PREFIX + ":%"),
        ).fetchone()
        print(json.dumps({
            "source_key": SOURCE_KEY,
            "source_sha256": source_hash,
            "groups_file": str(groups_path),
            "db": str(db_path),
            "inserted_groups_this_run": inserted_groups,
            "inserted_members_this_run": inserted_members,
            "new_evidence_links_this_run": linked_evidence,
            "artifacts_checked_or_updated": updated_artifacts,
            "paths_in_source": path_count,
            "catalog_groups_for_source": imported["groups"],
            "catalog_members_for_source": imported["members"],
            "catalog_evidence_links": imported["evidence_links"],
            "foreign_key_violations": len(fk_errors),
        }, indent=2))
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()
