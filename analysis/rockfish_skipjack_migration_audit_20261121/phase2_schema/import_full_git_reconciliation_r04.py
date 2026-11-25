#!/usr/bin/env python3
"""Import the fresh 29-repository Git ref/dirty-state reconciliation into SQL."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
EVIDENCE = AUDIT / "phase2_data/full_git_reconciliation_r04_20261123"
RECEIPTS = EVIDENCE / "full_git_reconciliation_r04_receipts.json"
REPORT = EVIDENCE / "full_git_reconciliation_r04_report_20261123.md"
MATRIX = EVIDENCE / "full_git_reconciliation_r04_matrix_20261123.tsv"
SNAPSHOT_UTC = "2026-11-23T10:14:48Z"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def register_source(con: sqlite3.Connection, path: Path, key: str, kind: str,
                    captured: str, as_of: str, note: str) -> int:
    path = path.resolve(strict=True)
    digest = sha256(path)
    source_key = f"{key}:{digest[:16]}"
    prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (source_key,)).fetchone()
    if prior and tuple(prior) != (str(path), digest):
        raise RuntimeError("registry source collision: " + source_key)
    con.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
            immutable_snapshot,authority_note,notes)
           VALUES(?,?,?,?,?,?,1,?,?)""",
        (source_key, str(path), kind, digest, captured, as_of,
         "Exact local metadata evidence bytes identified by SHA-256; no repository objects were fetched by this importer.", note),
    )
    return con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (source_key,)).fetchone()[0]


def observe_ref(con: sqlite3.Connection, repository_id: int, ref_name: str, ref_kind: str,
                sha: str, observed: str, upstream: str | None, merged: int | None,
                dirty: int | None, notes: str) -> int:
    prior = con.execute(
        """SELECT git_ref_observation_id FROM git_ref_observation
           WHERE repository_id=? AND observed_utc=? AND ref_name=? AND ref_kind=? AND commit_sha=?""",
        (repository_id, observed, ref_name, ref_kind, sha),
    ).fetchone()
    if prior:
        return int(prior[0])
    cur = con.execute(
        """INSERT INTO git_ref_observation
           (repository_id,observed_utc,ref_name,ref_kind,commit_sha,upstream_ref,
            is_merged_to_canonical,dirty_worktree,notes)
           VALUES(?,?,?,?,?,?,?,?,?)""",
        (repository_id, observed, ref_name, ref_kind, sha, upstream, merged, dirty, notes),
    )
    return int(cur.lastrowid)


def main() -> None:
    data = json.loads(RECEIPTS.read_text(encoding="utf-8"))
    repos = data.get("repositories", [])
    if len(repos) != 29 or data.get("fetch_performed") or data.get("push_performed"):
        raise RuntimeError("unexpected repository count or remote mutation in R04 Git receipts")
    captured = data["captured_utc"]
    live_as_of = data["live_remote_capture_utc"]
    if live_as_of != "2026-11-23T10:13:09Z":
        raise RuntimeError("unexpected live-ref capture timestamp")

    con = sqlite3.connect(DB, timeout=600)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=600000")
    con.execute("PRAGMA synchronous=NORMAL")
    try:
        con.execute("BEGIN IMMEDIATE")
        srcs = [
            register_source(con, RECEIPTS, "rockfish_r04_full_git_receipts", "FULL_GIT_RECONCILIATION_RECEIPTS", captured, live_as_of,
                            "Exact 29-repository branch/object/status matrix and live-ref observations; no fetch or push."),
            register_source(con, REPORT, "rockfish_r04_full_git_report", "FULL_GIT_RECONCILIATION_REPORT", captured, live_as_of,
                            "Human-readable Git merge eligibility and unresolved-object report for 25 yi and four ycao73_home repositories."),
            register_source(con, MATRIX, "rockfish_r04_full_git_matrix", "FULL_GIT_RECONCILIATION_MATRIX", captured, live_as_of,
                            "Per-repository current branch, object, ancestry, dirty state, live default and action matrix."),
        ]
        repo_map: dict[str, int] = {}
        ref_observations = 0
        for row in repos:
            path = row["path"]
            rec = con.execute("SELECT repository_id,remote_url FROM git_repository WHERE local_path=?", (path,)).fetchone()
            if not rec:
                upstream = row.get("origin_url")
                con.execute(
                    """INSERT OR IGNORE INTO git_repository
                       (repository_key,local_path,remote_url,ownership_status,canonical_branch,notes)
                       VALUES(?,?,?,'THIRD_PARTY',?,?)""",
                    ("rockfish:" + path, path, upstream, row.get("live_default_branch"),
                     "Added from the exact 29-repository R04 Git state capture; this is a user storage path for an upstream third-party checkout."),
                )
                rec = con.execute("SELECT repository_id,remote_url FROM git_repository WHERE local_path=?", (path,)).fetchone()
                if not rec:
                    raise RuntimeError("failed to add missing Git repository mapping: " + path)
            repo_id = int(rec["repository_id"])
            repo_map[path] = repo_id
            live_branch = row.get("live_default_branch")
            live_sha = row.get("live_default_sha")
            if live_branch:
                con.execute("UPDATE git_repository SET canonical_branch=? WHERE repository_id=?", (live_branch, repo_id))
            status = row.get("status_counts") or {}
            dirty = int(any(int(status.get(k, 0) or 0) > 0 for k in ("conflicted", "staged", "unstaged", "untracked")))
            notes_common = (f"R04 snapshot of {path}; {row.get('action_code')}. "
                            "Original Rockfish refs/worktree were not modified. Ignored paths are reported separately.")

            for local in row.get("local_branches", []):
                name = local.get("branch")
                sha = local.get("sha")
                if not name or not sha or len(sha) != 40:
                    continue
                relation = local.get("branch_vs_live_default") or {}
                merged = 1 if relation.get("branch_is_ancestor") is True else 0 if relation.get("branch_is_ancestor") is False else None
                checked_out = name == row.get("current_branch")
                observe_ref(con, repo_id, "refs/heads/" + name, "BRANCH", sha, captured,
                    "refs/remotes/origin/" + (live_branch or "") if live_branch else None,
                    merged, dirty if checked_out else None,
                    notes_common + (" Live ancestry is unknown where the live object was absent locally." if merged is None else ""))
                ref_observations += 1

            cached_sha = row.get("cached_default_sha")
            cached_branch = row.get("cached_default_ref")
            if cached_sha and len(cached_sha) == 40 and cached_branch:
                observe_ref(con, repo_id, cached_branch, "REMOTE_BRANCH", cached_sha, captured,
                    "refs/heads/" + str(live_branch) if live_branch else None,
                    None, None,
                    notes_common + " This is the existing cached remote-tracking ref, not a fresh fetch.")
                ref_observations += 1

            if live_branch and live_sha and len(live_sha) == 40:
                observe_ref(con, repo_id, "live:refs/heads/" + live_branch, "REMOTE_BRANCH", live_sha, live_as_of,
                    "refs/heads/" + live_branch, 1, None,
                    notes_common + " Live SHA captured with ls-remote; no fetch was performed and this object may be absent from Rockfish.")
                ref_observations += 1

        for source_id in srcs:
            for path, repo_id in repo_map.items():
                con.execute(
                    """INSERT OR IGNORE INTO registry_entity_ref
                       (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                       VALUES(?,'repository_scope',?,'GIT_REPOSITORY',?,'OBSERVES_REFS',?)""",
                    (source_id, path, str(repo_id), "Full R04 repository matrix or summary evidence for this Rockfish Git checkout."),
                )
        fk = con.execute("PRAGMA foreign_key_check").fetchall()
        if fk:
            raise RuntimeError("foreign-key check failed: " + repr(fk[:5]))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    print(json.dumps({"status": "IMPORTED_GIT_REF_SNAPSHOT", "repositories": len(repo_map),
        "ref_observations_inserted_or_present": ref_observations,
        "live_ref_capture_utc": live_as_of, "fetch_performed": False, "push_performed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
