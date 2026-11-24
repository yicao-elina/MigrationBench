#!/usr/bin/env python3
"""Record the current unified local mainline without fetching or changing refs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


AUDIT = Path(__file__).resolve().parent.parent
REPO = AUDIT.parents[1]
DB = AUDIT / "provenance_catalog.sqlite"
OUTPUT = AUDIT / "phase2_data" / "unified_main_git_snapshot_20261123.json"
REVIEW = AUDIT / "phase2_data" / "LOCAL_GIT_ALIGNMENT_STATUS_20261123.md"
LOCAL_ROOT_ID = "LOCAL-MAIN"
REPOSITORY_KEY = "unified-migrationbench-main-local"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_remote(url: str) -> str:
    parts = urlsplit(url)
    if not parts.scheme or not parts.hostname:
        return url
    host = parts.hostname
    if parts.port:
        host += ":" + str(parts.port)
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


def main() -> None:
    observed = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if branch != "main" or status:
        raise SystemExit("unified worktree must be clean on main before recording its snapshot")
    upstream = git("rev-parse", "--abbrev-ref", "main@{upstream}") if subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "main@{upstream}"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0 else None
    remotes = {}
    for line in git("remote", "-v").splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[2] in ("(fetch)", "(push)"):
            remote = remotes.setdefault(fields[0], {"fetch_url": None, "push_url": None})
            remote["fetch_url" if fields[2] == "(fetch)" else "push_url"] = clean_remote(fields[1])
    refs = []
    for line in git("for-each-ref", "--format=%(refname)|%(objectname)|%(upstream)", "refs/heads", "refs/remotes").splitlines():
        full_ref, commit, upstream_ref = line.split("|", 2)
        if full_ref.endswith("/HEAD"):
            continue
        if full_ref.startswith("refs/heads/"):
            ref_name, ref_kind = full_ref[len("refs/heads/"):], "BRANCH"
        elif full_ref.startswith("refs/remotes/"):
            ref_name, ref_kind = full_ref[len("refs/remotes/"):], "REMOTE_BRANCH"
        else:
            continue
        is_ancestor = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", commit, head]).returncode == 0
        counts = git("rev-list", "--left-right", "--count", "main..." + full_ref).split()
        refs.append({
            "full_ref": full_ref,
            "ref_name": ref_name,
            "ref_kind": ref_kind,
            "commit_sha": commit,
            "upstream_ref": upstream_ref or None,
            "is_merged_to_canonical": bool(is_ancestor),
            "current_checkout": full_ref == "refs/heads/main",
            "local_only_ref": bool(ref_kind == "BRANCH" and not upstream_ref),
            "mainline_only_commits": int(counts[0]),
            "branch_only_commits": int(counts[1]),
        })
    snapshot = {
        "observed_utc": observed,
        "repository_key": REPOSITORY_KEY,
        "repository_path": str(REPO),
        "canonical_branch": "main",
        "head_ref": "refs/heads/main",
        "head_sha": head,
        "upstream_ref": upstream,
        "worktree_dirty": False,
        "status_porcelain": status,
        "remotes": remotes,
        "ref_scope": "local heads and locally cached remote-tracking refs; no fetch performed",
        "live_remote_tips_checked": False,
        "refs": refs,
        "notes": "Current unified local mainline after phase-2 audit commit. GitHub was not pushed; external Rockfish repositories were not modified.",
    }
    OUTPUT.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    source_specs = []
    for base_key, path, kind, note in (
        ("unified_main_git_snapshot_20261123", OUTPUT, "UNIFIED_LOCAL_GIT_SNAPSHOT", "Current clean unified main branch and local cached ref graph; no fetch or push."),
        ("unified_main_git_alignment_review_20261123", REVIEW, "UNIFIED_LOCAL_GIT_ALIGNMENT_REVIEW", "Read-only ancestry review of unified local main, isolated candidate clones, and blocked Rockfish refs."),
    ):
        path = path.resolve(strict=True)
        digest = sha256(path)
        key = base_key + ":" + digest[:16] if path == OUTPUT.resolve() else base_key
        source_specs.append((key, path, kind, digest, note))

    con = sqlite3.connect(DB, timeout=180)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=180000")
    try:
        con.execute("BEGIN IMMEDIATE")
        root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (LOCAL_ROOT_ID,)).fetchone()
        if not root or Path(root[0]).resolve() != REPO.resolve():
            raise RuntimeError("LOCAL-MAIN root does not point to the unified worktree")
        sources = {}
        for source_key, path, kind, digest, note in source_specs:
            prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (source_key,)).fetchone()
            if prior and prior != (str(path), digest):
                raise RuntimeError("source key changed identity: " + source_key)
            con.execute(
                """INSERT OR IGNORE INTO registry_source
                   (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note)
                   VALUES(?,?,?,?,?,?,1,?)""",
                (source_key, str(path), kind, digest, observed, observed, note),
            )
            sources[source_key] = con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (source_key,)).fetchone()[0]
            rel = path.relative_to(REPO).as_posix()
            con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (LOCAL_ROOT_ID, rel, observed))
            path_id = con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (LOCAL_ROOT_ID, rel)).fetchone()[0]
            artifact_key = "GIT-LOCAL-SNAPSHOT:" + source_key
            con.execute(
                """INSERT OR IGNORE INTO artifact
                   (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
                    scientific_status,retention_class,notes)
                   VALUES('MIGRATIONBENCH',?,'GIT_REPOSITORY_SNAPSHOT',?,'USER_CONFIRMED',?,?,'VERIFIED','PERMANENT_SOURCE',?)""",
                (artifact_key, kind, digest, path.stat().st_size, note),
            )
            aid, ahash, asize = con.execute("SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
            if (ahash, asize) != (digest, path.stat().st_size):
                raise RuntimeError("Git snapshot artifact hash/size conflict")
            con.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)",
                        (aid, path_id, observed, observed))

        output_source_key = next(key for key, path, _, _, _ in source_specs if path == OUTPUT.resolve())
        source_id = sources[output_source_key]
        report_source_id = sources["unified_main_git_alignment_review_20261123"]
        con.execute(
            """INSERT OR IGNORE INTO git_repository
               (repository_key,local_path,remote_url,project_id,ownership_status,canonical_branch,notes)
               VALUES(?,?,?,'MIGRATIONBENCH','USER','main',?)""",
            (REPOSITORY_KEY, str(REPO), remotes.get("origin", {}).get("fetch_url"),
             "Unified local MigrationBench repository. This root records branch snapshots; no push is implied."),
        )
        repo = con.execute("SELECT repository_id,local_path,canonical_branch FROM git_repository WHERE repository_key=?", (REPOSITORY_KEY,)).fetchone()
        if repo[1:] != (str(REPO), "main"):
            raise RuntimeError("unified repository identity conflict")
        repo_id = repo[0]
        con.execute(
            """INSERT OR IGNORE INTO git_repository_snapshot
               (repository_id,source_id,observed_utc,head_ref,head_sha,status_exit_code,
                tracked_dirty_count,staged_dirty_count,unstaged_dirty_count,untracked_file_count,
                ignored_matching_count,worktree_dirty,status_error,fetch_status,remote_tip_status,
                shallow_clone,classification,classification_basis,notes)
               VALUES(?,?,?,'refs/heads/main',?,0,0,0,0,0,0,0,'','NOT_RECORDED','UNKNOWN_NOT_VERIFIED',0,
                      'CANONICAL_LOCAL_WORKTREE','Clean main; remote tips not fetched',?)""",
            (repo_id, source_id, observed, head,
             "Read-only Git status and ref graph after audit commit; cached remote refs are not asserted current."),
        )
        snap_id = con.execute("SELECT repository_snapshot_id FROM git_repository_snapshot WHERE repository_id=? AND source_id=?", (repo_id, source_id)).fetchone()[0]
        for name, url in remotes.items():
            con.execute(
                """INSERT OR IGNORE INTO git_remote_observation
                   (repository_snapshot_id,remote_name,fetch_url,push_url,fetch_status,current_tip_status,current_tip_sha,notes)
                   VALUES(?,?,?,?,'NOT_ATTEMPTED','UNKNOWN_NOT_VERIFIED',NULL,?)""",
                (snap_id, name, url.get("fetch_url"), url.get("push_url"), "Remote URLs captured locally; no ls-remote/fetch performed during this snapshot."),
            )
        for ref in refs:
            upstream_ref = ref["upstream_ref"]
            if upstream_ref and not upstream_ref.startswith("refs/"):
                upstream_ref = "refs/remotes/" + upstream_ref
            graph = "EQUAL" if ref["mainline_only_commits"] == ref["branch_only_commits"] == 0 else (
                "MAIN_AHEAD" if ref["branch_only_commits"] == 0 else (
                    "BRANCH_AHEAD" if ref["mainline_only_commits"] == 0 else "DIVERGED"))
            cursor = con.execute(
                """INSERT INTO git_ref_observation
                   (repository_id,observed_utc,ref_name,ref_kind,commit_sha,upstream_ref,
                    is_merged_to_canonical,dirty_worktree,notes)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (repo_id, observed, ref["ref_name"], ref["ref_kind"], ref["commit_sha"], upstream_ref,
                 int(ref["is_merged_to_canonical"]), 0,
                 "Local cached ref snapshot only; merge-base against unified main computed locally."),
            )
            con.execute(
                """INSERT INTO git_ref_context_snapshot
                   (git_ref_observation_id,repository_snapshot_id,source_id,is_current_checkout,
                    is_local_only_ref,local_mainline_ref,graph_relation_to_mainline,
                    mainline_only_commits,branch_only_commits,remote_tip_status,notes)
                   VALUES(?,?,?,?,?,'refs/heads/main',?, ?,?,'UNKNOWN_NOT_VERIFIED',?)""",
                (cursor.lastrowid, snap_id, source_id, int(ref["current_checkout"]), int(ref["local_only_ref"]), graph,
                 ref["mainline_only_commits"], ref["branch_only_commits"],
                 "Ancestry is measured only against locally available Git objects."),
            )
            if ref["full_ref"] != "refs/heads/main":
                con.execute(
                    """INSERT OR IGNORE INTO git_branch_pair_graph_snapshot
                       (repository_snapshot_id,ref_a,ref_b,relation,a_only_commits,b_only_commits,
                        common_base_ref_or_oid,notes)
                       VALUES(?,'refs/heads/main',?,?,?,?,?,?)""",
                    (snap_id, ref["full_ref"], graph, ref["mainline_only_commits"], ref["branch_only_commits"],
                     "git merge-base main " + ref["full_ref"], "Local graph only; remote tip freshness unverified."),
                )
        for sid, entity_type, entity_key, notes in (
            (source_id, "git_repository_snapshot", REPOSITORY_KEY, "Exact local main and cached ref graph snapshot."),
            (report_source_id, "git_alignment_review", REPOSITORY_KEY, "Review conclusion; external Rockfish refs remain unchanged and unpushed."),
        ):
            con.execute(
                """INSERT OR IGNORE INTO registry_entity_ref
                   (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
                   VALUES(?,?,?,'GIT_REPOSITORY',?,'EVIDENCE_FOR',?)""",
                (sid, entity_type, entity_key, REPOSITORY_KEY, notes),
            )
        if con.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("foreign-key check failed")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    print(json.dumps({"repository_key": REPOSITORY_KEY, "repository_id": repo_id,
                      "head_sha": head, "observed_refs": len(refs), "worktree_dirty": False,
                      "fetch_performed": False, "push_performed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
