#!/usr/bin/env python3
"""Register isolated local Git mainline reconciliation receipts in the catalog."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sqlite3
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
GIT_DIR = AUDIT / "phase2_data" / "git_merge_candidates_20261122"
RECEIPT = GIT_DIR / "remote_snapshot_and_merge_receipts.json"
REPORT = AUDIT / "phase2_data" / "LOCAL_GIT_MERGE_CANDIDATES_REPORT.md"
LOCAL_ROOT_ID = "LOCAL-MAIN"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(path: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(path), *args], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def main() -> None:
    evidence = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if evidence.get("push_performed") is not False or evidence.get("rockfish_refs_changed") is not False:
        raise SystemExit("Git receipt indicates a push or Rockfish ref mutation")
    created = evidence["created_utc"].replace("+00:00", "Z")
    sources = []
    for key, path, kind, note in (
        ("git_local_merge_receipts_20261122", RECEIPT, "GIT_LOCAL_MERGE_RECEIPT_JSON",
         "Isolated clone refs and local merge receipts; GitHub was not pushed and Rockfish refs were untouched."),
        ("git_local_merge_candidates_report_20261122", REPORT, "GIT_LOCAL_MERGE_SUMMARY",
         "Summary of local mainline candidates, including blocked branches and no-push status."),
    ):
        sources.append((key, path.resolve(), kind, sha256(path), note))

    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=120000")
    try:
        con.execute("BEGIN IMMEDIATE")
        for key, path, kind, digest, note in sources:
            prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (key,)).fetchone()
            if prior and prior != (str(path), digest):
                raise RuntimeError("Git evidence source key already points to changed content: " + key)
            con.execute(
                """INSERT OR IGNORE INTO registry_source
                   (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note)
                   VALUES(?,?,?,?,?,?,1,?)""",
                (key, str(path), kind, digest, created, created, note),
            )
            root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (LOCAL_ROOT_ID,)).fetchone()
            if not root:
                raise RuntimeError("LOCAL-MAIN storage root missing")
            rel = path.relative_to(Path(root[0])).as_posix()
            con.execute(
                "INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)",
                (LOCAL_ROOT_ID, rel, created),
            )
            path_id = con.execute(
                "SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (LOCAL_ROOT_ID, rel)
            ).fetchone()[0]
            artifact_key = "GIT-MERGE-EVIDENCE:" + path.name
            con.execute(
                """INSERT OR IGNORE INTO artifact
                   (artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
                   VALUES(?,'GIT_RECONCILIATION_EVIDENCE',?,'USER_CONFIRMED',?,?,'VERIFIED','PERMANENT_SOURCE',?)""",
                (artifact_key, kind, digest, path.stat().st_size, note),
            )
            artifact_id, artifact_hash = con.execute(
                "SELECT artifact_id,sha256 FROM artifact WHERE artifact_key=?", (artifact_key,)
            ).fetchone()
            if artifact_hash != digest:
                raise RuntimeError("Git evidence artifact hash mismatch: " + artifact_key)
            con.execute(
                """INSERT OR IGNORE INTO artifact_location
                   (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
                   VALUES(?,?,'PRESENT',?,?)""",
                (artifact_id, path_id, created, created),
            )

        source_id = con.execute(
            "SELECT source_id FROM registry_source WHERE source_key='git_local_merge_receipts_20261122'"
        ).fetchone()[0]
        candidate_repositories = {}
        candidate_specs = (
            ("MigrationBench", 9, "MigrationBench", "main"),
            ("md_path_search", 18, "md_path_search", "main"),
        )
        for evidence_name, parent_repo_id, repo_name, canonical in candidate_specs:
            row = evidence["repositories"][evidence_name]
            clone = Path(row["clone_path"]).resolve()
            if git(clone, "status", "--porcelain=v1", "--untracked-files=all"):
                raise RuntimeError("Isolated candidate worktree is no longer clean: " + str(clone))
            head = git(clone, "rev-parse", "HEAD")
            if git(clone, "branch", "--show-current") != canonical or head != row["result_main"]:
                raise RuntimeError("Isolated candidate HEAD differs from merge receipt: " + str(clone))
            parent_key = con.execute("SELECT repository_key FROM git_repository WHERE repository_id=?", (parent_repo_id,)).fetchone()[0]
            candidate_key = "local-merge-candidate:" + evidence_name + ":2026-11-22"
            con.execute(
                """INSERT OR IGNORE INTO git_repository
                   (repository_key,local_path,remote_url,ownership_status,canonical_branch,notes)
                   VALUES(?,?,?,'USER',?,'Isolated local candidate clone; original worktree and remote refs were left untouched.')""",
                (candidate_key, str(clone), row["url"], canonical),
            )
            candidate_id = con.execute("SELECT repository_id FROM git_repository WHERE repository_key=?", (candidate_key,)).fetchone()[0]
            candidate_repositories[evidence_name] = candidate_id
            shallow = git(clone, "rev-parse", "--is-shallow-repository") == "true"
            snapshot_notes = (
                "Clean isolated clone at merge-receipt HEAD. Remote ls-remote refs at the receipt timestamp are preserved in the linked source; "
                "this local worktree snapshot does not claim a newer live remote-tip check."
            )
            con.execute(
                """INSERT OR IGNORE INTO git_repository_snapshot
                   (repository_id,source_id,observed_utc,head_ref,head_sha,status_exit_code,
                    tracked_dirty_count,staged_dirty_count,unstaged_dirty_count,untracked_file_count,
                    worktree_dirty,fetch_status,remote_tip_status,shallow_clone,classification,
                    classification_basis,parent_repo_key,parent_index_relation,notes)
                   VALUES(?,?,?,?,?,0,0,0,0,0,0,'NOT_RECORDED','UNKNOWN_NOT_VERIFIED',?,?,?,?,?,?)""",
                (candidate_id, source_id, created, canonical, head, int(shallow), "LOCAL_MAINLINE_CANDIDATE",
                 "Temporary clone constructed to evaluate a non-pushed mainline reconciliation.", parent_key,
                 "isolated clone of " + parent_key, snapshot_notes),
            )
            canonical_sha = head
            refs_text = git(clone, "for-each-ref", "--format=%(refname)|%(objectname)", "refs/heads", "refs/remotes/origin")
            for ref_line in refs_text.splitlines():
                full_ref, commit = ref_line.split("|", 1)
                if full_ref.endswith("/HEAD"):
                    continue
                if full_ref.startswith("refs/heads/"):
                    ref_name, kind = full_ref[len("refs/heads/"):], "BRANCH"
                elif full_ref.startswith("refs/remotes/"):
                    ref_name, kind = full_ref[len("refs/remotes/"):], "REMOTE_BRANCH"
                else:
                    continue
                ancestor = subprocess.run(["git", "-C", str(clone), "merge-base", "--is-ancestor", commit, canonical_sha]).returncode == 0
                existing_ref = con.execute(
                    "SELECT git_ref_observation_id FROM git_ref_observation WHERE repository_id=? AND observed_utc=? AND ref_name=?",
                    (candidate_id, created, ref_name),
                ).fetchone()
                if not existing_ref:
                    con.execute(
                        """INSERT INTO git_ref_observation
                           (repository_id,observed_utc,ref_name,ref_kind,commit_sha,is_merged_to_canonical,dirty_worktree,notes)
                           VALUES(?,?,?,?,?,?,0,?)""",
                        (candidate_id, created, ref_name, kind, commit, int(ancestor),
                         "Captured from the isolated clone linked to git repository snapshot source_id=" + str(source_id)),
                    )
        # The existing event rows represent actions in these candidate clones.
        for repo_key, candidate_name in ((9, "MigrationBench"), (18, "md_path_search")):
            con.execute(
                "UPDATE git_merge_observation SET repository_id=? WHERE evidence_source_id=? AND repository_id=?",
                (candidate_repositories[candidate_name], source_id, repo_key),
            )
        con.execute(
            """UPDATE git_merge_observation SET pre_target_sha=resulting_sha
               WHERE evidence_source_id=? AND source_ref='dev' AND merge_result='ALREADY_CONTAINS'""",
            (source_id,),
        )
        rows = [
            (candidate_repositories["MigrationBench"], "main", "origin/main", evidence["repositories"]["MigrationBench"]["starting_main"],
             evidence["repositories"]["MigrationBench"]["live_heads"]["main"],
             evidence["repositories"]["MigrationBench"]["result_main"], "FAST_FORWARD",
             evidence["repositories"]["MigrationBench"]["clone_path"],
             "Temporary isolated clone only; repository candidate fast-forwarded to the current remote main."),
            (candidate_repositories["MigrationBench"], "main", "dev", evidence["repositories"]["MigrationBench"]["result_main"],
             evidence["repositories"]["MigrationBench"]["dev"],
             evidence["repositories"]["MigrationBench"]["result_main"], "ALREADY_CONTAINS",
             evidence["repositories"]["MigrationBench"]["clone_path"],
             "The observed dev tip is already contained in the resulting main history; no separate merge was needed."),
            (candidate_repositories["md_path_search"], "main", "hpc", evidence["repositories"]["md_path_search"]["starting_main"],
             evidence["repositories"]["md_path_search"]["local_hpc"],
             evidence["repositories"]["md_path_search"]["result_main"], "MERGE_COMMIT",
             evidence["repositories"]["md_path_search"]["clone_path"],
             "Local merge candidate in an isolated clone; original dirty worktree remains untouched."),
        ]
        for repo, target, source, pre, src_sha, result, outcome, worktree, note in rows:
            prior = con.execute(
                """SELECT git_merge_observation_id FROM git_merge_observation
                   WHERE repository_id=? AND target_ref=? AND source_ref=? AND pre_target_sha=? AND source_sha=? AND resulting_sha=?""",
                (repo, target, source, pre, src_sha, result),
            ).fetchone()
            if prior:
                continue
            con.execute(
                """INSERT INTO git_merge_observation
                   (repository_id,evidence_source_id,observed_utc,target_ref,source_ref,pre_target_sha,source_sha,resulting_sha,
                    merge_result,worktree_path,worktree_clean,push_status,notes)
                   VALUES(?,?,?,?,?,?,?,?,?,?,1,'NOT_PUSHED',?)""",
                (repo, source_id, created, target, source, pre, src_sha, result, outcome, worktree, note),
            )
        if con.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Foreign-key check failed")
        con.commit()
        print(json.dumps({"observations": con.execute("SELECT count(*) FROM git_merge_observation").fetchone()[0],
                          "candidate_repositories": candidate_repositories,
                          "receipt_sha256": sha256(RECEIPT), "report_sha256": sha256(REPORT),
                          "push_performed": False}, sort_keys=True))
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()
