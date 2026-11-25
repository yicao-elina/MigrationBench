#!/usr/bin/env python3
"""Normalize isolated Git-candidate refs into ancestry, upstream, and remote tables."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
SOURCE = AUDIT / "phase2_data" / "git_merge_candidates_20261122" / "remote_snapshot_and_merge_receipts.json"
EXPECTED_SOURCE_SHA = "0e6e22e4b91c02c28462323953c1e356ee1d69c3f0a81fe0d6ee71fed58419c7"
OBSERVED = "2026-11-22T22:10:04Z"


def git(path: str, *args: str) -> str:
    return subprocess.run(["git", "-C", path, *args], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def main() -> None:
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if source_sha != EXPECTED_SOURCE_SHA:
        raise RuntimeError("Isolated clone/remote receipt JSON hash changed")
    receipt = json.loads(SOURCE.read_text(encoding="utf-8"))
    if receipt.get("push_performed") or receipt.get("rockfish_refs_changed"):
        raise RuntimeError("Expected local-only clone receipts with no push or Rockfish ref changes")

    con = sqlite3.connect(DB, timeout=120)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=120000")
    try:
        con.execute("BEGIN IMMEDIATE")
        source = con.execute("SELECT source_id,sha256 FROM registry_source WHERE source_key='git_local_merge_receipts_20261122'").fetchone()
        if not source or source["sha256"] != source_sha:
            raise RuntimeError("Immutable Git merge source is not registered at its expected hash")

        configs = {
            "MigrationBench": {
                "repo_id": 27, "snapshot_id": 43, "mainline": "main",
                "ref_context": {
                    "dev": (0, 0, "main", "dev", "ANCESTOR_OF_MAIN", 0, 20, "ANCESTOR", 20, 0),
                    "main": (1, 0, None, None, "CURRENT_CHECKOUT", None, None, "CANONICAL", 0, 0),
                    "origin/dev": (0, 0, "main", "origin/dev", "ANCESTOR_OF_MAIN", 0, 20, "ANCESTOR", 20, 0),
                    "origin/main": (0, 0, "main", "origin/main", "EQUAL_TO_LOCAL_MAIN", 0, 0, "EQUAL_TO_LOCAL_MAIN", 0, 0),
                },
                "pairs": [("main", "dev", "A_CONTAINS_B", 20, 0, "d48e652af7acb82ca71f860cf0d7df2330df3b9c"),
                          ("main", "origin/dev", "A_CONTAINS_B", 20, 0, "d48e652af7acb82ca71f860cf0d7df2330df3b9c")],
                "remote_heads": {"main": "817acf22af7374bcb2a79d123d54ad77a7411c1c", "dev": "d48e652af7acb82ca71f860cf0d7df2330df3b9c"},
            },
            "md_path_search": {
                "repo_id": 28, "snapshot_id": 44, "mainline": "main",
                "ref_context": {
                    "hpc": (0, 0, "main", "hpc", "ANCESTOR_OF_MERGED_MAIN", 0, 2, "ANCESTOR", 2, 0),
                    "main": (1, 0, None, None, "CURRENT_CHECKOUT", None, None, "CANONICAL", 0, 0),
                    "origin/hpc": (0, 0, "main", "origin/hpc", "ANCESTOR_OF_MERGED_MAIN", 0, 2, "ANCESTOR", 2, 0),
                    "origin/main": (0, 0, "main", "origin/main", "ANCESTOR_OF_LOCAL_MAIN", 6, 0, "ANCESTOR", 6, 0),
                },
                "pairs": [("main", "hpc", "A_CONTAINS_B", 2, 0, "24a635f3da5bc782a7798e16148366ceaec9a799"),
                          ("main", "origin/hpc", "A_CONTAINS_B", 2, 0, "24a635f3da5bc782a7798e16148366ceaec9a799"),
                          ("main", "origin/main", "A_AHEAD_OF_B", 6, 0, "a73fc45e2035b83e7ed07978389090aa6c5e8004")],
                "remote_heads": {"main": "a73fc45e2035b83e7ed07978389090aa6c5e8004", "hpc": "24a635f3da5bc782a7798e16148366ceaec9a799"},
            },
        }

        for name, cfg in configs.items():
            snapshot = con.execute("SELECT head_sha,fetch_status,remote_tip_status FROM git_repository_snapshot WHERE repository_snapshot_id=? AND repository_id=?",
                                   (cfg["snapshot_id"], cfg["repo_id"])).fetchone()
            repo_path = con.execute("SELECT local_path FROM git_repository WHERE repository_id=?", (cfg["repo_id"],)).fetchone()[0]
            receipt_repo = receipt["repositories"][name]
            if not snapshot or snapshot["head_sha"] != receipt_repo["result_main"]:
                raise RuntimeError("Candidate snapshot HEAD differs from its merge receipt for " + name)
            if git(repo_path, "rev-parse", "HEAD") != snapshot["head_sha"] or git(repo_path, "status", "--porcelain=v1"):
                raise RuntimeError("Candidate clone changed or is dirty: " + name)
            ls_remote = dict(line.split("\t", 1)[::-1] for line in receipt_repo["ls_remote_stdout"].splitlines())
            for branch, sha in cfg["remote_heads"].items():
                if ls_remote.get("refs/heads/" + branch) != sha:
                    raise RuntimeError("Captured ls-remote head changed for " + name + "/" + branch)

            refs = con.execute("SELECT git_ref_observation_id,ref_name,commit_sha,ref_kind FROM git_ref_observation WHERE repository_id=? AND observed_utc=?",
                               (cfg["repo_id"], OBSERVED)).fetchall()
            ref_ids = {row["ref_name"]: row for row in refs}
            if set(ref_ids) != set(cfg["ref_context"]):
                raise RuntimeError("Candidate refs differ from the reviewed snapshot for " + name)

            remote_url = git(repo_path, "remote", "get-url", "origin")
            current_remote_names = {r[0] for r in con.execute("SELECT remote_name FROM git_remote_observation WHERE repository_snapshot_id=?", (cfg["snapshot_id"],))}
            if "origin" not in current_remote_names:
                con.execute(
                    """INSERT INTO git_remote_observation
                       (repository_snapshot_id,remote_name,fetch_url,push_url,fetch_status,current_tip_status,current_tip_sha,notes)
                       VALUES(?, 'origin', ?, ?, 'NOT_ATTEMPTED','UNKNOWN_NOT_VERIFIED',NULL,?)""",
                    (cfg["snapshot_id"], remote_url, remote_url,
                     "Remote ls-remote heads at 2026-11-22T22:10:04Z are preserved in source_id=" + str(source["source_id"]) +
                     "; this clone was not fetched or pushed, and no later remote-tip check is claimed."),
                )

            for ref_name, values in cfg["ref_context"].items():
                (is_current, local_only, target, source_ref, state, ahead, behind, relation, mainline_only, branch_only) = values
                ref_id = ref_ids[ref_name]["git_ref_observation_id"]
                upstream = "origin/" + ref_name if ref_ids[ref_name]["ref_kind"] == "BRANCH" else None
                if upstream:
                    con.execute("UPDATE git_ref_observation SET upstream_ref=? WHERE git_ref_observation_id=?", (upstream, ref_id))
                notes = ("Local isolated-clone ref ancestry derived with git rev-list/merge-base; "
                         "remote tracking refs are snapshot values at 2026-11-22T22:10:04Z, not a claim of current remote state.")
                con.execute(
                    """INSERT OR IGNORE INTO git_ref_context_snapshot
                       (git_ref_observation_id,repository_snapshot_id,source_id,is_current_checkout,is_local_only_ref,
                        comparison_target,comparison_source,comparison_state,ahead_commits,behind_commits,
                        local_mainline_ref,graph_relation_to_mainline,mainline_only_commits,branch_only_commits,
                        remote_tip_status,notes)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,'UNKNOWN_NOT_VERIFIED',?)""",
                    (ref_id, cfg["snapshot_id"], source["source_id"], is_current, local_only, target, source_ref,
                     state, ahead, behind, cfg["mainline"], relation, mainline_only, branch_only, notes),
                )

            for ref_a, ref_b, relation, a_only, b_only, base in cfg["pairs"]:
                actual = subprocess.run(["git", "-C", repo_path, "rev-list", "--left-right", "--count", ref_a + "..." + ref_b],
                                        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.split()
                actual_base = git(repo_path, "merge-base", ref_a, ref_b)
                if (int(actual[0]), int(actual[1]), actual_base) != (a_only, b_only, base):
                    raise RuntimeError("Branch-pair ancestry changed for " + name + ": " + ref_a + "..." + ref_b)
                con.execute(
                    """INSERT OR IGNORE INTO git_branch_pair_graph_snapshot
                       (repository_snapshot_id,ref_a,ref_b,relation,a_only_commits,b_only_commits,common_base_ref_or_oid,notes)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (cfg["snapshot_id"], ref_a, ref_b, relation, a_only, b_only, base,
                     "Computed locally from the clean isolated clone; remote refs remain unpushed snapshot values."),
                )

        if con.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Git context links failed the foreign-key check")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    with sqlite3.connect(DB) as verify:
        print("candidate_context_rows=" + str(verify.execute("SELECT count(*) FROM git_ref_context_snapshot WHERE repository_snapshot_id IN (43,44)").fetchone()[0]))
        print("candidate_remote_rows=" + str(verify.execute("SELECT count(*) FROM git_remote_observation WHERE repository_snapshot_id IN (43,44)").fetchone()[0]))
        print("candidate_branch_pair_rows=" + str(verify.execute("SELECT count(*) FROM git_branch_pair_graph_snapshot WHERE repository_snapshot_id IN (43,44)").fetchone()[0]))


if __name__ == "__main__":
    main()
