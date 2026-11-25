#!/usr/bin/env python3
"""Stage existing Rockfish Git snapshots into the additive provenance catalog.

Default mode validates the saved reports and prints intended row counts without
opening the catalog. Use --apply only after the catalog importer is finished
and this plan has been reviewed. Snapshot remote-tracking refs are cached local
data; this importer never contacts GitHub and never claims a live remote tip.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
AUDIT_DIR = HERE.parent
REPO_ROOT = HERE.parents[2]
PHASE2_DATA = AUDIT_DIR / "phase2_data"
CATALOG_DEFAULT = AUDIT_DIR / "provenance_catalog.sqlite"
MIGRATION_PATH = HERE / "004_git_snapshot_observations.sql"

RECON_JSONL = PHASE2_DATA / "git_reconciliation_review_20261122.jsonl"
RECON_TSV = PHASE2_DATA / "git_reconciliation_review_20261122.tsv"
BRANCH_TSV = PHASE2_DATA / "git_reconciliation_review_20261122_branches.tsv"
GRAPH_JSONL = PHASE2_DATA / "first_party_git_branch_graph_20261122.jsonl"
LAMMPS_JSON = PHASE2_DATA / "lammps_repo26_git_snapshot_20261122.json"

MIGRATION_VERSION = 4
MIGRATION_NAME = MIGRATION_PATH.name


def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def source_spec(path: Path, registry_kind: str, note: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Git snapshot source: {path}")
    checksum = sha256_file(path)
    observed = utc_iso(path.stat().st_mtime)
    key = f"{path.stem}_{checksum[:12]}"
    return {
        "source_key": key,
        "source_path": str(path.resolve()),
        "registry_kind": registry_kind,
        "sha256": checksum,
        "captured_utc": now_utc(),
        "as_of_utc": observed,
        "immutable_snapshot": 1,
        "authority_note": note,
    }


def parse_remotes(lines: list[str]) -> dict[str, dict[str, str | None]]:
    remotes: dict[str, dict[str, str | None]] = {}
    for line in lines:
        if "\t" not in line:
            continue
        name, value = line.split("\t", 1)
        kind = None
        if value.endswith(" (fetch)"):
            value, kind = value[:-8], "fetch_url"
        elif value.endswith(" (push)"):
            value, kind = value[:-7], "push_url"
        if kind is None:
            kind = "fetch_url"
        remotes.setdefault(name, {"fetch_url": None, "push_url": None})[kind] = value
    return remotes


def full_ref(name: str, kind: str) -> str:
    if name.startswith("refs/") or name == "HEAD":
        return name
    if kind == "BRANCH":
        return f"refs/heads/{name}"
    if kind == "REMOTE_BRANCH":
        return f"refs/remotes/{name}"
    return name


def int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def ownership_status(classification: str) -> str:
    low = classification.lower()
    if "third-party" in low or "vendor" in low or "dependency" in low:
        return "THIRD_PARTY"
    if "first-party" in low:
        return "USER"
    return "UNKNOWN"


def prepare_inputs() -> dict[str, Any]:
    base_repo_rows = read_jsonl(RECON_JSONL)
    recon_tsv_rows = read_tsv(RECON_TSV)
    base_branch_rows = read_tsv(BRANCH_TSV)
    graph_rows = read_jsonl(GRAPH_JSONL)
    lammps_row = json.loads(LAMMPS_JSON.read_text(encoding="utf-8"))

    base_repo_paths = [r["repo_path"] for r in base_repo_rows]
    if len(base_repo_paths) != len(set(base_repo_paths)):
        raise ValueError("Duplicate repo_path rows in reconciliation JSONL")
    if len(base_repo_rows) != 25:
        raise ValueError(f"Expected 25 base repository snapshots, found {len(base_repo_rows)}")
    if {r["repo_path"] for r in recon_tsv_rows} != set(base_repo_paths):
        raise ValueError("Repository TSV and JSONL path sets differ")

    base_branch_keys = {(r["repo_path"], r["branch"], r["oid"]) for r in base_branch_rows}
    if len(base_branch_keys) != len(base_branch_rows):
        raise ValueError("Duplicate repository/branch/SHA rows in local branches TSV")
    if len(base_branch_rows) != 30:
        raise ValueError(f"Expected 30 base local branch refs, found {len(base_branch_rows)}")
    if any(r["repo_path"] not in set(base_repo_paths) for r in base_branch_rows):
        raise ValueError("Branch TSV contains a repository absent from the 25-repo snapshot")

    expected_lammps_path = "/data/pclancy3/ycao73_home/flare/lammps-flare/lammps"
    if lammps_row.get("repo_path") != expected_lammps_path:
        raise ValueError(f"Repo 26 path mismatch: {lammps_row.get('repo_path')}")
    if lammps_row.get("head") != "b620c52b58f1b5810f3c78e23ab34283f3a2d700" or lammps_row.get("branch") != "develop":
        raise ValueError("Repo 26 commit/branch differs from the recorded LAMMPS snapshot")
    if lammps_row.get("tracked_dirty_entry_count") != 1 or lammps_row.get("untracked_file_entry_count") != 406 or lammps_row.get("status_entry_count") != 407:
        raise ValueError("Repo 26 tracked/untracked counts differ from its source-change manifest summary")
    source_manifest = Path(lammps_row["source_change_manifest_path"])
    if sha256_file(source_manifest) != lammps_row.get("source_change_manifest_sha256"):
        raise ValueError("Repo 26 source-change manifest SHA-256 differs from the supplemental snapshot")
    source_change_rows = read_jsonl(source_manifest)
    source_changes = [r for r in source_change_rows if r.get("record") == "source_file"]
    full_hashes = [r for r in source_changes if r.get("sha256")]
    symlink_targets = [r for r in source_changes if r.get("symlink_target")]
    if len(source_changes) != 407 or len(full_hashes) != 388 or len(symlink_targets) != 19:
        raise ValueError("Repo 26 source-change manifest coverage differs from 388 regular files + 19 symlinks")
    if any(not r.get("sha256") and not r.get("symlink_target") for r in source_changes):
        raise ValueError("Repo 26 contains a dirty source path without file hash or symlink target evidence")
    if sum(r.get("git_status") == " M" for r in source_changes) != 1 or sum(r.get("git_status") == "??" for r in source_changes) != 406:
        raise ValueError("Repo 26 source-change statuses do not match one tracked edit plus 406 untracked paths")
    if lammps_row.get("remote_tip_status") != "UNKNOWN_NOT_VERIFIED" or lammps_row.get("remote_current_tip_sha") is not None:
        raise ValueError("Repo 26 remote tip must remain unverified without a live query")

    lammps_branch_row = {
        "repo_path": expected_lammps_path,
        "branch": "develop",
        "oid": lammps_row["head"],
        "upstream": "origin/develop",
        "comparison_target": "origin/develop",
        "comparison_source": "cached_configured_upstream",
        "comparison": "up_to_date_cached",
        "ahead_commits": "0",
        "behind_commits": "0",
        "track_display": "",
        "local_only_ref": "False",
    }
    branch_rows = base_branch_rows + [lammps_branch_row]
    repo_rows = base_repo_rows + [lammps_row]
    repo_paths = [r["repo_path"] for r in repo_rows]
    branch_keys = {(r["repo_path"], r["branch"], r["oid"]) for r in branch_rows}
    if len(repo_paths) != 26 or len(set(repo_paths)) != 26:
        raise ValueError("Expected 26 distinct repositories after adding the LAMMPS repo 26 snapshot")
    if len(branch_rows) != 31 or len(branch_keys) != 31:
        raise ValueError("Expected 31 distinct local branch refs after adding LAMMPS develop")

    graph_paths = [r["repo_path"] for r in graph_rows]
    if len(graph_paths) != 16 or len(graph_paths) != len(set(graph_paths)):
        raise ValueError(f"Expected 16 distinct first-party graph snapshots, found {len(graph_paths)}")
    if not set(graph_paths).issubset(set(repo_paths)):
        raise ValueError("First-party graph report contains a repository absent from the 25-repo snapshot")
    graph_refs = {
        (r["repo_path"], b["name"], b["oid"])
        for r in graph_rows
        for b in r.get("branches", [])
    }
    if len(graph_refs) != 21:
        raise ValueError(f"Expected 21 local refs in first-party graph report, found {len(graph_refs)}")
    if not graph_refs.issubset(branch_keys):
        raise ValueError("First-party graph refs do not all match the 25-repo local-branch TSV")

    # Cross-check the simple TSV view against the richer JSONL status snapshot.
    tsv_by_path = {r["repo_path"]: r for r in recon_tsv_rows}
    for row in base_repo_rows:
        t = tsv_by_path[row["repo_path"]]
        for key_json, key_tsv in (
            ("head", "head"),
            ("branch", "branch"),
            ("tracked_dirty_entry_count", "tracked_dirty_entry_count"),
            ("untracked_file_entry_count", "untracked_file_entry_count"),
            ("ignored_matching_entry_count", "ignored_matching_entry_count"),
        ):
            if str(row.get(key_json, "")) != str(t.get(key_tsv, "")):
                raise ValueError(f"JSONL/TSV mismatch for {row['repo_path']} field {key_json}")

    files = [
        source_spec(RECON_JSONL, "GIT_REPOSITORY_STATUS_JSONL", "Rockfish /data/pclancy3/yi local repository status and cached refs; no live remote tips verified."),
        source_spec(RECON_TSV, "GIT_REPOSITORY_STATUS_TSV", "Tabular companion to the Rockfish local Git reconciliation snapshot."),
        source_spec(BRANCH_TSV, "GIT_LOCAL_BRANCH_REF_TSV", "Local branch OIDs and cached tracking-ref comparison from the Git reconciliation snapshot; not a live fetch."),
        source_spec(GRAPH_JSONL, "GIT_LOCAL_COMMIT_GRAPH_JSONL", "Read-only local commit-graph review for 16 first-party repositories; no refs changed or fetched."),
        source_spec(LAMMPS_JSON, "GIT_REPOSITORY_STATUS_SUPPLEMENTAL_JSON", "Supplemental snapshot of repo 26: shallow upstream LAMMPS base with user customization; cached refs only, current remote tips not verified."),
    ]
    cached_remote_tracking_refs = sum(len(r.get("remote_tracking_refs", [])) for r in repo_rows)
    remote_observations = sum(len(parse_remotes(r.get("remotes", [])).keys()) for r in repo_rows)
    if cached_remote_tracking_refs != 297:
        raise ValueError(f"Expected 297 cached remote-tracking refs after repo 26, found {cached_remote_tracking_refs}")
    if remote_observations != 25:
        raise ValueError(f"Expected 25 configured remote observations after repo 26, found {remote_observations}")
    return {
        "repo_rows": repo_rows,
        "branch_rows": branch_rows,
        "graph_rows": graph_rows,
        "source_specs": files,
        "counts": {
            "repository_rows": len(repo_rows),
            "local_branch_refs": len(branch_rows),
            "cached_remote_tracking_refs": cached_remote_tracking_refs,
            "remote_observations": remote_observations,
            "first_party_graph_repositories": len(graph_rows),
            "first_party_graph_branch_refs": len(graph_refs),
            "first_party_named_pair_comparisons": sum(len(r.get("recognized_branch_pairs", [])) for r in graph_rows),
        },
    }


def require_migration(connection: sqlite3.Connection) -> None:
    schema = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migration'"
    ).fetchone()
    if not schema:
        raise RuntimeError("Catalog schema is not initialized; run the base importer first")
    applied = connection.execute(
        "SELECT migration_file FROM schema_migration WHERE version=?", (MIGRATION_VERSION,)
    ).fetchone()
    if not applied:
        raise RuntimeError(
            f"Migration version {MIGRATION_VERSION} is not recorded as applied; "
            "this staging importer will not apply or replay migrations."
        )
    if applied[0] != MIGRATION_NAME:
        raise RuntimeError(f"Migration version {MIGRATION_VERSION} is already assigned to {applied[0]}")


def source_ids(connection: sqlite3.Connection, specs: list[dict[str, Any]]) -> dict[str, int]:
    result = {}
    for spec in specs:
        prior = connection.execute(
            "SELECT source_id,source_path,sha256 FROM registry_source WHERE source_key=?",
            (spec["source_key"],),
        ).fetchone()
        expected = (spec["source_path"], spec["sha256"])
        if prior and (prior[1], prior[2]) != expected:
            raise RuntimeError(f"Existing registry_source differs for {spec['source_key']}")
        connection.execute(
            """INSERT OR IGNORE INTO registry_source
               (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
                immutable_snapshot,authority_note)
               VALUES (?,?,?,?,?,?,?,?)""",
            (spec["source_key"], spec["source_path"], spec["registry_kind"], spec["sha256"],
             spec["captured_utc"], spec["as_of_utc"], spec["immutable_snapshot"], spec["authority_note"]),
        )
        result[spec["source_path"]] = connection.execute(
            "SELECT source_id FROM registry_source WHERE source_key=?", (spec["source_key"],)
        ).fetchone()[0]
    return result


def ensure_git_repository(connection: sqlite3.Connection, row: dict[str, Any]) -> int:
    path = row["repo_path"]
    key = "rockfish:" + path
    remotes = parse_remotes(row.get("remotes", []))
    origin = remotes.get("origin")
    remote_url = origin.get("fetch_url") if origin else next(
        (r.get("fetch_url") for r in remotes.values() if r.get("fetch_url")), None
    )
    owner = ownership_status(row.get("classification", ""))
    existing_path_rows = connection.execute(
        "SELECT repository_id,repository_key,remote_url,ownership_status FROM git_repository WHERE local_path=?",
        (path,),
    ).fetchall()
    if len(existing_path_rows) > 1:
        raise RuntimeError(f"Multiple catalog Git repositories already point at {path}")
    if existing_path_rows:
        repository_id = existing_path_rows[0][0]
        return repository_id
    project_id = row.get("project_id") or None
    if project_id:
        connection.execute(
            "INSERT OR IGNORE INTO project(project_id,project_name,owner_scope,description,status) VALUES (?,?,?,?,'ACTIVE')",
            (project_id, "FLARE and LAMMPS source", "USER",
             "User-customized LAMMPS/FLARE build project; upstream LAMMPS repository base remains third-party."),
        )
    connection.execute(
        """INSERT OR IGNORE INTO git_repository
           (repository_key,local_path,remote_url,project_id,ownership_status,canonical_branch,notes)
           VALUES (?,?,?,?,?,NULL,?)""",
        (key, path, remote_url, project_id, owner,
         json.dumps({"classification": row.get("classification"),
                     "classification_basis": row.get("classification_basis"),
                     "note": "Canonical/default branch intentionally not inferred from cached local refs."}, ensure_ascii=False)),
    )
    found = connection.execute(
        "SELECT repository_id,local_path FROM git_repository WHERE repository_key=?", (key,)
    ).fetchone()
    if not found or found[1] != path:
        raise RuntimeError(f"Unable to establish stable Git repository row for {path}")
    return found[0]


def insert_repo_snapshot(connection: sqlite3.Connection, repository_id: int, source_id: int,
                         row: dict[str, Any], observed_utc: str) -> int:
    status_ok = row.get("status_exit_code") == 0
    tracked = row.get("tracked_dirty_entry_count", row.get("tracked_dirty_entries"))
    untracked = row.get("untracked_file_entry_count", row.get("untracked_file_entries"))
    ignored = row.get("ignored_matching_entry_count", row.get("ignored_matching_entries"))
    worktree_dirty = None if not status_ok else int((tracked or 0) > 0 or (untracked or 0) > 0)
    remotes = parse_remotes(row.get("remotes", []))
    fetch_status = "NOT_RECORDED" if remotes else "NO_REMOTE_CONFIGURED"
    tip_status = "UNKNOWN_NOT_VERIFIED" if remotes else "NOT_APPLICABLE"
    head_ref = full_ref(row.get("branch") or "", "BRANCH") if row.get("branch") and row.get("branch") != "DETACHED" else "HEAD"
    connection.execute(
        """INSERT OR IGNORE INTO git_repository_snapshot
           (repository_id,source_id,observed_utc,head_ref,head_sha,status_exit_code,
            tracked_dirty_count,staged_dirty_count,unstaged_dirty_count,untracked_file_count,
            ignored_matching_count,worktree_dirty,status_error,fetch_status,remote_tip_status,
            shallow_clone,classification,classification_basis,parent_repo_key,parent_index_relation,notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (repository_id, source_id, observed_utc, head_ref, row.get("head"), row.get("status_exit_code"),
         tracked, row.get("staged_dirty_entry_count"), row.get("unstaged_dirty_entry_count"),
         untracked, ignored, worktree_dirty, row.get("status_error", ""), fetch_status, tip_status,
         int(row["shallow_clone"]) if row.get("shallow_clone") is not None else None,
         row.get("classification"), row.get("classification_basis"), row.get("parent_repo_path") or None,
         row.get("parent_index_relation"),
         "Only local Git metadata/cached tracking refs are represented; current remote tips are unknown."),
    )
    return connection.execute(
        "SELECT repository_snapshot_id FROM git_repository_snapshot WHERE repository_id=? AND source_id=?",
        (repository_id, source_id),
    ).fetchone()[0]


def insert_remotes(connection: sqlite3.Connection, snapshot_id: int, remote_lines: list[str]) -> None:
    remotes = parse_remotes(remote_lines)
    for name, urls in remotes.items():
        connection.execute(
            """INSERT OR IGNORE INTO git_remote_observation
               (repository_snapshot_id,remote_name,fetch_url,push_url,fetch_status,
                current_tip_status,current_tip_sha,notes)
               VALUES (?,?,?,?,?,?,NULL,?)""",
            (snapshot_id, name, urls.get("fetch_url"), urls.get("push_url"),
             "NOT_RECORDED", "UNKNOWN_NOT_VERIFIED",
             "No live fetch/ls-remote result is present in the source reports; remote-tracking OIDs are cached only."),
        )


def insert_ref(connection: sqlite3.Connection, repository_id: int, source_id: int,
               observed_utc: str, ref_name: str, ref_kind: str, commit_sha: str,
               upstream_ref: str | None, dirty_worktree: int | None, notes: str) -> int:
    row = connection.execute(
        """SELECT git_ref_observation_id FROM git_ref_observation
           WHERE repository_id=? AND observed_utc=? AND ref_name=? AND ref_kind=? AND commit_sha=?""",
        (repository_id, observed_utc, ref_name, ref_kind, commit_sha),
    ).fetchone()
    if row:
        return row[0]
    connection.execute(
        """INSERT INTO git_ref_observation
           (repository_id,observed_utc,ref_name,ref_kind,commit_sha,upstream_ref,
            is_merged_to_canonical,dirty_worktree,notes)
           VALUES (?,?,?,?,?,?,NULL,?,?)""",
        (repository_id, observed_utc, ref_name, ref_kind, commit_sha,
         upstream_ref, dirty_worktree, notes),
    )
    return connection.execute("SELECT last_insert_rowid()").fetchone()[0]


def insert_ref_context(connection: sqlite3.Connection, ref_id: int, snapshot_id: int, source_id: int,
                       *, current: int | None, local_only: int | None,
                       target: str | None = None, comparison_source: str | None = None,
                       comparison_state: str | None = None, ahead: int | None = None,
                       behind: int | None = None, mainline: str | None = None,
                       graph_relation: str | None = None, mainline_only: int | None = None,
                       branch_only: int | None = None, remote_tip_status: str = "UNKNOWN_NOT_VERIFIED",
                       notes: str = "") -> None:
    connection.execute(
        """INSERT OR IGNORE INTO git_ref_context_snapshot
           (git_ref_observation_id,repository_snapshot_id,source_id,is_current_checkout,
            is_local_only_ref,comparison_target,comparison_source,comparison_state,ahead_commits,
            behind_commits,local_mainline_ref,graph_relation_to_mainline,mainline_only_commits,
            branch_only_commits,remote_tip_status,notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (ref_id, snapshot_id, source_id, current, local_only, target, comparison_source,
         comparison_state, ahead, behind, mainline, graph_relation, mainline_only,
         branch_only, remote_tip_status, notes),
    )


def apply_catalog(catalog: Path, data: dict[str, Any]) -> None:
    connection = sqlite3.connect(catalog, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        require_migration(connection)
        connection.execute("BEGIN IMMEDIATE")
        # Re-hash input snapshots just before write to prevent staging stale/different bytes.
        for spec in data["source_specs"]:
            actual = sha256_file(Path(spec["source_path"]))
            if actual != spec["sha256"]:
                raise RuntimeError(f"Snapshot changed after validation: {spec['source_path']}")
        source_by_path = {s["source_path"]: s for s in data["source_specs"]}
        source_id_by_path = source_ids(connection, data["source_specs"])
        recon_spec = source_by_path[str(RECON_JSONL.resolve())]
        recon_tsv_spec = source_by_path[str(RECON_TSV.resolve())]
        branches_spec = source_by_path[str(BRANCH_TSV.resolve())]
        graph_spec = source_by_path[str(GRAPH_JSONL.resolve())]

        repo_by_path = {r["repo_path"]: r for r in data["repo_rows"]}
        branch_rows_by_repo: dict[str, list[dict[str, str]]] = {}
        for row in data["branch_rows"]:
            branch_rows_by_repo.setdefault(row["repo_path"], []).append(row)
        graph_by_path = {r["repo_path"]: r for r in data["graph_rows"]}

        repository_ids: dict[str, int] = {}
        primary_snapshots: dict[str, int] = {}
        branch_snapshot_ids: dict[str, int] = {}

        for path, row in repo_by_path.items():
            repository_id = ensure_git_repository(connection, row)
            repository_ids[path] = repository_id
            snap_id = insert_repo_snapshot(
                connection, repository_id, source_id_by_path[recon_spec["source_path"]], row,
                recon_spec["as_of_utc"],
            )
            primary_snapshots[path] = snap_id
            insert_remotes(connection, snap_id, row.get("remotes", []))

            branch_by_key = {(b["branch"], b["oid"]): b for b in branch_rows_by_repo.get(path, [])}
            dirty = None if row.get("status_exit_code") != 0 else int(
                row.get("tracked_dirty_entry_count", 0) > 0 or row.get("untracked_file_entry_count", 0) > 0
            )
            for name, oid, upstream in [
                (b["name"], b["oid"], b.get("upstream") or None)
                for b in row.get("local_branches", [])
            ]:
                if (name, oid) not in branch_by_key:
                    raise RuntimeError(f"JSONL/branch TSV ref mismatch: {path} {name} {oid}")
            for b in branch_rows_by_repo.get(path, []):
                upstream = b.get("upstream") or None
                upstream_full = full_ref(upstream, "REMOTE_BRANCH") if upstream else None
                ref_id = insert_ref(
                    connection, repository_id, source_id_by_path[branches_spec["source_path"]],
                    branches_spec["as_of_utc"], full_ref(b["branch"], "BRANCH"), "BRANCH", b["oid"],
                    upstream_full, dirty, "Local branch OID from saved read-only Git reconciliation; cached-upstream comparison only.",
                )
                branch_snapshot_ids[path + "\0" + b["branch"]] = ref_id
                target = full_ref(b["comparison_target"], "REMOTE_BRANCH") if b.get("comparison_target") else None
                insert_ref_context(
                    connection, ref_id, snap_id, source_id_by_path[branches_spec["source_path"]],
                    current=int(row.get("branch") == b["branch"]),
                    local_only=int(str(b.get("local_only_ref", "")).lower() == "true"),
                    target=target, comparison_source=b.get("comparison_source") or None,
                    comparison_state=b.get("comparison") or None,
                    ahead=int_or_none(b.get("ahead_commits")),
                    behind=int_or_none(b.get("behind_commits")),
                    remote_tip_status="UNKNOWN_NOT_VERIFIED" if upstream else "NOT_APPLICABLE",
                    notes="Comparison target is a cached local tracking ref, not a freshly fetched server tip.",
                )

            for remote_ref in row.get("remote_tracking_refs", []):
                name = remote_ref["name"]
                ref_id = insert_ref(
                    connection, repository_id, source_id_by_path[recon_spec["source_path"]],
                    recon_spec["as_of_utc"], full_ref(name, "REMOTE_BRANCH"), "REMOTE_BRANCH",
                    remote_ref["oid"], None, dirty,
                    "Cached local remote-tracking ref; current remote tip not verified. origin/HEAD remains a symbolic alias in the source snapshot.",
                )
                insert_ref_context(
                    connection, ref_id, snap_id, source_id_by_path[recon_spec["source_path"]],
                    current=0, local_only=0, comparison_source="cached_remote_tracking_ref",
                    comparison_state="CACHED_ONLY", remote_tip_status="UNKNOWN_NOT_VERIFIED",
                    notes="OID is present in local Git metadata only; do not treat it as the current server tip.",
                )

            if row.get("branch") == "DETACHED" and row.get("head"):
                ref_id = insert_ref(
                    connection, repository_id, source_id_by_path[recon_spec["source_path"]],
                    recon_spec["as_of_utc"], "HEAD", "DETACHED", row["head"], None,
                    dirty, "Detached HEAD from saved local Git reconciliation snapshot.",
                )
                insert_ref_context(connection, ref_id, snap_id, source_id_by_path[recon_spec["source_path"]],
                                   current=1, local_only=None, remote_tip_status="UNKNOWN_NOT_VERIFIED")

        for path, graph in graph_by_path.items():
            row = repo_by_path[path]
            repository_id = repository_ids[path]
            snap_id = insert_repo_snapshot(
                connection, repository_id, source_id_by_path[graph_spec["source_path"]], graph,
                graph_spec["as_of_utc"],
            )
            dirty = None if graph.get("status_exit_code") != 0 else int(
                graph.get("tracked_dirty_entries", 0) > 0 or graph.get("untracked_file_entries", 0) > 0
            )
            for b in graph.get("branches", []):
                upstream = b.get("upstream") or None
                ref_name = full_ref(b["name"], "BRANCH")
                ref_id = insert_ref(
                    connection, repository_id, source_id_by_path[graph_spec["source_path"]],
                    graph_spec["as_of_utc"], ref_name, "BRANCH", b["oid"],
                    full_ref(upstream, "REMOTE_BRANCH") if upstream else None, dirty,
                    "Local branch OID from later first-party local commit-graph snapshot; no fetch or ref mutation.",
                )
                comp = next((x for x in graph.get("branches_vs_mainline", []) if x.get("branch") == b["name"]), None)
                if comp:
                    relation = comp.get("relation")
                    state = relation
                    ahead = comp.get("b_only_commits")
                    behind = comp.get("a_only_commits")
                    if relation == "no_local_mainline":
                        ahead = behind = None
                    graph_notes = "Local graph only; mainline chosen by audit as local main, else local master."
                    insert_ref_context(
                        connection, ref_id, snap_id, source_id_by_path[graph_spec["source_path"]],
                        current=int(b["name"] == graph.get("current_branch")),
                        local_only=None, target=full_ref(graph.get("mainline_branch"), "BRANCH") if graph.get("mainline_branch") else None,
                        comparison_source="local_commit_graph", comparison_state=state,
                        ahead=ahead, behind=behind,
                        mainline=full_ref(graph.get("mainline_branch"), "BRANCH") if graph.get("mainline_branch") else None,
                        graph_relation=relation, mainline_only=comp.get("a_only_commits"),
                        branch_only=comp.get("b_only_commits"),
                        remote_tip_status="UNKNOWN_NOT_VERIFIED" if upstream else "NOT_APPLICABLE",
                        notes=graph_notes,
                    )
                else:
                    insert_ref_context(
                        connection, ref_id, snap_id, source_id_by_path[graph_spec["source_path"]],
                        current=int(b["name"] == graph.get("current_branch")), local_only=None,
                        comparison_source="local_commit_graph", comparison_state="NO_COMPARISON_ROW",
                        remote_tip_status="UNKNOWN_NOT_VERIFIED" if upstream else "NOT_APPLICABLE",
                        notes="No comparable main/master baseline in the graph source." )
            for pair in graph.get("recognized_branch_pairs", []):
                connection.execute(
                    """INSERT OR IGNORE INTO git_branch_pair_graph_snapshot
                       (repository_snapshot_id,ref_a,ref_b,relation,a_only_commits,b_only_commits,
                        common_base_ref_or_oid,notes)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (snap_id, full_ref(pair["ref_a"], "BRANCH"), full_ref(pair["ref_b"], "BRANCH"),
                     pair.get("relation", "UNKNOWN"), pair.get("a_only_commits"), pair.get("b_only_commits"),
                     pair.get("common_base_oid"), "Raw audit field common_base_oid; may be a ref name for identical refs, otherwise a commit OID. Local merge-base/rev-list comparison; no remote state."),
                )

        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(f"Foreign-key violations after staging: {violations[:5]}")
        connection.commit()
        print(json.dumps({"catalog": str(catalog), "status": "APPLIED", "counts": data["counts"]}, indent=2))
    except Exception:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG_DEFAULT,
                        help="Catalog to write; used only with --apply")
    parser.add_argument("--apply", action="store_true",
                        help="Explicitly apply migration 004 and append the saved Git observations")
    args = parser.parse_args()

    data = prepare_inputs()
    output = {"mode": "APPLY_REQUESTED" if args.apply else "DRY_RUN_ONLY",
              "inputs": [{"source_path": s["source_path"], "sha256": s["sha256"],
                          "observed_utc_from_file_mtime": s["as_of_utc"],
                          "registry_kind": s["registry_kind"]} for s in data["source_specs"]],
              "counts": data["counts"],
              "remote_state_policy": "No live remote query is performed; current remote tips remain UNKNOWN_NOT_VERIFIED."}
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if args.apply:
        if not args.catalog.is_file():
            raise SystemExit(f"Catalog not found; importer must complete first: {args.catalog}")
        apply_catalog(args.catalog, data)


if __name__ == "__main__":
    main()
