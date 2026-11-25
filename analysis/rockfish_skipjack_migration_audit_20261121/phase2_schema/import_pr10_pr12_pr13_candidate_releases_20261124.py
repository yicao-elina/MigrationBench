#!/usr/bin/env python3
"""Add PR #10/#12/#13 evidence as separate review candidates in the audit catalog.

The importer is additive and idempotent. It imports hashes and pinned source
locations from the captured PR manifests; it never imports payload bytes,
changes the submitted Revision 1 release, or upgrades a source status based on
the fact that a file or Git commit exists.

Run from any directory:

  python3 import_pr10_pr12_pr13_candidate_releases_20261124.py \
    --db /path/to/provenance_catalog.sqlite \
    --snapshot-dir /path/to/phase2_data/pr_release_candidates_20261124
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import quote


AUDIT = Path(__file__).resolve().parent.parent
WORKTREE = AUDIT.parents[1]
DEFAULT_DB = AUDIT / "provenance_catalog.sqlite"
DEFAULT_SNAPSHOT_DIR = AUDIT / "phase2_data/pr_release_candidates_20261124"
REPOSITORY = "yicao-elina/MigrationBench"
ROCKFISH_REPO_KEY = "rockfish:/scratch16/pclancy3/yi/MigrationBench"
LOCAL_ROOT_ID = "LOCAL-MAIN"
ROCKFISH_ROOT_ID = "RF-SCRATCH16-YI"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PR10_HASH_DEFECT = {
    "commit": "51cb06f71732e46cd10eb92cee1f0c5f8f4833c9",
    "repository_path": "data/raw_compact/revision1/from_scratch_failure_attribution_20261013/dft_batch_selection_remote.csv",
    "relative_name": "dft_batch_selection_remote.csv",
    "source_declared_sha256": "78c1b5038761a14960836dccdd7899b77c3e4cfe9cc9464ce4f253315d207c6d1",
    "expected_blob_oid": "87ada3e542c7b376d25ba3222d09e7c1847b0fbd",
    "expected_byte_size": 29197,
    "expected_recomputed_sha256": "78c1b5038761a14960836dccdd7899b77c3e4cfe9cc946ce4f253315d207c6d1",
}
PR13_TREE_AUDIT_FILENAME = "pr13_git_tree_reconciliation_20261124.json"
PR13_TREE_AUDIT_SHA256 = "7f4ceb1f21ef1a611fe0aaa7808f4e5d0655a67de26a95a5d47d428cfe9fc3c6"
PR13_TREE_REPORT_FILENAME = "PR13_GIT_TREE_RECONCILIATION_20261124.md"
PR13_TREE_REPORT_SHA256 = "06e9834315ea2148aad09ec8e8d132c514ee054a2dfa84602147b57251686d9b"
PR13_PACKAGE_PREFIX = "analysis/dft_mace_47_site_error_attribution_20261023"
PR13_ARCHIVE_CAPTURE_DIRNAME = "pr13_raw_archive_capture_20261124"
PR13_ARCHIVE_ROOT = "/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_pr13_endpoint47_raw_evidence"
PR13_ARCHIVE_RELATIVE_ROOT = "MigrationBench_data_archive/revision1_pr13_endpoint47_raw_evidence"
PR13_ARCHIVE_ROOT_STORAGE_PATH = "/scratch16/pclancy3/yi"
PR13_ARCHIVE_MANIFEST_SHA256 = "438b2255565a8d8fe29620f655fecf742768d526fcbe5df4512e45b7c4bfb142"
PR13_ARCHIVE_REPORT_SHA256 = "a6edb301e70e7c696187b301e2d7cde1c96260d8359d28299e5bf37110b12cdf"
PR13_ARCHIVE_README_SHA256 = "5601cc2d3b921e5dfe20ed9823f9b2e946fa7424a059c6d4bb35e7bea6489891"
PR13_RAW_SOURCE_MANIFEST_PATH = f"{PR13_PACKAGE_PREFIX}/tables/FILE_PROVENANCE_MANIFEST.tsv"
PR13_RAW_SOURCE_BUNDLE_PATH = f"{PR13_PACKAGE_PREFIX}/raw_evidence_bundle.tar.gz"
PR13_RAW_SOURCE_BUNDLE_SHA256 = "4930f34a9935d1e368a49c3081a8a1185e3418a2d30761a0a911660d9dfad7be"
PR13_RAW_SOURCE_BUNDLE_OID = "b7f65ae69e7a18afd3a555d259568ffc716d7a6e"
PR13_RAW_SOURCE_BUNDLE_BYTES = 3237382


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def github_blob_url(commit: str, relative_path: str) -> str:
    normalized = PurePosixPath(relative_path).as_posix()
    if normalized.startswith("/") or ".." in PurePosixPath(normalized).parts:
        raise ValueError(f"Not a repository-relative package path: {relative_path}")
    return f"https://github.com/{REPOSITORY}/blob/{commit}/{quote(normalized, safe='/@+=,-._~') }"


def role_for_path(path: str, declared_role: str = "") -> str:
    value = path.lower()
    if "claim_evidence_matrix" in value or "claim_matrix" in value:
        return "SOURCE_CLAIM_MATRIX"
    if "release_manifest" in value or "manifest" in value:
        return "SOURCE_MANIFEST"
    if "figure" in value and value.endswith((".pdf", ".png", ".svg")):
        return "FIGURE_EXPORT"
    if "table" in value or value.endswith((".csv", ".tsv")):
        return "COMPACT_TABLE_OR_DATA"
    if value.endswith((".py", ".slurm", ".sh")):
        return "REBUILD_SOURCE"
    if declared_role:
        return declared_role.upper()
    return "PACKAGE_EVIDENCE"


def conservative_claim_status(source_status: str) -> tuple[str, str]:
    """Keep source status elsewhere; catalog assessment remains conservative."""
    value = (source_status or "").strip().upper()
    if value == "EXCLUDE_FROM_PUBLICATION":
        return "EXCLUDE_FROM_PUBLICATION", "EXCLUDE"
    if value == "FAILURE_MODE":
        return "FAILURE_MODE", "QUALIFY"
    if value.startswith("RECONSTRUCTION"):
        return "RECONSTRUCTION", "QUALIFY"
    # The catalog import checks lineage and hash declarations only; it does
    # not reproduce the calculations or independently pass scientific gates.
    return "PROVENANCE_PARTIAL", "PENDING"


class Importer:
    def __init__(self, db: sqlite3.Connection, snapshot_dir: Path, capture_rows: list[dict[str, str]]):
        self.db = db
        self.snapshot_dir = snapshot_dir.resolve(strict=True)
        self.capture_rows = capture_rows
        self.captures: dict[str, dict] = {}
        self.sources: dict[str, int] = {}
        self.capture_artifacts: dict[str, int] = {}
        self.artifacts_by_digest: dict[str, int] = {}
        self.package_file_artifacts: dict[str, dict[str, int | None]] = {}
        self.releases: dict[int, dict] = {}
        self.counts: dict[str, int] = {}

    def add_count(self, key: str, value: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + value

    def capture(self, filename: str) -> dict:
        if filename not in self.captures:
            path = self.snapshot_dir / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            rows = [row for row in self.capture_rows if row["capture_file"] == filename]
            if len(rows) != 1:
                raise ValueError(f"Expected one CAPTURE_MANIFEST row for {filename}, got {len(rows)}")
            row = rows[0]
            actual_hash = sha256(path)
            actual_size = path.stat().st_size
            if actual_hash != row["sha256"] or actual_size != int(row["byte_size"]):
                raise RuntimeError(f"Capture identity mismatch for {filename}")
            row = dict(row)
            row["local_path"] = path
            row["capture_sha256"] = actual_hash
            row["capture_size"] = actual_size
            self.captures[filename] = row
        return self.captures[filename]

    def add_source(self, filename: str, registry_kind: str, note: str) -> int:
        capture = self.capture(filename)
        pr = capture["pr_number"]
        source_key = f"migrationbench_pr{pr}:{filename}:{capture['head_sha'][:16]}:{capture['capture_sha256'][:12]}"
        url = capture["source_url"]
        captured_utc = capture["capture_utc"]
        metadata = load_json(self.snapshot_dir / f"pr{pr}_metadata.json")
        as_of = metadata.get("updatedAt") or captured_utc
        self.db.execute(
            """INSERT OR IGNORE INTO registry_source
               (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
                immutable_snapshot,authority_note,notes)
               VALUES(?,?,?,?,?,?,1,?,?)""",
            (source_key, url, registry_kind, capture["capture_sha256"], captured_utc, as_of,
             f"Pinned GitHub PR #{pr} source at commit {capture['head_sha']}; capture content SHA-256 verified locally.",
             note),
        )
        source = self.db.execute(
            "SELECT source_id,source_path,sha256 FROM registry_source WHERE source_key=?", (source_key,)
        ).fetchone()
        if not source or source[1] != url or source[2] != capture["capture_sha256"]:
            raise RuntimeError(f"Registry source identity conflict: {source_key}")
        self.sources[filename] = int(source[0])

        artifact_id = self.add_artifact(
            digest=capture["capture_sha256"], size=capture["capture_size"],
            kind="PR_SOURCE_CAPTURE", role=f"PR #{pr} source snapshot {filename}",
            note=f"Byte-identical local capture of {url}; input to the additive catalog import.",
        )
        local_root = self.db.execute(
            "SELECT scoped_path FROM storage_root WHERE root_id=?", (LOCAL_ROOT_ID,)
        ).fetchone()
        if not local_root or Path(local_root[0]).resolve() != WORKTREE.resolve():
            raise RuntimeError("LOCAL-MAIN does not resolve to the expected unified worktree")
        relative = capture["local_path"].resolve().relative_to(WORKTREE.resolve()).as_posix()
        self.add_artifact_location(artifact_id, LOCAL_ROOT_ID, relative, captured_utc)
        self.capture_artifacts[filename] = artifact_id
        self.add_count("capture_artifacts")
        return int(source[0])

    def add_artifact(self, digest: str | None, size: int | None, kind: str, role: str, note: str) -> int | None:
        if digest is None:
            return None
        if not SHA256_RE.fullmatch(digest):
            raise ValueError(f"Invalid SHA-256: {digest}")
        key = f"MB-PR-CONTENT-SHA256:{digest}"
        self.db.execute(
            """INSERT OR IGNORE INTO artifact
               (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
                scientific_status,retention_class,notes)
               VALUES('MIGRATIONBENCH',?,?,?,'USER_CONFIRMED',?,?,'PROVENANCE_PARTIAL','PUBLICATION_EVIDENCE',?)""",
            (key, kind, role, digest, size, note),
        )
        row = self.db.execute(
            "SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (key,)
        ).fetchone()
        if not row or row[1] != digest or (size is not None and row[2] is not None and row[2] != size):
            raise RuntimeError(f"Artifact identity conflict for {key}")
        if size is not None and row[2] is None:
            # Fill a previously unknown byte count only when a captured file
            # supplied the exact size for the same content SHA-256.
            self.db.execute("UPDATE artifact SET byte_size=? WHERE artifact_id=? AND byte_size IS NULL",(size,row[0]))
        self.artifacts_by_digest[digest] = int(row[0])
        self.add_count("artifact_rows_touched")
        return int(row[0])

    def add_artifact_location(self, artifact_id: int, root_id: str, relative_path: str, verified_utc: str) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)",
            (root_id, relative_path, verified_utc),
        )
        path_row = self.db.execute(
            "SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (root_id, relative_path)
        ).fetchone()
        if not path_row:
            raise RuntimeError(f"Could not register storage path {root_id}:{relative_path}")
        self.db.execute(
            """INSERT OR IGNORE INTO artifact_location
               (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
               VALUES(?,?,'PRESENT',?,?)""",
            (artifact_id, int(path_row[0]), verified_utc, verified_utc),
        )

    def add_entity_ref(self, source_id: int, entity_type: str, entity_key: str,
                       target_type: str, target_key: str, relationship: str, note: str) -> None:
        self.db.execute(
            """INSERT OR IGNORE INTO registry_entity_ref
               (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
               VALUES(?,?,?,?,?,?,?)""",
            (source_id, entity_type, entity_key, target_type, target_key, relationship, note),
        )

    def add_git_observation(self, pr_number: int, metadata: dict, captured_utc: str) -> None:
        repo = self.db.execute(
            "SELECT repository_id,remote_url FROM git_repository WHERE repository_key=?", (ROCKFISH_REPO_KEY,)
        ).fetchone()
        if not repo or repo[1] != f"https://github.com/{REPOSITORY}.git":
            raise RuntimeError("Expected Rockfish MigrationBench repository registration is missing or changed")
        head = metadata["headRefOid"]
        if not re.fullmatch(r"[0-9a-f]{40}", head):
            raise ValueError(f"PR #{pr_number} head SHA is not a full Git commit: {head}")
        ref = f"refs/remotes/origin/{metadata['headRefName']}"
        notes = (
            f"GitHub PR #{pr_number} observed {metadata['state']} at {captured_utc}; "
            f"base={metadata['baseRefName']}; title={metadata['title']}; "
            f"PR metadata updatedAt={metadata['updatedAt']}. Not a merge or manuscript release."
        )
        exists = self.db.execute(
            """SELECT 1 FROM git_ref_observation WHERE repository_id=? AND observed_utc=?
               AND ref_name=? AND commit_sha=? AND notes=? LIMIT 1""",
            (repo[0], captured_utc, ref, head, notes),
        ).fetchone()
        if not exists:
            self.db.execute(
                """INSERT INTO git_ref_observation
                   (repository_id,observed_utc,ref_name,ref_kind,commit_sha,upstream_ref,
                    is_merged_to_canonical,dirty_worktree,notes)
                   VALUES(?,?,?,'REMOTE_BRANCH',?,'refs/heads/main',0,NULL,?)""",
                (repo[0], captured_utc, ref, head, notes),
            )
            self.add_count("git_ref_observations")

    def add_release(self, release_id: str, pr_number: int, metadata: dict, manifest_artifact_id: int,
                    source_manifest: dict, note: str) -> None:
        repo = self.db.execute(
            "SELECT repository_id FROM git_repository WHERE repository_key=?", (ROCKFISH_REPO_KEY,)
        ).fetchone()
        if not repo:
            raise RuntimeError("Rockfish MigrationBench repository registration is missing")
        if pr_number == 13:
            package_commit = source_manifest["bundle_commit"]
        else:
            package_commit = metadata["headRefOid"]
        notes = (
            f"GitHub PR #{pr_number} state={metadata['state']}; source branch={metadata['headRefName']} "
            f"head={metadata['headRefOid']}; package commit={package_commit}. {note}"
        )
        self.db.execute(
            """INSERT OR IGNORE INTO paper_release
               (release_id,project_id,release_label,manuscript_commit,repository_id,
                overleaf_version,released_utc,status,manifest_artifact_id,notes)
               VALUES(?,'MIGRATIONBENCH',?,?,?,NULL,NULL,'REVIEW',?,?)""",
            (release_id, f"PR #{pr_number} review candidate: {metadata['title']}",
             package_commit, repo[0], manifest_artifact_id, notes),
        )
        row = self.db.execute(
            "SELECT project_id,release_label,manuscript_commit,repository_id,status,manifest_artifact_id,notes "
            "FROM paper_release WHERE release_id=?", (release_id,),
        ).fetchone()
        expected = ("MIGRATIONBENCH", f"PR #{pr_number} review candidate: {metadata['title']}",
                    package_commit, repo[0], "REVIEW", manifest_artifact_id, notes)
        if not row or tuple(row) != expected:
            raise RuntimeError(f"Existing candidate release differs from source snapshot: {release_id}")
        self.releases[pr_number] = {"release_id": release_id, "manifest_artifact_id": manifest_artifact_id}
        self.add_count("review_releases")

    def add_package_index_item(self, release_id: str, content: str, captured_utc: str) -> int:
        self.db.execute(
            """INSERT OR IGNORE INTO paper_release_item
               (release_id,source_item_id,section,panel,content,source_status,notes)
               VALUES(?,'PACKAGE_FILE_INDEX','package evidence','all',?,'HASH_INDEXED',?)""",
            (release_id, content,
             "Content-addressed file index from the pinned PR source manifests. Payload bytes are not copied."),
        )
        row = self.db.execute(
            """SELECT release_item_id,content,source_status,notes FROM paper_release_item
               WHERE release_id=? AND source_item_id='PACKAGE_FILE_INDEX' AND section='package evidence' AND panel='all'""",
            (release_id,),
        ).fetchone()
        expected_notes = "Content-addressed file index from the pinned PR source manifests. Payload bytes are not copied."
        if not row or tuple(row[1:]) != (content, "HASH_INDEXED", expected_notes):
            raise RuntimeError(f"Package index item conflict for {release_id}")
        return int(row[0])

    def add_package_file(self, *, pr_number: int, release_id: str, item_id: int, source_id: int,
                         commit: str, repository_path: str, digest: str | None,
                         size: int | None, declared_role: str, note: str) -> int | None:
        if repository_path.startswith("/") or ".." in PurePosixPath(repository_path).parts:
            raise ValueError(f"Unsafe package path: {repository_path}")
        url = github_blob_url(commit, repository_path)
        role = role_for_path(repository_path, declared_role)
        artifact_id = self.add_artifact(
            digest, size, "MIGRATIONBENCH_PR_PACKAGE_FILE", role,
            f"PR #{pr_number}; pinned source={url}; {note}",
        )
        self.db.execute(
            """INSERT OR IGNORE INTO paper_release_item_evidence
               (release_item_id,artifact_id,source_path,evidence_role,availability,relationship,notes)
               VALUES(?,?,?,'PACKAGE_FILE','EXTERNAL_POINTER','PINNED_GITHUB_COMMIT_PATH',?)""",
            (item_id, artifact_id, url,
             f"sha256={digest or 'NOT_DECLARED'}; bytes={size if size is not None else 'NOT_DECLARED'}; "
             f"manifest_role={declared_role}; {note}"),
        )
        self.add_entity_ref(
            source_id, "package_file", repository_path,
            "ARTIFACT" if artifact_id is not None else "EXTERNAL_POINTER",
            f"MB-PR-CONTENT-SHA256:{digest}" if artifact_id is not None else url,
            "PINNED_GITHUB_COMMIT_FILE",
            f"PR #{pr_number}; commit={commit}; sha256={digest or 'NOT_DECLARED'}; "
            f"bytes={size if size is not None else 'NOT_DECLARED'}; role={declared_role}.",
        )
        self.package_file_artifacts.setdefault(release_id, {})[repository_path] = artifact_id
        self.add_count("package_files_indexed")
        return artifact_id

    def add_source_capture_evidence(self, release_id: str, item_id: int, filename: str,
                                    source_id: int, captured_utc: str) -> None:
        artifact_id = self.capture_artifacts[filename]
        capture = self.capture(filename)
        local_rel = capture["local_path"].resolve().relative_to(WORKTREE.resolve()).as_posix()
        self.db.execute(
            """INSERT OR IGNORE INTO paper_release_item_evidence
               (release_item_id,artifact_id,source_path,evidence_role,availability,relationship,notes)
               VALUES(?,?,?,'SOURCE_CAPTURE','PRESENT','CAPTURED_PINNED_SOURCE_BYTES',?)""",
            (item_id, artifact_id, local_rel,
             f"Captured {capture['source_url']} at {capture['head_sha']}; "
             f"sha256={capture['capture_sha256']}; registry_source_id={source_id}; verified in local audit tree."),
        )

    def add_figure(self, *, figure_id: str, release_id: str, source_id: int, title: str,
                   chart_kind: str, chart_status: str, scientific_boundary: str,
                   assets: list[tuple[str, str, int | None, str]]) -> None:
        self.db.execute(
            """INSERT OR IGNORE INTO paper_figure
               (figure_id,project_id,title,chart_kind,chart_status,provenance_status,
                source_registry_source_id,scientific_boundary,notes)
               VALUES(?,'MIGRATIONBENCH',?,?,?,'PROVENANCE_PARTIAL',?,?,?)""",
            (figure_id, title, chart_kind, chart_status, source_id, scientific_boundary,
             f"Separate review candidate for {release_id}; not merged into the submitted release."),
        )
        row = self.db.execute(
            "SELECT project_id,title,chart_kind,chart_status,provenance_status,source_registry_source_id,scientific_boundary "
            "FROM paper_figure WHERE figure_id=?", (figure_id,),
        ).fetchone()
        if not row or tuple(row) != ("MIGRATIONBENCH",title,chart_kind,chart_status,"PROVENANCE_PARTIAL",source_id,scientific_boundary):
            raise RuntimeError(f"Figure record conflict: {figure_id}")
        for asset_role, source_path, artifact_id, note in assets:
            if asset_role not in {"FIGURE_EXPORT","DERIVED_DATA","PROVENANCE_MANIFEST","CLAIM_MATRIX","SOURCE_ASSET","SOURCE_MANIFEST","REBUILD_METHOD","ARCHIVE_INDEX"}:
                raise ValueError(f"Invalid paper_figure_asset role: {asset_role}")
            url_or_path = source_path
            availability = "EXTERNAL_POINTER" if source_path.startswith("https://") or source_path.startswith("/") else "EXTERNAL_POINTER"
            self.db.execute(
                """INSERT OR IGNORE INTO paper_figure_asset
                   (figure_id,artifact_id,asset_role,source_path,availability,relationship,notes)
                   VALUES(?,?,?,?,?,'PINNED_PR_SOURCE',?)""",
                (figure_id, artifact_id, asset_role, url_or_path, availability,
                 note + "; no candidate payload was copied by this importer."),
            )
            self.add_count("figure_asset_links")
        self.add_count("paper_figures")

    def add_release_item(self, release_id: str, source_item_id: str, section: str, panel: str,
                         content: str, final_path: str | None, final_hash: str | None,
                         generator: str | None, plotted_data: str | None, source_status: str,
                         notes: str) -> int:
        self.db.execute(
            """INSERT OR IGNORE INTO paper_release_item
               (release_id,source_item_id,section,panel,content,final_asset_path,final_asset_sha256,
                generator,plotted_data,source_status,notes)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (release_id,source_item_id,section,panel,content,final_path,final_hash,generator,plotted_data,source_status,notes),
        )
        row = self.db.execute(
            """SELECT release_item_id,content,final_asset_path,final_asset_sha256,generator,plotted_data,source_status,notes
               FROM paper_release_item WHERE release_id=? AND source_item_id=? AND section=? AND panel=?""",
            (release_id,source_item_id,section,panel),
        ).fetchone()
        expected=(content,final_path,final_hash,generator,plotted_data,source_status,notes)
        if not row or tuple(row[1:]) != expected:
            raise RuntimeError(f"Release item conflict: {release_id}/{source_item_id}/{section}/{panel}")
        self.add_count("paper_release_items")
        return int(row[0])

    def add_item_evidence(self, item_id: int, artifact_id: int | None, source_path: str,
                          role: str, availability: str, relationship: str, notes: str) -> None:
        self.db.execute(
            """INSERT OR IGNORE INTO paper_release_item_evidence
               (release_item_id,artifact_id,source_path,evidence_role,availability,relationship,notes)
               VALUES(?,?,?,?,?,?,?)""",
            (item_id,artifact_id,source_path,role,availability,relationship,notes),
        )
        self.add_count("release_item_evidence_links")

    def import_claim(self, *, release_id: str, claim_id: str, source_row_key: str,
                     source_claim_id: str, source_id: int, claim_text: str,
                     source_status: str, manuscript_location: str, source_or_table: str,
                     primary_artifact: str, rebuild_script: str, response_location: str,
                     boundary: str, evidence_refs: list[str], figure_ids: list[str],
                     capture_utc: str, matrix_artifact_id: int) -> None:
        catalog_status, action = conservative_claim_status(source_status)
        notes = (
            f"Source status preserved verbatim as {source_status}. Catalog assessment is "
            "PROVENANCE_PARTIAL unless explicitly excluded/failure/reconstruction; this import does not "
            "re-run the underlying calculation or independently accept the scientific claim. " +
            ("Boundary: " + boundary if boundary else "")
        ).strip()
        self.db.execute(
            """INSERT OR IGNORE INTO paper_claim
               (claim_id,release_id,manuscript_path,section_or_panel,claim_text,scientific_status,publication_action,notes)
               VALUES(?,?,?,?,?,?,?,?)""",
            (claim_id,release_id,manuscript_location or None,source_claim_id,claim_text,catalog_status,action,notes),
        )
        claim = self.db.execute(
            "SELECT release_id,manuscript_path,section_or_panel,claim_text,scientific_status,publication_action,notes "
            "FROM paper_claim WHERE claim_id=?", (claim_id,),
        ).fetchone()
        if not claim or tuple(claim) != (release_id, manuscript_location or None, source_claim_id,
                                        claim_text, catalog_status, action, notes):
            raise RuntimeError(f"Paper claim conflict: {claim_id}")

        self.db.execute(
            """INSERT OR IGNORE INTO paper_release_claim_observation
               (release_id,source_row_key,source_claim_id,question,source_scientific_status,
                primary_artifact,source_or_table,rebuild_script,manuscript_location,
                response_location,boundary,catalog_claim_id,notes)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (release_id,source_row_key,source_claim_id,claim_text,source_status,primary_artifact,
             source_or_table,rebuild_script,manuscript_location,response_location,boundary,claim_id,
             "Verbatim source claim observation from the pinned PR matrix; not a merged manuscript claim."),
        )
        obs = self.db.execute(
            "SELECT catalog_claim_id,source_scientific_status,question FROM paper_release_claim_observation "
            "WHERE release_id=? AND source_row_key=?", (release_id,source_row_key),
        ).fetchone()
        if not obs or tuple(obs) != (claim_id,source_status,claim_text):
            raise RuntimeError(f"Source claim observation conflict: {release_id}/{source_row_key}")
        self.db.execute(
            """INSERT OR IGNORE INTO paper_claim_source_observation
               (source_id,source_claim_key,claim_id,source_status,source_publication_action,
                source_target,source_evidence_refs,observed_utc)
               VALUES(?,?,?,?,?,?,?,?)""",
            (source_id,source_row_key,claim_id,source_status,"",source_or_table,
             json.dumps(evidence_refs,ensure_ascii=False),capture_utc),
        )
        # Claim matrix links are source observations, not independently assessed support.
        exists = self.db.execute(
            "SELECT 1 FROM claim_evidence WHERE claim_id=? AND artifact_id=? AND run_id IS NULL "
            "AND evidence_role='SOURCE_CLAIM_MATRIX' LIMIT 1", (claim_id,matrix_artifact_id),
        ).fetchone()
        if not exists:
            self.db.execute(
                """INSERT INTO claim_evidence
                   (claim_id,artifact_id,run_id,evidence_role,support_status,notes)
                   VALUES(?,?,NULL,'SOURCE_CLAIM_MATRIX','UNKNOWN',?)""",
                (claim_id,matrix_artifact_id,"Imported source matrix row; no scientific re-review performed."),
            )
        for path in evidence_refs:
            artifact_id = self.package_file_artifacts.get(release_id,{}).get(path)
            if artifact_id:
                exists = self.db.execute(
                    "SELECT 1 FROM claim_evidence WHERE claim_id=? AND artifact_id=? AND run_id IS NULL "
                    "AND evidence_role='SOURCE_MATRIX_LISTED_ARTIFACT' LIMIT 1", (claim_id,artifact_id),
                ).fetchone()
                if not exists:
                    self.db.execute(
                        """INSERT INTO claim_evidence
                           (claim_id,artifact_id,run_id,evidence_role,support_status,notes)
                           VALUES(?,?,NULL,'SOURCE_MATRIX_LISTED_ARTIFACT','UNKNOWN',?)""",
                        (claim_id,artifact_id,"Source matrix listed this exact package path; hash identity only, no support adjudication."),
                    )
        for figure_id in figure_ids:
            self.db.execute(
                """INSERT OR IGNORE INTO paper_claim_figure
                   (claim_id,figure_id,source_id,source_claim_key,relationship,notes)
                   VALUES(?,?,?,?, 'ILLUSTRATES',?)""",
                (claim_id,figure_id,source_id,source_row_key,
                 "Explicit source-matrix claim-to-figure reference; no panel-level mapping inferred."),
            )
        self.add_count("paper_claims")
        self.add_count("paper_claim_source_observations")


def parse_semicolon_paths(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(";") if part.strip()]


def verify_pr10_hash_correction(snapshot_dir: Path, p10_manifest: dict,
                                p10_submission: dict, p10_metadata: dict) -> dict:
    """Verify one malformed PR #10 digest against the immutable Git blob.

    The captured source files remain untouched. This narrowly scoped correction
    is allowed only when the pinned commit, exact repository path, malformed
    source declaration, Git blob OID, byte size, and recomputed SHA-256 all match.
    """
    defect = PR10_HASH_DEFECT
    compact = p10_manifest.get("compact_evidence", {})
    manifest_root = str(compact.get("root", "")).rstrip("/")
    declared = compact.get("files", {}).get(defect["relative_name"])
    submission_declared = p10_submission.get("source_selection_sha256")
    commit = p10_metadata.get("headRefOid")
    repository_path = f"{manifest_root}/{defect['relative_name']}"
    if (manifest_root != defect["repository_path"].rsplit("/", 1)[0]
            or repository_path != defect["repository_path"]
            or declared != defect["source_declared_sha256"]
            or submission_declared != defect["source_declared_sha256"]
            or len(str(declared)) != 65
            or commit != defect["commit"]):
        raise RuntimeError("PR #10 hash correction preconditions changed; refusing to normalize a different source declaration")

    git_prefix = ["git", "-C", str(WORKTREE)]
    try:
        blob_oid = subprocess.run(
            git_prefix + ["rev-parse", f"{commit}:{repository_path}"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        byte_size = int(subprocess.run(
            git_prefix + ["cat-file", "-s", f"{commit}:{repository_path}"],
            check=True, capture_output=True, text=True,
        ).stdout.strip())
        blob_bytes = subprocess.run(
            git_prefix + ["show", f"{commit}:{repository_path}"],
            check=True, capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise RuntimeError(f"Could not independently verify pinned PR #10 Git blob: {exc}") from exc
    recomputed = hashlib.sha256(blob_bytes).hexdigest()
    if (blob_oid != defect["expected_blob_oid"]
            or byte_size != defect["expected_byte_size"]
            or len(blob_bytes) != defect["expected_byte_size"]
            or not SHA256_RE.fullmatch(recomputed)
            or recomputed != defect["expected_recomputed_sha256"]):
        raise RuntimeError(
            "Pinned PR #10 Git blob no longer matches the recorded OID, size, and independently recomputed SHA-256"
        )
    return {
        "repository_path": repository_path,
        "commit": commit,
        "source_declared_sha256": declared,
        "submission_record_declared_sha256": submission_declared,
        "source_declaration_length": len(str(declared)),
        "blob_oid": blob_oid,
        "byte_size": byte_size,
        "recomputed_sha256": recomputed,
        "correction": "Use recomputed Git-blob SHA-256 for indexed package evidence; preserve the captured manifest and submission record unchanged.",
    }


def verify_pr13_git_tree_reconciliation(snapshot_dir: Path, p13_manifest: dict,
                                        p13_metadata: dict, p13_files: list[dict[str,str]]) -> tuple[dict,dict]:
    """Check the immutable audit of every PR #13 manifest row against both Git trees."""
    audit_path = snapshot_dir.parent / PR13_TREE_AUDIT_FILENAME
    report_path = snapshot_dir.parent / PR13_TREE_REPORT_FILENAME
    if not audit_path.is_file() or sha256(audit_path) != PR13_TREE_AUDIT_SHA256:
        raise RuntimeError("PR #13 Git-tree reconciliation sidecar is missing or changed")
    if not report_path.is_file() or sha256(report_path) != PR13_TREE_REPORT_SHA256:
        raise RuntimeError("PR #13 Git-tree reconciliation report is missing or changed")
    audit = load_json(audit_path)
    source_manifest = snapshot_dir / "pr13_file_provenance.tsv"
    bundle_commit = p13_manifest.get("bundle_commit")
    head_commit = p13_metadata.get("headRefOid")
    expected_rows = {row["relative_path"].strip(): row for row in p13_files}
    audit_rows = audit.get("rows", [])
    if (audit.get("schema") != "migrationbench_pr13_git_tree_reconciliation_v1"
            or audit.get("repository_root") != "/scratch16/pclancy3/yi/MigrationBench"
            or audit.get("package_prefix") != PR13_PACKAGE_PREFIX
            or audit.get("bundle_commit") != bundle_commit
            or audit.get("pr_head_commit") != head_commit
            or audit.get("source_manifest_sha256") != sha256(source_manifest)
            or audit.get("manifest_row_count") != len(p13_files)
            or audit.get("duplicate_manifest_paths") != 0
            or audit.get("tree_differences_on_manifest_paths") != 0
            or audit.get("bundle_tree_blob_count_under_prefix") != 62
            or audit.get("pr_head_tree_blob_count_under_prefix") != 62):
        raise RuntimeError("PR #13 Git-tree reconciliation metadata differs from the pinned source rows/commits")
    observed = {row.get("relative_path"): row for row in audit_rows}
    if len(audit_rows) != 1384 or len(observed) != 1384 or set(observed) != set(expected_rows):
        raise RuntimeError("PR #13 Git-tree reconciliation rows do not exactly cover the source manifest")

    matching = []
    stale = []
    absent = []
    for relative_path, source_row in expected_rows.items():
        row = observed[relative_path]
        if (row.get("role") != source_row["role"]
                or row.get("manifest_bytes") != int(source_row["bytes"])
                or row.get("manifest_sha256") != source_row["sha256"].strip().lower()):
            raise RuntimeError(f"PR #13 tree audit no longer matches manifest row: {relative_path}")
        tree_records = [row.get("bundle", {}), row.get("pr_head", {})]
        if [record.get("commit") for record in tree_records] != [bundle_commit, head_commit]:
            raise RuntimeError(f"PR #13 tree audit commit mismatch: {relative_path}")
        for record in tree_records:
            if record.get("present"):
                if (not re.fullmatch(r"[0-9a-f]{40}", str(record.get("blob_oid", "")))
                        or not SHA256_RE.fullmatch(str(record.get("sha256", "")))
                        or not isinstance(record.get("bytes"), int)):
                    raise RuntimeError(f"PR #13 Git blob observation is incomplete: {relative_path}")
            elif any(record.get(field) is not None for field in ("blob_oid", "bytes", "sha256")):
                raise RuntimeError(f"PR #13 absent Git path has unexpected blob metadata: {relative_path}")
        bundle = row["bundle"]
        head = row["pr_head"]
        same_blob = all(bundle.get(key) == head.get(key) for key in ("present", "blob_oid", "bytes", "sha256"))
        if row.get("tree_entries_identical") is not same_blob:
            raise RuntimeError(f"PR #13 tree-difference flag is inconsistent: {relative_path}")
        if not same_blob:
            raise RuntimeError(f"PR #13 manifest path changed between pinned trees: {relative_path}")
        if not bundle.get("present"):
            status = "MANIFEST_ROW_NOT_TRACKED_IN_EITHER_PINNED_TREE"
            if row.get("status") != status:
                raise RuntimeError(f"PR #13 missing-path status mismatch: {relative_path}")
            absent.append(row)
        elif (bundle["bytes"] == int(source_row["bytes"])
                and bundle["sha256"] == source_row["sha256"].strip().lower()):
            status = "MANIFEST_MATCHES_BOTH_PINNED_GIT_BLOBS"
            if row.get("status") != status:
                raise RuntimeError(f"PR #13 matched-path status mismatch: {relative_path}")
            matching.append(row)
        else:
            status = "MANIFEST_DECLARATION_MISMATCHES_BOTH_PINNED_GIT_BLOBS"
            if row.get("status") != status:
                raise RuntimeError(f"PR #13 stale-manifest status mismatch: {relative_path}")
            stale.append(row)

    if (len(matching) != 47 or len(stale) != 1 or len(absent) != 1336
            or any(row["role"] != "raw_evidence" for row in absent)
            or stale[0]["relative_path"] != "README.md"
            or stale[0]["manifest_bytes"] != 1477
            or stale[0]["manifest_sha256"] != "fcc5419f83cca28a7354d9ac9ecfb6c755b4c71259194347f23e212a559ad204"
            or stale[0]["bundle"].get("bytes") != 2216
            or stale[0]["bundle"].get("sha256") != "05ad3cc1eacdcb921fb3c3923ed5db68ceca3fa4229af06200b42900a7dbe9d0"
            or stale[0]["bundle"].get("blob_oid") != "4903df26e4ea7d595dea267c009adfe8e4fc6cb4"
            or len(audit.get("unlisted_git_paths", [])) != 14):
        raise RuntimeError("PR #13 tree audit discrepancy counts or the README mismatch changed")
    summary = {
        "audit_path": audit_path.resolve().relative_to(WORKTREE.resolve()).as_posix(),
        "audit_sha256": PR13_TREE_AUDIT_SHA256,
        "report_path": report_path.resolve().relative_to(WORKTREE.resolve()).as_posix(),
        "report_sha256": PR13_TREE_REPORT_SHA256,
        "bundle_commit": bundle_commit,
        "pr_head_commit": head_commit,
        "manifest_rows": len(audit_rows),
        "git_tracked_rows": len(matching) + len(stale),
        "manifest_matches": len(matching),
        "stale_manifest_rows": len(stale),
        "not_tracked_in_either_tree": len(absent),
        "unlisted_git_paths": len(audit["unlisted_git_paths"]),
        "tree_differences_on_manifest_paths": audit["tree_differences_on_manifest_paths"],
    }
    return audit, summary


def verify_pr13_raw_archive(snapshot_dir: Path, p13_files: list[dict[str,str]],
                            p13_metadata: dict) -> tuple[list[dict[str,str]],dict]:
    """Validate the Rockfish raw archive manifest and re-hash every remote payload."""
    archive_capture_dir=snapshot_dir.parent/PR13_ARCHIVE_CAPTURE_DIRNAME
    manifest_path=archive_capture_dir/"ARCHIVE_MANIFEST.tsv"
    report_path=archive_capture_dir/"VERIFICATION_REPORT.json"
    readme_path=archive_capture_dir/"README.md"
    expected_captures=((manifest_path,PR13_ARCHIVE_MANIFEST_SHA256),
                       (report_path,PR13_ARCHIVE_REPORT_SHA256),
                       (readme_path,PR13_ARCHIVE_README_SHA256))
    for path,digest in expected_captures:
        if not path.is_file() or sha256(path)!=digest:
            raise RuntimeError(f"PR #13 archive metadata capture is missing or changed: {path.name}")

    archive_rows=read_tsv(manifest_path)
    archive_report=load_json(report_path)
    source_manifest_path=PR13_RAW_SOURCE_MANIFEST_PATH+".tsv" if not PR13_RAW_SOURCE_MANIFEST_PATH.endswith(".tsv") else PR13_RAW_SOURCE_MANIFEST_PATH
    source_manifest_digest=sha256(snapshot_dir/"pr13_file_provenance.tsv")
    source_commit=p13_metadata.get("headRefOid")
    expected_raw={row["relative_path"].strip():row for row in p13_files if row["role"]=="raw_evidence"}
    archive_raw={row["archive_path"]:row for row in archive_rows if row.get("source_role")=="raw_evidence"}
    bundle_rows=[row for row in archive_rows if row.get("source_role")=="source_bundle"]
    if (len(archive_rows)!=1337 or len(expected_raw)!=1336 or len(archive_raw)!=1336
            or set(archive_raw)!=set(expected_raw) or len(bundle_rows)!=1):
        raise RuntimeError("PR #13 archive manifest must contain exactly 1,336 raw files and one source bundle row")
    if len(archive_rows)!=(len({row["archive_path"] for row in archive_rows})):
        raise RuntimeError("PR #13 archive manifest contains duplicate archive paths")
    for relative_path,source_row in expected_raw.items():
        archived=archive_raw[relative_path]
        expected_fields={
            "source_role":"raw_evidence",
            "source_manifest_path":source_manifest_path,
            "byte_size":str(int(source_row["bytes"])),
            "sha256":source_row["sha256"].strip().lower(),
            "source_commit":source_commit,
            "source_manifest_sha256":source_manifest_digest,
        }
        if any(archived.get(key)!=value for key,value in expected_fields.items()):
            raise RuntimeError(f"PR #13 archive row differs from source manifest: {relative_path}")
    bundle=bundle_rows[0]
    expected_bundle_fields={
        "archive_path":"source_bundle.tar.gz","source_role":"source_bundle",
        "source_manifest_path":PR13_RAW_SOURCE_BUNDLE_PATH,
        "byte_size":str(PR13_RAW_SOURCE_BUNDLE_BYTES),"sha256":PR13_RAW_SOURCE_BUNDLE_SHA256,
        "source_commit":source_commit,"source_manifest_sha256":source_manifest_digest,
    }
    if any(bundle.get(key)!=value for key,value in expected_bundle_fields.items()):
        raise RuntimeError("PR #13 raw archive source-bundle row differs from the pinned Git bundle")
    report_expected={
        "destination_root":PR13_ARCHIVE_ROOT,
        "source_commit":source_commit,
        "source_manifest_path":source_manifest_path,
        "source_manifest_sha256":source_manifest_digest,
        "source_manifest_rows_total":1384,
        "source_manifest_raw_evidence_rows":1336,
        "source_manifest_other_rows_not_extracted":48,
        "extracted_manifest_listed_files":1336,
        "extracted_manifest_listed_bytes":8942992,
        "source_bundle_path":PR13_RAW_SOURCE_BUNDLE_PATH,
        "source_bundle_bytes":PR13_RAW_SOURCE_BUNDLE_BYTES,
        "source_bundle_sha256":PR13_RAW_SOURCE_BUNDLE_SHA256,
        "source_bundle_git_blob_oid":PR13_RAW_SOURCE_BUNDLE_OID,
    }
    if any(archive_report.get(key)!=value for key,value in report_expected.items()):
        raise RuntimeError("PR #13 archive verification report differs from its source manifest or expected bundle")
    security=archive_report.get("security_checks",{})
    if not all(security.get(key) is True for key in ("all_tar_paths_confined_to_source_root",
        "destination_absent_before_creation","files_created_exclusively_without_overwrite",
        "no_path_traversal","no_symlinks_or_hardlinks")):
        raise RuntimeError("PR #13 archive security checks are incomplete or failed")
    excluded=archive_report.get("unlisted_regular_files_excluded",[])
    if excluded!=[{"byte_size":5137,"path":"raw/source_geometry_ensemble/scripts/__pycache__/run_mace_source_geometry_ensemble.cpython-39.pyc"}]:
        raise RuntimeError("PR #13 archive excluded-file record changed")

    expected_payload={"raw_rows":[
        {"archive_path":path,"source_role":"raw_evidence","byte_size":int(row["bytes"]),
         "sha256":row["sha256"].strip().lower()} for path,row in sorted(expected_raw.items())],
        "source_manifest_sha256":source_manifest_digest,
        "metadata_sha256":{"ARCHIVE_MANIFEST.tsv":PR13_ARCHIVE_MANIFEST_SHA256,
                            "VERIFICATION_REPORT.json":PR13_ARCHIVE_REPORT_SHA256,
                            "README.md":PR13_ARCHIVE_README_SHA256},
    }
    remote_source=r'''import csv,hashlib,json,os,stat,subprocess,sys
root='/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_pr13_endpoint47_raw_evidence'
repo='/scratch16/pclancy3/yi/MigrationBench'
gitbin='/data/apps/linux-centos8-cascadelake/gcc-9.3.0/git-2.28.0-uh6lz2zi2m3rwkfujclkz2qrtm6awu6s/bin/git'
commit='2a40d7a171acb32e3352572eff67f06ab8634286'
manifest_rel='analysis/dft_mace_47_site_error_attribution_20261023/tables/FILE_PROVENANCE_MANIFEST.tsv'
bundle_rel='analysis/dft_mace_47_site_error_attribution_20261023/raw_evidence_bundle.tar.gz'
p=json.load(sys.stdin); expected=p['raw_rows']; expected_sha=p['source_manifest_sha256']; errors=[]
def sha_file(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  while True:
   b=f.read(1048576)
   if not b: break
   h.update(b)
 return h.hexdigest()
def git(args):
 q=subprocess.Popen([gitbin,'-C',repo]+args,stdout=subprocess.PIPE,stderr=subprocess.PIPE);out,err=q.communicate()
 if q.returncode: raise RuntimeError(err.decode('utf8','replace'))
 return out
manifest_path=os.path.join(root,'ARCHIVE_MANIFEST.tsv')
with open(manifest_path,newline='') as f: archive_rows=list(csv.DictReader(f,delimiter='\t'))
raw_rows=[r for r in archive_rows if r.get('source_role')=='raw_evidence']; bundle_rows=[r for r in archive_rows if r.get('source_role')=='source_bundle']
actual={r['archive_path']:r for r in raw_rows}; expected_map={r['archive_path']:r for r in expected}
if len(archive_rows)!=1337 or len(raw_rows)!=1336 or len(bundle_rows)!=1 or len(actual)!=1336 or set(actual)!=set(expected_map): errors.append('archive_manifest_row_shape')
for rel,e in expected_map.items():
 r=actual.get(rel)
 if not r: continue
 fields={'source_role':'raw_evidence','source_manifest_path':manifest_rel,'byte_size':str(e['byte_size']),'sha256':e['sha256'],'source_commit':commit,'source_manifest_sha256':expected_sha}
 for k,v in fields.items():
  if r.get(k)!=v: errors.append('manifest_field:'+rel+':'+k)
 path=os.path.join(root,rel)
 try:
  real_root=os.path.realpath(root); real_path=os.path.realpath(path)
  if os.path.commonpath([real_root,real_path])!=real_root: errors.append('path_traversal:'+rel); continue
  st=os.lstat(path)
  if not stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode): errors.append('not_regular:'+rel); continue
  if st.st_size!=e['byte_size'] or sha_file(path)!=e['sha256']: errors.append('content_mismatch:'+rel)
 except Exception as exc: errors.append('read_error:'+rel+':'+str(exc))
bundle=bundle_rows[0] if len(bundle_rows)==1 else {}
tar_path=os.path.join(root,'source_bundle.tar.gz'); tar_sha=sha_file(tar_path); tar_size=os.path.getsize(tar_path)
tar_oid=git(['rev-parse',commit+':'+bundle_rel]).decode().strip(); git_size=int(git(['cat-file','-s',commit+':'+bundle_rel]).decode().strip()); git_sha=hashlib.sha256(git(['show',commit+':'+bundle_rel])).hexdigest()
bundle_fields={'archive_path':'source_bundle.tar.gz','source_role':'source_bundle','source_manifest_path':bundle_rel,'byte_size':str(tar_size),'sha256':tar_sha,'source_commit':commit,'source_manifest_sha256':expected_sha}
if any(bundle.get(k)!=v for k,v in bundle_fields.items()) or tar_oid!='b7f65ae69e7a18afd3a555d259568ffc716d7a6e' or tar_size!=3237382 or git_size!=tar_size or tar_sha!='4930f34a9935d1e368a49c3081a8a1185e3418a2d30761a0a911660d9dfad7be' or tar_sha!=git_sha: errors.append('source_bundle_identity')
source_manifest=git(['show',commit+':'+manifest_rel]); source_sha=hashlib.sha256(source_manifest).hexdigest()
if source_sha!=expected_sha: errors.append('source_manifest_git_sha256')
for name,digest in p['metadata_sha256'].items():
 if sha_file(os.path.join(root,name))!=digest: errors.append('metadata_sha256:'+name)
with open(os.path.join(root,'VERIFICATION_REPORT.json')) as f: report=json.load(f)
if report.get('source_manifest_sha256')!=expected_sha or report.get('source_manifest_raw_evidence_rows')!=1336 or report.get('extracted_manifest_listed_files')!=1336 or report.get('extracted_manifest_listed_bytes')!=sum(r['byte_size'] for r in expected): errors.append('verification_report')
result={'archive_root':root,'archive_manifest_sha256':sha_file(manifest_path),'archive_manifest_data_rows':len(archive_rows),'raw_files_expected':len(expected),'raw_files_verified':len(expected)-sum(1 for e in errors if e.startswith(('content_mismatch:','read_error:','not_regular:','path_traversal:'))),'raw_bytes_verified':sum(r['byte_size'] for r in expected),'content_mismatches':errors,'verification_report_sha256':sha_file(os.path.join(root,'VERIFICATION_REPORT.json')),'archive_readme_sha256':sha_file(os.path.join(root,'README.md')),'source_commit':commit,'source_manifest_path':manifest_rel,'source_manifest_sha256':source_sha,'source_bundle':{'archive_path':'source_bundle.tar.gz','git_path':bundle_rel,'git_blob_oid':tar_oid,'bytes':tar_size,'sha256':tar_sha,'git_blob_sha256':git_sha}}
print(json.dumps(result,sort_keys=True))'''
    encoded=base64.b64encode(remote_source.encode("utf-8")).decode("ascii")
    command=f"python3 -c \"import base64;exec(base64.b64decode('{encoded}'))\""
    try:
        proc=subprocess.run(["ssh","rockfish",command],input=json.dumps(expected_payload,separators=(",",":")).encode("utf-8"),
                            capture_output=True,timeout=300,check=True)
        remote_result=json.loads(proc.stdout.decode("utf-8"))
    except (OSError,subprocess.CalledProcessError,subprocess.TimeoutExpired,json.JSONDecodeError) as exc:
        stderr=getattr(exc,"stderr",b"")
        if isinstance(stderr,bytes): stderr=stderr.decode("utf-8","replace")
        raise RuntimeError(f"Could not verify PR #13 Rockfish raw archive files over SSH: {exc} {stderr}") from exc
    if (remote_result.get("content_mismatches") or remote_result.get("raw_files_verified")!=1336
            or remote_result.get("raw_bytes_verified")!=8942992
            or remote_result.get("archive_manifest_sha256")!=PR13_ARCHIVE_MANIFEST_SHA256
            or remote_result.get("verification_report_sha256")!=PR13_ARCHIVE_REPORT_SHA256
            or remote_result.get("archive_readme_sha256")!=PR13_ARCHIVE_README_SHA256
            or remote_result.get("source_manifest_sha256")!=source_manifest_digest
            or remote_result.get("source_bundle",{}).get("git_blob_oid")!=PR13_RAW_SOURCE_BUNDLE_OID
            or remote_result.get("source_bundle",{}).get("sha256")!=PR13_RAW_SOURCE_BUNDLE_SHA256):
        raise RuntimeError(f"PR #13 Rockfish raw archive live verification failed: {remote_result}")
    summary={"archive_root":PR13_ARCHIVE_ROOT,"archive_relative_root":PR13_ARCHIVE_RELATIVE_ROOT,
        "archive_manifest_sha256":PR13_ARCHIVE_MANIFEST_SHA256,"archive_manifest_data_rows":len(archive_rows),
        "raw_files_verified":1336,"raw_bytes_verified":8942992,"source_bundle_sha256":PR13_RAW_SOURCE_BUNDLE_SHA256,
        "source_bundle_git_blob_oid":PR13_RAW_SOURCE_BUNDLE_OID,"source_commit":source_commit,
        "source_manifest_sha256":source_manifest_digest,"verification_report_sha256":PR13_ARCHIVE_REPORT_SHA256,
        "archive_readme_sha256":PR13_ARCHIVE_README_SHA256,"live_content_mismatches":0,
        "archive_report_created_utc":archive_report["created_utc"],
        "live_verification_utc":datetime.now(timezone.utc).isoformat()}
    return archive_rows,summary


def validate_inputs(snapshot_dir: Path) -> dict:
    snapshot_dir=snapshot_dir.resolve(strict=True)
    capture_rows=read_tsv(snapshot_dir/"CAPTURE_MANIFEST.tsv")
    if len(capture_rows)!=19:
        raise RuntimeError(f"Expected 19 immutable source captures, got {len(capture_rows)}")
    for row in capture_rows:
        path=snapshot_dir/row["capture_file"]
        if not path.is_file() or sha256(path)!=row["sha256"] or path.stat().st_size!=int(row["byte_size"]):
            raise RuntimeError(f"Source capture hash/size mismatch: {row['capture_file']}")
        meta=load_json(snapshot_dir/f"pr{row['pr_number']}_metadata.json")
        if meta["headRefOid"]!=row["head_sha"] or meta["state"]!="OPEN":
            raise RuntimeError(f"PR #{row['pr_number']} metadata differs from capture manifest")
    p10_manifest=load_json(snapshot_dir/"pr10_manifest.json")
    p10_claims=read_tsv(snapshot_dir/"pr10_claim_matrix.tsv")
    p10_fig=load_json(snapshot_dir/"pr10_figure_manifest.json")
    if p10_manifest.get("status")!="MLIP_RELEASE_VERIFIED_DFT_PENDING" or len(p10_claims)!=13:
        raise RuntimeError("PR #10 status or 13-row claim matrix does not match reviewed source")
    if p10_manifest["paper_integration"].get("figure_manifest_sha256")!=sha256(snapshot_dir/"pr10_figure_manifest.json"):
        raise RuntimeError("PR #10 figure manifest hash mismatch")
    compact=p10_manifest["compact_evidence"]["files"]
    for filename,digest in (("pr10_claim_matrix.tsv",compact["CLAIM_EVIDENCE_MATRIX.tsv"]),
                            ("pr10_submission_record.json",compact["SUBMISSION_RECORD.json"])):
        if digest!=sha256(snapshot_dir/filename):
            raise RuntimeError(f"PR #10 manifest hash mismatch for {filename}")
    p10_submission=load_json(snapshot_dir/"pr10_submission_record.json")
    p10_metadata=load_json(snapshot_dir/"pr10_metadata.json")
    p10_hash_correction=verify_pr10_hash_correction(snapshot_dir,p10_manifest,p10_submission,p10_metadata)
    p12_candidates=read_tsv(snapshot_dir/"pr12_candidates.tsv")
    p12_selection=read_tsv(snapshot_dir/"pr12_selection.tsv")
    if len(p12_candidates)!=3 or len(p12_selection)!=3:
        raise RuntimeError("PR #12 source must contain three candidates and three selection rows")
    if any(r["selected"].lower()!="no" for r in p12_selection+p12_candidates):
        raise RuntimeError("PR #12 has a selection other than no; stop for review")
    if any(r["status"]!="PROVENANCE_PARTIAL" for r in p12_candidates):
        raise RuntimeError("PR #12 candidate source status changed")
    for filename in ("pr12_fig057_manifest.json","pr12_fig058_manifest.json","pr12_fig059_manifest.json"):
        manifest=load_json(snapshot_dir/filename)
        if manifest.get("status")!="PROVENANCE_PARTIAL":
            raise RuntimeError(f"PR #12 {filename} source status changed")
    p13_manifest=load_json(snapshot_dir/"pr13_manifest.json")
    p13_files=read_tsv(snapshot_dir/"pr13_file_provenance.tsv")
    p13_claims=read_tsv(snapshot_dir/"pr13_claim_matrix.tsv")
    if p13_manifest.get("release_status")!="READY_FOR_REVIEW_NOT_YET_MERGED" or len(p13_claims)!=6:
        raise RuntimeError("PR #13 source status or C1-C6 claim matrix does not match reviewed source")
    if len(p13_files)!=1384 or len({r["relative_path"] for r in p13_files})!=1384:
        raise RuntimeError("PR #13 file manifest must contain 1,384 unique file rows")
    for rel,digest in p13_manifest["key_artifact_hashes"].items():
        matches=[r for r in p13_files if r["relative_path"]==rel]
        if len(matches)!=1 or matches[0]["sha256"]!=digest:
            raise RuntimeError(f"PR #13 key artifact hash mismatch: {rel}")
    sensitive=[r["relative_path"] for r in p13_files
               if any(x in r["relative_path"].lower() for x in ("queue_snapshot","squeue_capture","checkpoint.pth","model.pt","model.pth"))]
    if sensitive:
        raise RuntimeError(f"PR #13 file index contains prohibited payload/snapshot paths: {sensitive[:5]}")
    p13_metadata=load_json(snapshot_dir/"pr13_metadata.json")
    _,p13_git_tree_summary=verify_pr13_git_tree_reconciliation(snapshot_dir,p13_manifest,p13_metadata,p13_files)
    _,p13_archive_summary=verify_pr13_raw_archive(snapshot_dir,p13_files,p13_metadata)
    return {"capture_count":len(capture_rows),"pr10_claims":len(p10_claims),
            "pr10_compact_files":len(compact),"pr10_figure_outputs":len(p10_fig["outputs"]),
            "pr12_candidates":len(p12_candidates),"pr12_all_unselected":True,
            "pr13_claims":len(p13_claims),"pr13_package_files":len(p13_files),
            "pr10_hash_correction":p10_hash_correction,
            "pr13_git_tree_reconciliation":p13_git_tree_summary,
            "pr13_raw_archive":p13_archive_summary,
            "capture_manifest_sha256":sha256(snapshot_dir/"CAPTURE_MANIFEST.tsv")}


def run_import(db_path: Path, snapshot_dir: Path) -> dict:
    validation = validate_inputs(snapshot_dir)
    capture_manifest = snapshot_dir / "CAPTURE_MANIFEST.tsv"
    capture_rows = read_tsv(capture_manifest)
    if len(capture_rows) != 19:
        raise RuntimeError(f"Expected 19 immutable source captures, got {len(capture_rows)}")
    capture_manifest_digest = sha256(capture_manifest)
    capture_utc_values = {row["capture_utc"] for row in capture_rows}
    if len(capture_utc_values) != 1:
        raise RuntimeError("Capture manifest rows must share one capture timestamp")
    capture_utc = capture_utc_values.pop()
    importer = Importer(sqlite3.connect(db_path, timeout=180), snapshot_dir, capture_rows)
    db = importer.db
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=180000")

    required_tables = {
        "schema_migration","registry_source","project","storage_root","storage_path","artifact",
        "artifact_location","paper_release","paper_release_item","paper_release_item_evidence",
        "paper_release_claim_observation","paper_claim","paper_claim_source_observation","claim_evidence",
        "paper_figure","paper_figure_asset","paper_claim_figure","registry_entity_ref",
        "git_repository","git_ref_observation",
    }
    found_tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = required_tables - found_tables
    if missing:
        raise RuntimeError(f"Catalog schema is missing tables: {sorted(missing)}")
    migrations = {int(row[0]) for row in db.execute("SELECT version FROM schema_migration")}
    if not {9,10,12,13}.issubset(migrations):
        raise RuntimeError("Catalog must have paper-figure and Revision 1 migrations 009, 010, 012, and 013 applied")

    db.execute("BEGIN IMMEDIATE")
    try:
        project = db.execute("SELECT project_name FROM project WHERE project_id='MIGRATIONBENCH'").fetchone()
        if not project:
            raise RuntimeError("MIGRATIONBENCH project row is missing")
        archive_root = db.execute("SELECT scoped_path FROM storage_root WHERE root_id=?",(ROCKFISH_ROOT_ID,)).fetchone()
        if not archive_root or Path(archive_root[0]).resolve()!=Path(PR13_ARCHIVE_ROOT_STORAGE_PATH).resolve():
            raise RuntimeError("RF-SCRATCH16-YI storage root does not resolve to the verified raw archive parent")
        if not db.execute("SELECT 1 FROM paper_release WHERE release_id='MB-REVISION1-SUBMITTED-20261110'").fetchone():
            raise RuntimeError("Submitted Revision 1 release is missing; refusing to import a parallel candidate")

        for row in capture_rows:
            filename = row["capture_file"]
            source = importer.add_source(filename, "GITHUB_PR_SOURCE_CAPTURE", "Pinned source snapshot; used only as evidence input for a separate review candidate.")
            importer.add_entity_ref(source, "github_pull_request", row["pr_number"], "EXTERNAL_POINTER",
                                    f"https://github.com/{REPOSITORY}/pull/{row['pr_number']}",
                                    "OBSERVED_PR_SOURCE", f"head={row['head_sha']}; source={row['source_url']}")

        # PR #10 — compact 639-candidate MLIP evidence, with DFT pending.
        p10_meta = load_json(snapshot_dir / "pr10_metadata.json")
        p10_manifest = load_json(snapshot_dir / "pr10_manifest.json")
        p10_matrix = read_tsv(snapshot_dir / "pr10_claim_matrix.tsv")
        p10_figure_manifest = load_json(snapshot_dir / "pr10_figure_manifest.json")
        if p10_meta["state"] != "OPEN" or p10_meta["headRefOid"] != importer.capture("pr10_manifest.json")["head_sha"]:
            raise RuntimeError("PR #10 metadata does not match the pinned package manifest capture")
        if p10_manifest.get("status") != "MLIP_RELEASE_VERIFIED_DFT_PENDING" or len(p10_matrix) != 13:
            raise RuntimeError("PR #10 manifest/claim row count does not match the audited package")
        p10_submit = load_json(snapshot_dir / "pr10_submission_record.json")
        if p10_submit.get("status") != "SUBMITTED_PENDING" or p10_submit.get("dft_ready_mode") != "local_periodic_shell":
            raise RuntimeError("PR #10 DFT submission record no longer matches the reviewed pending reconstruction state")
        p10_id = f"MB-PR10-639-REVIEW-{p10_meta['headRefOid'][:8]}"
        p10_manifest_source = importer.sources["pr10_manifest.json"]
        p10_claim_source = importer.sources["pr10_claim_matrix.tsv"]
        p10_figure_source = importer.sources["pr10_figure_manifest.json"]
        p10_manifest_art = importer.capture_artifacts["pr10_manifest.json"]
        p10_claim_art = importer.capture_artifacts["pr10_claim_matrix.tsv"]
        importer.add_release(p10_id,10,p10_meta,p10_manifest_art,p10_manifest,
                             f"Manifest source_commit={p10_manifest['source_commit']}; source release status=MLIP_RELEASE_VERIFIED_DFT_PENDING. DFT job {p10_submit['slurm_job']} is SUBMITTED_PENDING for a {p10_submit['dft_ready_natoms']}-atom local-periodic-shell RECONSTRUCTION. No causal mechanism, DFT correctness, or transport claim is promoted.")
        importer.add_entity_ref(p10_manifest_source,"github_pull_request","10","PAPER_RELEASE",p10_id,"OPEN_PR_REVIEW_CANDIDATE",p10_manifest["status"])
        importer.add_git_observation(10,p10_meta,capture_utc)
        importer.add_package_index_item(
            p10_id,
            "PR #10 package file index: compact manifest, claim matrix, submission record, figure manifest, and hashed compact evidence. Heavy model checkpoints and 2050-atom payloads remain external.",
            capture_utc,
        )
        p10_index_item = importer.db.execute(
            "SELECT release_item_id FROM paper_release_item WHERE release_id=? AND source_item_id='PACKAGE_FILE_INDEX' AND section='package evidence' AND panel='all'",(p10_id,)
        ).fetchone()[0]

        p10_files: dict[str, tuple[str|None,int|None,str,str]] = {}
        compact_root = p10_manifest["compact_evidence"]["root"].rstrip("/")
        hash_correction = validation["pr10_hash_correction"]
        for relative, digest in p10_manifest["compact_evidence"]["files"].items():
            note = "SHA-256 declared in the PR #10 release manifest."
            if relative == PR10_HASH_DEFECT["relative_name"]:
                if digest != hash_correction["source_declared_sha256"]:
                    raise RuntimeError("PR #10 source hash declaration changed after validation")
                digest = hash_correction["recomputed_sha256"]
                note = (
                    "Source manifest and SUBMISSION_RECORD both contain the same malformed 65-character SHA-256 declaration "
                    f"{hash_correction['source_declared_sha256']}; for this exact path only, the indexed digest is corrected "
                    f"to {digest}, independently recomputed from pinned commit {hash_correction['commit']} Git blob "
                    f"{hash_correction['blob_oid']} ({hash_correction['byte_size']} bytes). Captured source declarations remain unchanged."
                )
            p10_files[f"{compact_root}/{relative}"] = (digest,None,role_for_path(relative),note)
        p10_capture_names = {
            "pr10_manifest.json": "registry/from_scratch_mlip_candidate_selection_20261013.json",
            "pr10_claim_matrix.tsv": "data/raw_compact/revision1/from_scratch_failure_attribution_20261013/CLAIM_EVIDENCE_MATRIX.tsv",
            "pr10_readme.md": "data/raw_compact/revision1/from_scratch_failure_attribution_20261013/README.md",
            "pr10_submission_record.json": "data/raw_compact/revision1/from_scratch_failure_attribution_20261013/SUBMISSION_RECORD.json",
            "pr10_figure_manifest.json": p10_manifest["paper_integration"]["figure_manifest"],
        }
        for filename, repository_path in p10_capture_names.items():
            cap = importer.capture(filename)
            prior = p10_files.get(repository_path)
            if prior and prior[0] != cap["capture_sha256"]:
                raise RuntimeError(f"PR #10 source manifest hash disagrees with captured file: {repository_path}")
            p10_files[repository_path] = (cap["capture_sha256"],cap["capture_size"],role_for_path(repository_path),"Byte hash verified from the pinned PR source capture.")
        for relative, digest in p10_figure_manifest["outputs"].items():
            path = "paper/revision1/figures/" + relative
            p10_files[path] = (digest,None,"publication_figure","SHA-256 declared by the figure reproduction manifest.")
        renderer_path = p10_manifest["paper_integration"]["figure_script"]
        p10_files[renderer_path] = (p10_manifest["paper_integration"]["figure_script_sha256"],None,"reproduction_script","SHA-256 declared by the figure reproduction manifest.")
        for repository_path,(digest,size,role,note) in sorted(p10_files.items()):
            artifact_id = importer.add_package_file(pr_number=10,release_id=p10_id,item_id=p10_index_item,
                source_id=p10_manifest_source,commit=p10_meta["headRefOid"],repository_path=repository_path,
                digest=digest,size=size,declared_role=role,note=note)
        importer.add_source_capture_evidence(p10_id,p10_index_item,"pr10_manifest.json",p10_manifest_source,capture_utc)
        importer.add_source_capture_evidence(p10_id,p10_index_item,"pr10_claim_matrix.tsv",p10_claim_source,capture_utc)
        p10_figure_id="MB-PR10-FIGS-CANDIDATE"
        p10_figure_assets=[]
        for relative,digest in p10_figure_manifest["outputs"].items():
            path="paper/revision1/figures/"+relative
            aid=importer.package_file_artifacts[p10_id].get(path)
            p10_figure_assets.append(("FIGURE_EXPORT",github_blob_url(p10_meta["headRefOid"],path),aid,"Source figure manifest hash; static MLIP diagnostic composite."))
        figure_manifest_url=github_blob_url(p10_meta["headRefOid"],p10_manifest["paper_integration"]["figure_manifest"])
        p10_figure_assets.append(("PROVENANCE_MANIFEST",figure_manifest_url,p10_figure_source and importer.capture_artifacts["pr10_figure_manifest.json"],"Source figure manifest records 639 candidates and 1,917 finite static MLIP rows."))
        p10_figure_assets.append(("REBUILD_METHOD",github_blob_url(p10_meta["headRefOid"],renderer_path),None,"Renderer SHA declared by manifest; input/model lineage is still subject to its scientific boundaries."))
        importer.add_figure(figure_id=p10_figure_id,release_id=p10_id,source_id=p10_figure_source,
            title="From-scratch candidate static force diagnostics",chart_kind="COMPOSITE_DIAGNOSTIC",
            chart_status="PR_REVIEW_CANDIDATE",scientific_boundary=p10_figure_manifest["interpretation_boundary"],assets=p10_figure_assets)
        p10_item=importer.add_release_item(p10_id,"FIGS-FROM-SCRATCH","Supplementary Information","all",
            "Three-model static force diagnostic over 639 candidates.",
            github_blob_url(p10_meta["headRefOid"],"paper/revision1/figures/figS_from_scratch_failure_attribution.pdf"),
            p10_figure_manifest["outputs"].get("figS_from_scratch_failure_attribution.pdf"),
            github_blob_url(p10_meta["headRefOid"],renderer_path),
            github_blob_url(p10_meta["headRefOid"],"data/raw_compact/revision1/from_scratch_failure_attribution_20261013/mlip_results/mlip_candidate_results_all.csv"),
            "CANDIDATE_NOT_SUBMITTED",
            "PR #10 remains open. Figure integration status is source-reported VERIFIED, but the import does not re-run the validator or validate DFT correctness.")
        importer.add_item_evidence(p10_item, importer.package_file_artifacts[p10_id].get("paper/revision1/figures/figS_from_scratch_failure_attribution.pdf"),
            github_blob_url(p10_meta["headRefOid"],"paper/revision1/figures/figS_from_scratch_failure_attribution.pdf"),"FINAL_ASSET","EXTERNAL_POINTER","PINNED_GITHUB_COMMIT_PATH","Source figure hash is recorded; not a member of the submitted release catalog row.")
        for r in p10_matrix:
            ev=parse_semicolon_paths(r.get("evidence",""))
            source_row=f"PR10:{r['claim_id']}"
            figure_ids=[p10_figure_id] if r["claim_id"]=="C013" else []
            importer.import_claim(release_id=p10_id,claim_id=f"MB-PR10-{r['claim_id']}",source_row_key=source_row,
                source_claim_id=r["claim_id"],source_id=p10_claim_source,claim_text=r["claim"],
                source_status=r["status"],manuscript_location="paper/revision1/sn-article-SI.tex" if r["claim_id"]=="C013" else "",
                source_or_table=r["evidence"],primary_artifact=r["evidence"],rebuild_script=renderer_path if r["claim_id"]=="C013" else "",
                response_location="",boundary=r["boundary_or_gap"],evidence_refs=ev,figure_ids=figure_ids,
                capture_utc=capture_utc,matrix_artifact_id=p10_claim_art)

        # PR #12 — preserve candidates and explicit no-selection decision; no blank claims.
        p12_meta=load_json(snapshot_dir/"pr12_metadata.json")
        p12_candidates=read_tsv(snapshot_dir/"pr12_candidates.tsv")
        p12_selection=read_tsv(snapshot_dir/"pr12_selection.tsv")
        if p12_meta["state"]!="OPEN" or len(p12_candidates)!=3 or len(p12_selection)!=3:
            raise RuntimeError("PR #12 metadata/candidate selection rows do not match the observed source")
        if any(row["selected"].strip().lower()!="no" for row in p12_selection):
            raise RuntimeError("PR #12 selection changed; refusing to infer release selection")
        if any(row["selected"].strip().lower()!="no" or row["status"]!="PROVENANCE_PARTIAL" for row in p12_candidates):
            raise RuntimeError("PR #12 candidate status changed; review the new state before import")
        p12_id=f"MB-PR12-CRNEB-REVIEW-{p12_meta['headRefOid'][:8]}"
        p12_candidates_source=importer.sources["pr12_candidates.tsv"]
        p12_selection_source=importer.sources["pr12_selection.tsv"]
        p12_selection_art=importer.capture_artifacts["pr12_selection.tsv"]
        p12_manifest={"release_status":"NO_FIGURE_SELECTED","source_selection":"paper_evidence/release_selection.tsv"}
        importer.add_release(p12_id,12,p12_meta,p12_selection_art,p12_manifest,
            "Source selection file marks FIG-057/058/059 no; no Overleaf-ready release was produced. All three remain PROVENANCE_PARTIAL review candidates.")
        importer.add_entity_ref(p12_candidates_source,"github_pull_request","12","PAPER_RELEASE",p12_id,"OPEN_PR_REVIEW_CANDIDATE","All three figure candidates are unselected.")
        importer.add_git_observation(12,p12_meta,capture_utc)
        p12_index=importer.add_package_index_item(p12_id,"PR #12 candidate package file index: selection, registry, manifests, hashed inputs/outputs, and linked source scripts/tables; no figure selected for release.",capture_utc)
        p12_files: dict[str,tuple[str|None,int|None,str,str]]={}
        cap_paths={
            "pr12_readme.md":"paper_evidence/README.md",
            "pr12_selection.tsv":"paper_evidence/release_selection.tsv",
            "pr12_candidates.tsv":"paper_evidence/candidates/FIGURE_CANDIDATES.tsv",
            "pr12_fig057_manifest.json":p12_candidates[0]["provenance_source"],
            "pr12_fig058_manifest.json":p12_candidates[1]["provenance_source"],
            "pr12_fig059_manifest.json":p12_candidates[2]["provenance_source"],
        }
        for filename,path in cap_paths.items():
            cap=importer.capture(filename)
            p12_files[path]=(cap["capture_sha256"],cap["capture_size"],role_for_path(path),"Byte hash verified from pinned PR #12 capture.")
        manifest_by_candidate={
            "FIG-057":(load_json(snapshot_dir/"pr12_fig057_manifest.json"),"pr12_fig057_manifest.json"),
            "FIG-058":(load_json(snapshot_dir/"pr12_fig058_manifest.json"),"pr12_fig058_manifest.json"),
            "FIG-059":(load_json(snapshot_dir/"pr12_fig059_manifest.json"),"pr12_fig059_manifest.json"),
        }
        figure_ids={}
        for candidate in p12_candidates:
            cid=candidate["candidate_id"]
            manifest,manifest_capture=manifest_by_candidate[cid]
            figure_id=f"MB-PR12-{cid}-{candidate['version']}"
            figure_ids[cid]=figure_id
            figure_dir=str(PurePosixPath(candidate["figure_source"]).parent)
            input_prefix=(
                "analysis/task3_endpoint_only_neb/exploratory_energy_audit/rockfish_cr_neb_20261013"
                if cid=="FIG-057" else ""
            )
            for relative,digest in manifest.get("inputs",{}).items():
                if not SHA256_RE.fullmatch(digest):
                    continue
                if input_prefix:
                    path=f"{input_prefix}/{relative}"
                else:
                    # PR #12 source manifests sometimes name an external old
                    # workspace directory instead of a Git repository file.
                    continue
                p12_files[path]=(digest,None,role_for_path(path,"SOURCE_INPUT"),"SHA-256 declared by the pinned candidate manifest.")
            for relative,digest in manifest.get("outputs",{}).items():
                path=f"{figure_dir}/{relative}"
                p12_files[path]=(digest,None,role_for_path(path,"DERIVED_OUTPUT"),"SHA-256 declared by the pinned candidate manifest.")
            for field,asset_role in (("figure_source","FIGURE_EXPORT"),("data_source","DERIVED_DATA"),
                                     ("text_source","SOURCE_ASSET"),("tex_source","SOURCE_ASSET"),
                                     ("provenance_source","PROVENANCE_MANIFEST"),("rebuild_script","REBUILD_METHOD")):
                path=(candidate.get(field) or "").strip()
                if path and path not in p12_files:
                    p12_files[path]=(None,None,asset_role,"Pinned PR commit identifies this file; no per-file SHA-256 is declared in the figure manifest.")
            selection_note=next(row["manual_notes"] for row in p12_selection if row["candidate_id"]==cid)
            source_status=candidate["status"]
            boundary=f"{candidate['scientific_scope']}; {candidate['notes']} Manual selection note: {selection_note}"
            assets=[]
            role_paths=(("figure_source","FIGURE_EXPORT"),("data_source","DERIVED_DATA"),
                        ("text_source","SOURCE_ASSET"),("tex_source","SOURCE_ASSET"),
                        ("provenance_source","PROVENANCE_MANIFEST"),("rebuild_script","REBUILD_METHOD"))
            for field,asset_role in role_paths:
                path=(candidate.get(field) or "").strip()
                if not path:
                    continue
                digest=p12_files.get(path,(None,None,"", ""))[0]
                artifact_id=importer.add_artifact(digest,p12_files[path][1],"MIGRATIONBENCH_PR12_FILE",
                    role_for_path(path,asset_role),f"PR #12 candidate {cid}; " + p12_files[path][3]) if digest else None
                assets.append((asset_role,github_blob_url(p12_meta["headRefOid"],path),artifact_id,
                               f"Selected=no; source status={source_status}; {boundary}"))
            importer.add_figure(figure_id=figure_id,release_id=p12_id,source_id=p12_candidates_source,
                title=candidate["title"],chart_kind="FIGURE_CANDIDATE",chart_status="UNSELECTED_CANDIDATE",
                scientific_boundary=boundary,assets=assets)
        for path,(digest,size,role,note) in sorted(p12_files.items()):
            importer.add_package_file(pr_number=12,release_id=p12_id,item_id=p12_index,
                source_id=p12_candidates_source,commit=p12_meta["headRefOid"],repository_path=path,
                digest=digest,size=size,declared_role=role,note=note)
        for filename in ("pr12_candidates.tsv","pr12_selection.tsv"):
            importer.add_source_capture_evidence(p12_id,p12_index,filename,importer.sources[filename],capture_utc)

        # PR #13 — endpoint-47 handoff with source C1–C6 claims and full file hash index.
        p13_meta=load_json(snapshot_dir/"pr13_metadata.json")
        p13_manifest=load_json(snapshot_dir/"pr13_manifest.json")
        p13_file_rows=read_tsv(snapshot_dir/"pr13_file_provenance.tsv")
        p13_claims=read_tsv(snapshot_dir/"pr13_claim_matrix.tsv")
        if p13_meta["state"]!="OPEN" or p13_manifest.get("release_status")!="READY_FOR_REVIEW_NOT_YET_MERGED" or len(p13_claims)!=6:
            raise RuntimeError("PR #13 source release state or C1–C6 row count changed")
        p13_id=f"MB-PR13-ENDPOINT47-REVIEW-{p13_meta['headRefOid'][:8]}"
        p13_manifest_source=importer.sources["pr13_manifest.json"]
        p13_file_source=importer.sources["pr13_file_provenance.tsv"]
        p13_claim_source=importer.sources["pr13_claim_matrix.tsv"]
        p13_manifest_art=importer.capture_artifacts["pr13_manifest.json"]
        p13_claim_art=importer.capture_artifacts["pr13_claim_matrix.tsv"]
        importer.add_release(p13_id,13,p13_meta,p13_manifest_art,p13_manifest,
            "Source status=READY_FOR_REVIEW_NOT_YET_MERGED; Overleaf version=PENDING_USER_HANDOFF. No absolute MACE–DFT error, barrier, transport coefficient, or stability category is eligible from this package.")
        importer.add_entity_ref(p13_manifest_source,"github_pull_request","13","PAPER_RELEASE",p13_id,"OPEN_PR_REVIEW_CANDIDATE",
            f"Manifest bundle_commit={p13_manifest['bundle_commit']}; branch head={p13_meta['headRefOid']}; Overleaf handoff remains pending.")
        importer.add_git_observation(13,p13_meta,capture_utc)
        p13_index=importer.add_package_index_item(p13_id,
            "PR #13 FILE_PROVENANCE rows are retained as declarations. Only Git blobs independently present at both pinned commits are indexed as file artifacts; 1,336 raw_evidence paths are absent and README.md has a stale manifest row.",capture_utc)
        p13_prefix=PR13_PACKAGE_PREFIX
        p13_manifest_row_map: dict[str,tuple[str,int,str]]={}
        for row in p13_file_rows:
            path=(row.get("relative_path") or "").strip()
            digest=(row.get("sha256") or "").strip().lower()
            if not path or not SHA256_RE.fullmatch(digest):
                raise RuntimeError(f"Malformed PR #13 FILE_PROVENANCE_MANIFEST row: {row}")
            size=int(row["bytes"])
            repository_path=f"{p13_prefix}/{path}"
            if repository_path in p13_manifest_row_map and p13_manifest_row_map[repository_path]!=(digest,size,row["role"]):
                raise RuntimeError(f"Conflicting PR #13 file manifest rows for {repository_path}")
            p13_manifest_row_map[repository_path]=(digest,size,row["role"])
        if len(p13_file_rows)!=1384 or len(p13_manifest_row_map)!=1384:
            raise RuntimeError("PR #13 file provenance manifest must contain 1,384 unique paths")
        for rel,digest in p13_manifest["key_artifact_hashes"].items():
            path=f"{p13_prefix}/{rel}"
            if p13_manifest_row_map.get(path,(None,None,None))[0]!=digest:
                raise RuntimeError(f"PR #13 release manifest key hash disagrees with file manifest: {rel}")

        p13_tree_audit=load_json(snapshot_dir.parent/PR13_TREE_AUDIT_FILENAME)
        p13_tree_rows={row["relative_path"]:row for row in p13_tree_audit["rows"]}
        p13_file_map: dict[str,tuple[str,int,str]]={}
        p13_archive_capture_dir=snapshot_dir.parent/PR13_ARCHIVE_CAPTURE_DIRNAME
        p13_archive_manifest_path=p13_archive_capture_dir/"ARCHIVE_MANIFEST.tsv"
        p13_archive_report=load_json(p13_archive_capture_dir/"VERIFICATION_REPORT.json")
        p13_archive_rows=read_tsv(p13_archive_manifest_path)
        p13_archive_raw_by_path={row["archive_path"]:row for row in p13_archive_rows if row["source_role"]=="raw_evidence"}
        p13_archive_verified_utc=validation["pr13_raw_archive"]["live_verification_utc"]
        p13_audit_path=snapshot_dir.parent/PR13_TREE_AUDIT_FILENAME
        p13_audit_digest=validation["pr13_git_tree_reconciliation"]["audit_sha256"]
        p13_audit_rel=p13_audit_path.resolve().relative_to(WORKTREE.resolve()).as_posix()
        p13_audit_source_key=f"migrationbench_pr13_git_tree_audit_20261124:{p13_audit_digest[:20]}"
        p13_audit_source_path=p13_audit_path.resolve().as_uri()
        db.execute(
            """INSERT OR IGNORE INTO registry_source
               (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
                immutable_snapshot,authority_note,notes)
               VALUES(?,?,'PR13_GIT_TREE_RECONCILIATION',?,?,?,1,?,?)""",
            (p13_audit_source_key,p13_audit_source_path,p13_audit_digest,capture_utc,capture_utc,
             "Read-only Git tree reconciliation for the pinned PR #13 bundle and PR-head commits.",
             f"bundle={p13_manifest['bundle_commit']}; PR head={p13_meta['headRefOid']}; source manifest SHA-256={sha256(snapshot_dir/'pr13_file_provenance.tsv')}."),
        )
        p13_audit_source=db.execute(
            "SELECT source_id,source_path,sha256 FROM registry_source WHERE source_key=?",(p13_audit_source_key,)
        ).fetchone()
        if not p13_audit_source or tuple(p13_audit_source[1:])!=(p13_audit_source_path,p13_audit_digest):
            raise RuntimeError("PR #13 Git-tree reconciliation source identity conflict")
        p13_audit_art=importer.add_artifact(
            p13_audit_digest,p13_audit_path.stat().st_size,"PR13_GIT_TREE_RECONCILIATION","SOURCE_TREE_AUDIT",
            "Machine-readable, read-only comparison of all PR #13 file-manifest rows with the bundle and PR-head Git trees.")
        importer.add_artifact_location(p13_audit_art,LOCAL_ROOT_ID,p13_audit_rel,capture_utc)
        importer.add_item_evidence(p13_index,p13_audit_art,p13_audit_rel,"GIT_TREE_RECONCILIATION","PRESENT",
            "READ_ONLY_PINNED_COMMIT_COMPARISON",
            f"sha256={p13_audit_digest}; {validation['pr13_git_tree_reconciliation']['git_tracked_rows']} manifest paths found in both trees; "
            f"{validation['pr13_git_tree_reconciliation']['not_tracked_in_either_tree']} raw_evidence paths absent; README manifest row stale.")

        def register_archive_control(filename: str, semantic_role: str) -> tuple[int,int,str]:
            local_path=p13_archive_capture_dir/filename
            digest=sha256(local_path)
            size=local_path.stat().st_size
            remote_path=f"{PR13_ARCHIVE_ROOT}/{filename}"
            local_rel=local_path.resolve().relative_to(WORKTREE.resolve()).as_posix()
            source_key=f"migrationbench_pr13_archive_control_{filename.replace('.', '_')}_{digest[:16]}"
            archive_as_of=p13_archive_report["created_utc"]
            note=(f"Rockfish PR #13 raw archive control file; source manifest SHA-256="
                  f"{p13_archive_report['source_manifest_sha256']}; archive root={PR13_ARCHIVE_ROOT}.")
            db.execute(
                """INSERT OR IGNORE INTO registry_source
                   (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
                    immutable_snapshot,authority_note,notes)
                   VALUES(?,?,?,?,?,?,1,?,?)""",
                (source_key,remote_path,"PR13_ROCKFISH_RAW_ARCHIVE_CONTROL",digest,p13_archive_verified_utc,
                 archive_as_of,"Rockfish archive control file, copied byte-identically to the local audit tree.",note),
            )
            source=db.execute("SELECT source_id,source_path,sha256 FROM registry_source WHERE source_key=?",(source_key,)).fetchone()
            if not source or tuple(source[1:])!=(remote_path,digest):
                raise RuntimeError(f"PR #13 archive control source identity conflict: {filename}")
            artifact_id=importer.add_artifact(digest,size,"PR13_ROCKFISH_ARCHIVE_CONTROL",semantic_role,note)
            importer.add_artifact_location(artifact_id,ROCKFISH_ROOT_ID,
                f"{PR13_ARCHIVE_RELATIVE_ROOT}/{filename}",p13_archive_verified_utc)
            importer.add_artifact_location(artifact_id,LOCAL_ROOT_ID,local_rel,p13_archive_verified_utc)
            importer.add_item_evidence(p13_index,artifact_id,remote_path,"ROCKFISH_ARCHIVE_CONTROL_FILE","PRESENT",
                "BYTE_HASH_VERIFIED_ARCHIVE_CONTROL",f"sha256={digest}; bytes={size}; {note}")
            return int(source[0]),artifact_id,local_rel

        p13_archive_manifest_source,p13_archive_manifest_art,_=register_archive_control("ARCHIVE_MANIFEST.tsv","ARCHIVE_MANIFEST")
        p13_archive_report_source,p13_archive_report_art,_=register_archive_control("VERIFICATION_REPORT.json","ARCHIVE_VERIFICATION_REPORT")
        p13_archive_readme_source,p13_archive_readme_art,_=register_archive_control("README.md","ARCHIVE_README")

        p13_file_capture=importer.capture("pr13_file_provenance.tsv")
        p13_file_capture_art=importer.capture_artifacts["pr13_file_provenance.tsv"]
        p13_file_capture_rel=p13_file_capture["local_path"].resolve().relative_to(WORKTREE.resolve()).as_posix()
        for repository_path,(digest,size,role) in sorted(p13_manifest_row_map.items()):
            relative_path=repository_path.split("/",len(p13_prefix.split("/")))[-1]
            tree_row=p13_tree_rows.get(relative_path)
            if not tree_row:
                raise RuntimeError(f"PR #13 Git-tree audit row is missing: {relative_path}")
            archive_row=p13_archive_raw_by_path.get(relative_path)
            if (tree_row["status"]=="MANIFEST_ROW_NOT_TRACKED_IN_EITHER_PINNED_TREE") != (archive_row is not None):
                raise RuntimeError(f"PR #13 archive row does not match the independently reconciled Git absence: {relative_path}")
            tree_status=tree_row["status"]
            evidence_notes=(f"FILE_PROVENANCE_MANIFEST declaration: role={role}; bytes={size}; sha256={digest}. "
                            f"Pinned Git-tree reconciliation status={tree_status}. ")
            if tree_status=="MANIFEST_MATCHES_BOTH_PINNED_GIT_BLOBS":
                blob=tree_row["bundle"]
                evidence_notes+=(f"Both commits contain blob {blob['blob_oid']} ({blob['bytes']} bytes; sha256={blob['sha256']}); "
                                 "manifest size and SHA-256 independently match.")
            elif tree_status=="MANIFEST_DECLARATION_MISMATCHES_BOTH_PINNED_GIT_BLOBS":
                blob=tree_row["bundle"]
                evidence_notes+=(f"Both commits contain blob {blob['blob_oid']} ({blob['bytes']} bytes; sha256={blob['sha256']}); "
                                 "manifest row is retained as a stale declaration and is not used as the content hash.")
            else:
                evidence_notes+=("Path is absent from both pinned Git trees; the declared bytes and SHA-256 are not verified against Git content.")
            if archive_row:
                evidence_notes+=(f" Separate Rockfish archive status=PRESENT_HASH_VERIFIED at "
                    f"{PR13_ARCHIVE_ROOT}/{archive_row['archive_path']} with bytes={archive_row['byte_size']} "
                    f"and sha256={archive_row['sha256']}; archive content was independently re-hashed live.")
            row_locator=p13_file_capture_rel+"#relative_path="+quote(relative_path,safe="")
            importer.add_item_evidence(p13_index,p13_file_capture_art,row_locator,"SOURCE_MANIFEST_ROW","PRESENT",
                tree_status,evidence_notes)
            importer.add_entity_ref(p13_file_source,"PR13_FILE_PROVENANCE_ROW",relative_path,
                "PAPER_RELEASE_ITEM",str(p13_index),tree_status,evidence_notes)
            if tree_status!="MANIFEST_ROW_NOT_TRACKED_IN_EITHER_PINNED_TREE":
                blob=tree_row["pr_head"]
                verified_note=(
                    "SHA-256 and byte size independently verified from Git blob bytes at both pinned commits; "
                    f"blob_oid={blob['blob_oid']}; bundle={p13_manifest['bundle_commit']}; PR_head={p13_meta['headRefOid']}. "
                )
                if tree_status=="MANIFEST_DECLARATION_MISMATCHES_BOTH_PINNED_GIT_BLOBS":
                    verified_note+=(f"Current Git content SHA-256={blob['sha256']} bytes={blob['bytes']}; stale manifest declaration SHA-256={digest} bytes={size} is retained separately.")
                else:
                    verified_note+=("Verified Git content agrees with the source-manifest row.")
                artifact_id=importer.add_package_file(pr_number=13,release_id=p13_id,item_id=p13_index,
                    source_id=p13_audit_source[0],commit=p13_meta["headRefOid"],repository_path=repository_path,
                    digest=blob["sha256"],size=blob["bytes"],declared_role=role,note=verified_note)
                p13_file_map[repository_path]=(blob["sha256"],blob["bytes"],role)
                importer.add_item_evidence(p13_index,artifact_id,
                    github_blob_url(p13_manifest["bundle_commit"],repository_path),"BUNDLE_COMMIT_GIT_BLOB",
                    "EXTERNAL_POINTER","SAME_VERIFIED_BLOB_AT_BUNDLE_AND_PR_HEAD",
                    f"blob_oid={blob['blob_oid']}; sha256={blob['sha256']}; bytes={blob['bytes']}; verified against both pinned trees.")

        for relative_path,archive_row in sorted(p13_archive_raw_by_path.items()):
            repository_path=f"{p13_prefix}/{relative_path}"
            if archive_row.get("source_manifest_path")!=PR13_RAW_SOURCE_MANIFEST_PATH:
                raise RuntimeError(f"PR #13 archive row points to unexpected source manifest: {relative_path}")
            digest=archive_row["sha256"]
            size=int(archive_row["byte_size"])
            root_relative_path=f"{PR13_ARCHIVE_RELATIVE_ROOT}/{archive_row['archive_path']}"
            absolute_path=f"{PR13_ARCHIVE_ROOT}/{archive_row['archive_path']}"
            archive_note=(f"PR #13 raw_evidence bytes independently verified live on Rockfish and match the source manifest; "
                f"source_commit={archive_row['source_commit']}; source_manifest_sha256={archive_row['source_manifest_sha256']}; "
                f"archive_manifest_sha256={PR13_ARCHIVE_MANIFEST_SHA256}; role=raw_evidence; bytes={size}; sha256={digest}. "
                "This is a Rockfish archive location, distinct from the absent Git-tree path.")
            artifact_id=importer.add_artifact(digest,size,"MIGRATIONBENCH_PR13_RAW_ARCHIVE_FILE","RAW_EVIDENCE",archive_note)
            importer.add_artifact_location(artifact_id,ROCKFISH_ROOT_ID,root_relative_path,p13_archive_verified_utc)
            importer.add_item_evidence(p13_index,artifact_id,absolute_path,"ROCKFISH_ARCHIVE_FILE","PRESENT",
                "VERIFIED_ARCHIVE_BYTES_MATCH_SOURCE_MANIFEST",archive_note)
            importer.package_file_artifacts.setdefault(p13_id,{})[repository_path]=artifact_id
            content_key=f"MB-PR-CONTENT-SHA256:{digest}"
            importer.add_entity_ref(p13_archive_manifest_source,"PR13_ARCHIVE_MANIFEST_ROW",relative_path,
                "ARTIFACT",content_key,"PRESENT_ON_RF_SCRATCH16_YI",archive_note)
            importer.add_entity_ref(p13_file_source,"PR13_ARCHIVE_VERIFIED_FILE",relative_path,
                "ARTIFACT",content_key,"SOURCE_ROW_HASH_MATCHES_PRESENT_ARCHIVE_BYTES",archive_note)

        p13_archive_bundle=p13_archive_rows and next(row for row in p13_archive_rows if row["source_role"]=="source_bundle")
        p13_bundle_art=importer.add_artifact(PR13_RAW_SOURCE_BUNDLE_SHA256,PR13_RAW_SOURCE_BUNDLE_BYTES,
            "MIGRATIONBENCH_PR13_RAW_SOURCE_BUNDLE","SOURCE_BUNDLE",
            f"Exact source tar Git blob {PR13_RAW_SOURCE_BUNDLE_OID} from commit {p13_archive_bundle['source_commit']}; SHA-256 verified at Rockfish archive root.")
        importer.add_artifact_location(p13_bundle_art,ROCKFISH_ROOT_ID,
            f"{PR13_ARCHIVE_RELATIVE_ROOT}/source_bundle.tar.gz",p13_archive_verified_utc)
        importer.add_item_evidence(p13_index,p13_bundle_art,
            f"{PR13_ARCHIVE_ROOT}/source_bundle.tar.gz","ROCKFISH_SOURCE_BUNDLE","PRESENT",
            "ARCHIVE_COPY_MATCHES_PINNED_GIT_BLOB",
            f"sha256={PR13_RAW_SOURCE_BUNDLE_SHA256}; bytes={PR13_RAW_SOURCE_BUNDLE_BYTES}; Git blob_oid={PR13_RAW_SOURCE_BUNDLE_OID}; source commit={p13_archive_bundle['source_commit']}.")
        importer.add_item_evidence(p13_index,p13_bundle_art,
            github_blob_url(p13_archive_bundle["source_commit"],p13_archive_bundle["source_manifest_path"]),
            "SOURCE_GIT_BUNDLE","EXTERNAL_POINTER","SAME_HASH_AS_ARCHIVED_SOURCE_BUNDLE",
            f"Git blob_oid={PR13_RAW_SOURCE_BUNDLE_OID}; sha256={PR13_RAW_SOURCE_BUNDLE_SHA256}; bytes={PR13_RAW_SOURCE_BUNDLE_BYTES}.")
        importer.add_entity_ref(p13_archive_manifest_source,"PR13_ARCHIVE_SOURCE_BUNDLE","source_bundle.tar.gz",
            "ARTIFACT",f"MB-PR-CONTENT-SHA256:{PR13_RAW_SOURCE_BUNDLE_SHA256}","PRESENT_ARCHIVE_AND_PINNED_GIT_BLOB",
            f"Git blob_oid={PR13_RAW_SOURCE_BUNDLE_OID}; commit={p13_archive_bundle['source_commit']}; exact SHA-256 and size verified.")

        for filename, repository_path in (
            ("pr13_manifest.json",f"{p13_prefix}/paper_handoff/PAPER_RELEASE_MANIFEST.json"),
            ("pr13_file_provenance.tsv",f"{p13_prefix}/tables/FILE_PROVENANCE_MANIFEST.tsv"),
            ("pr13_claim_matrix.tsv",f"{p13_prefix}/paper_handoff/PAPER_CLAIM_EVIDENCE_MATRIX.tsv"),
            ("pr13_readme.md",f"{p13_prefix}/README.md"),
            ("pr13_table_claim_matrix.tsv",f"{p13_prefix}/tables/CLAIM_EVIDENCE_MATRIX.tsv"),
        ):
            cap=importer.capture(filename)
            value=(cap["capture_sha256"],cap["capture_size"],role_for_path(repository_path))
            if repository_path in p13_file_map:
                if p13_file_map[repository_path][:2]!=value[:2]:
                    raise RuntimeError(f"PR #13 captured Git file differs from audited blob: {repository_path}")
            else:
                # These captured Git files are outside the 1,384-row file manifest.
                importer.add_package_file(pr_number=13,release_id=p13_id,item_id=p13_index,
                    source_id=p13_audit_source[0],commit=p13_meta["headRefOid"],repository_path=repository_path,
                    digest=value[0],size=value[1],declared_role=value[2],
                    note="Current Git file captured from the pinned PR-head commit; not listed in FILE_PROVENANCE_MANIFEST.tsv.")
                p13_file_map[repository_path]=value
            importer.add_source_capture_evidence(p13_id,p13_index,filename,importer.sources[filename],capture_utc)

        figures={
            "FIG1_ENDPOINT47_LATTICE_SITE_DISTRIBUTION":("MB-PR13-FIG1-ENDPOINT47","47-source geometry distribution"),
            "FIG2_RETRY21_DFT_ENERGY_STRUCTURE":("MB-PR13-FIG2-RETRY21","Restricted retry-21 DFT relative-energy panel"),
            "FIG3_FOUR_MODEL_BASIN_SENSITIVITY":("MB-PR13-FIG3-BASIN-SENSITIVITY","Four-model, three-start basin sensitivity"),
            "FIG4_MACE12_SOURCE_GEOMETRY_DISAGREEMENT":("MB-PR13-FIG4-MACE12-DISAGREEMENT","12-model static source-geometry disagreement"),
        }
        for source_fig_id,(catalog_fig_id,title) in figures.items():
            pdf_path=f"{p13_prefix}/figures/{source_fig_id}.pdf"
            pdf_hash=p13_file_map[pdf_path][0]
            assets=[]
            for ext in ("pdf","svg","png"):
                path=f"{p13_prefix}/figures/{source_fig_id}.{ext}"
                info=p13_file_map.get(path)
                if info:
                    aid=importer.add_artifact(info[0],info[1],"MIGRATIONBENCH_PR13_FILE","FIGURE_EXPORT",
                        f"PR #13 figure; Git blob SHA-256 and byte size verified against the source manifest; {p13_manifest['scientific_boundaries'][0]}.")
                    assets.append(("FIGURE_EXPORT",github_blob_url(p13_meta["headRefOid"],path),aid,
                                   "Git blob hash and size independently match the source manifest at both pinned commits; candidate remains under review."))
            script_path=f"{p13_prefix}/scripts/render_publication_figures.py"
            script_info=p13_file_map.get(script_path)
            script_aid=importer.add_artifact(script_info[0],script_info[1],"MIGRATIONBENCH_PR13_FILE","REBUILD_SOURCE",
                "PR #13 figure rebuild source; Git blob SHA-256/size verified against both pinned commits; import does not execute it.") if script_info else None
            assets.append(("REBUILD_METHOD",github_blob_url(p13_meta["headRefOid"],script_path),script_aid,
                           "Source command is recorded in the PR #13 release manifest; not executed here."))
            assets.append(("PROVENANCE_MANIFEST",github_blob_url(p13_meta["headRefOid"],f"{p13_prefix}/paper_handoff/PAPER_RELEASE_MANIFEST.json"),p13_manifest_art,
                           f"Manifest bundle_commit={p13_manifest['bundle_commit']}; PR head={p13_meta['headRefOid']}."))
            importer.add_figure(figure_id=catalog_fig_id,release_id=p13_id,source_id=p13_claim_source,
                title=title,chart_kind="PUBLICATION_CANDIDATE",chart_status="REVIEW_CANDIDATE_NOT_MERGED",
                scientific_boundary="; ".join(p13_manifest["scientific_boundaries"]),assets=assets)
            importer.add_release_item(p13_id,source_fig_id,"PR13 review handoff","figure",title,
                github_blob_url(p13_meta["headRefOid"],pdf_path),pdf_hash,
                github_blob_url(p13_meta["headRefOid"],script_path),None,"REVIEW_CANDIDATE",
                "Release manifest is READY_FOR_REVIEW_NOT_YET_MERGED; do not treat as submitted or accepted.")

        table_map={
            "TABLE1_DFT47_GEOMETRY_ENERGY_AND_ENSEMBLE_SUMMARY":"DFT47 geometry, restricted retry energies, and ensemble summary",
            "TABLE2_LEGACY_FOUR_MODEL_SENSITIVITY":"Legacy four-model sensitivity; Foundation lineage is OMAT-derived fine-tuned",
            "TABLE3_MACE12_STRUCTURAL_ASSOCIATIONS":"MACE12 source-geometry structural associations; exploratory only",
        }
        for table_id,title in table_map.items():
            path=f"{p13_prefix}/tables/{table_id}.tsv"
            info=p13_file_map[path]
            aid=importer.add_artifact(info[0],info[1],"MIGRATIONBENCH_PR13_FILE","COMPACT_RESULT_TABLE",
                                      f"PR #13 table file; Git blob SHA-256/size verified against both pinned commits and matching the source manifest; {title}.")
            item=importer.add_release_item(p13_id,table_id,"PR13 review handoff","table",title,
                github_blob_url(p13_meta["headRefOid"],path),info[0],None,None,"REVIEW_CANDIDATE",
                "Source table is review-candidate evidence and is not part of the submitted Revision 1 release.")
            importer.add_item_evidence(item,aid,github_blob_url(p13_meta["headRefOid"],path),"DERIVED_DATA",
                "EXTERNAL_POINTER","PINNED_GITHUB_COMMIT_PATH","Hash declared in FILE_PROVENANCE_MANIFEST.tsv; payload not copied.")

        for row in p13_claims:
            source_claim=row["claim_id"]
            source_row=f"PR13:{source_claim}"
            named= [x.strip() for x in row["figure_or_table"].split(";") if x.strip()]
            figure_ids=[figures[x][0] for x in named if x in figures]
            primary=row["primary_artifact"]
            primary_path=f"{p13_prefix}/{primary}" if not primary.startswith("../") else primary
            evidence_refs=[primary_path]
            for named_item in named:
                if named_item.startswith("TABLE"):
                    table_path=f"{p13_prefix}/tables/{named_item}.tsv"
                    if table_path in importer.package_file_artifacts[p13_id]:
                        evidence_refs.append(table_path)
                elif named_item in figures:
                    fig_name=figures[named_item][0].replace("MB-PR13-","")
                    figure_path=f"{p13_prefix}/figures/{fig_name}.pdf"
                    if figure_path in importer.package_file_artifacts[p13_id]:
                        evidence_refs.append(figure_path)
            importer.import_claim(release_id=p13_id,claim_id=f"MB-PR13-{source_claim}",source_row_key=source_row,
                source_claim_id=source_claim,source_id=p13_claim_source,claim_text=row["paper_sentence_or_result"],
                source_status=row["status"],manuscript_location=row["manuscript_location"],
                source_or_table=row["figure_or_table"],primary_artifact=primary,rebuild_script="",
                response_location="",boundary=f"Allowed: {row['allowed_wording']}; forbidden/pending: {row['forbidden_or_pending']}; upstream: {row['upstream_dependency']}",
                evidence_refs=evidence_refs,figure_ids=figure_ids,capture_utc=capture_utc,matrix_artifact_id=p13_claim_art)

        # Capture manifest itself is an immutable local audit source.
        manifest_source_key=f"migrationbench_pr_candidate_capture_manifest_20261124:{capture_manifest_digest[:20]}"
        capture_manifest_url=(snapshot_dir/"CAPTURE_MANIFEST.tsv").resolve().as_uri()
        db.execute(
            """INSERT OR IGNORE INTO registry_source
               (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes)
               VALUES(?,?,'PR_CANDIDATE_CAPTURE_MANIFEST',?,?,?,1,?,?)""",
            (manifest_source_key,capture_manifest_url,capture_manifest_digest,capture_utc,capture_utc,
             "Local audit capture manifest for immutable GitHub PR files and API observations.",
             "Generated by the audit capture step; 19 files pinned to three exact PR head commits."),
        )
        capture_manifest_art=importer.add_artifact(capture_manifest_digest,capture_manifest.stat().st_size,
            "PR_SOURCE_CAPTURE_MANIFEST","SOURCE_CAPTURE_INDEX","Hashes and URLs for captured PR source files.")
        local_rel=capture_manifest.resolve().relative_to(WORKTREE.resolve()).as_posix()
        importer.add_artifact_location(capture_manifest_art,LOCAL_ROOT_ID,local_rel,capture_utc)
        for pr_number,release_info in importer.releases.items():
            importer.add_entity_ref(importer.sources[f"pr{pr_number}_metadata.json"],"capture_manifest",
                capture_manifest_digest,"ARTIFACT",f"MB-PR-CONTENT-SHA256:{capture_manifest_digest}","CAPTURE_INDEX",local_rel)

        fk = db.execute("PRAGMA foreign_key_check").fetchall()
        if fk:
            raise RuntimeError(f"Foreign-key violations before commit: {fk[:10]}")
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    # Post-commit counts make idempotency visible without scanning large tables.
    read = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=180)
    read.row_factory=sqlite3.Row
    result={"capture_manifest_sha256":capture_manifest_digest,"capture_utc":capture_utc,
            "releases":{},"claims":{},"figures":{},"package_files":{},"foreign_key_violations":0,
            "submitted_release":None,"import_counters":importer.counts,
            "pr13_git_tree_reconciliation":validation["pr13_git_tree_reconciliation"],
            "pr13_raw_archive":validation["pr13_raw_archive"]}
    for pr,info in sorted(importer.releases.items()):
        rid=info["release_id"]
        result["releases"][str(pr)]=dict(read.execute(
            "SELECT release_id,status,manuscript_commit,manifest_artifact_id FROM paper_release WHERE release_id=?",(rid,)
        ).fetchone())
        result["claims"][str(pr)]=read.execute("SELECT count(*) FROM paper_claim WHERE release_id=?",(rid,)).fetchone()[0]
        result["figures"][str(pr)]=read.execute("SELECT count(*) FROM paper_figure WHERE notes LIKE ?",(f"%{rid}%",)).fetchone()[0]
        result["package_files"][str(pr)]=read.execute(
            """SELECT count(*) FROM paper_release_item_evidence e JOIN paper_release_item i USING(release_item_id)
               WHERE i.release_id=? AND e.evidence_role='PACKAGE_FILE'""",(rid,)
        ).fetchone()[0]
    result["pr13_manifest_rows_by_tree_status"]={row[0]:row[1] for row in read.execute(
        """SELECT r.relationship,count(*) FROM registry_entity_ref r
           JOIN paper_release_item i ON CAST(r.catalog_entity_key AS INTEGER)=i.release_item_id
           WHERE i.release_id GLOB 'MB-PR13-ENDPOINT47-REVIEW-*'
             AND r.entity_type='PR13_FILE_PROVENANCE_ROW'
           GROUP BY r.relationship ORDER BY r.relationship""")}
    result["pr13_archive_file_evidence_count"]=read.execute(
        """SELECT count(*) FROM paper_release_item_evidence e JOIN paper_release_item i USING(release_item_id)
           WHERE i.release_id GLOB 'MB-PR13-ENDPOINT47-REVIEW-*' AND e.evidence_role='ROCKFISH_ARCHIVE_FILE'""").fetchone()[0]
    result["foreign_key_violations"]=len(read.execute("PRAGMA foreign_key_check").fetchall())
    result["submitted_release"]=dict(read.execute(
        "SELECT release_id,status,manuscript_commit,manifest_artifact_id FROM paper_release WHERE release_id='MB-REVISION1-SUBMITTED-20261110'"
    ).fetchone())
    read.close()
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db",type=Path,default=DEFAULT_DB)
    parser.add_argument("--snapshot-dir",type=Path,default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--validate-inputs-only",action="store_true",
                        help="Validate pinned captures, Git-tree audit, and live Rockfish archive bytes without opening or writing SQLite")
    args=parser.parse_args()
    if args.validate_inputs_only:
        print(json.dumps(validate_inputs(args.snapshot_dir),indent=2,sort_keys=True))
        return
    if not args.db.is_file():
        raise FileNotFoundError(args.db)
    result=run_import(args.db,args.snapshot_dir)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
