#!/usr/bin/env python3
"""Import the dated MigrationBench figure archive without asserting release acceptance.

This importer is additive and idempotent. It registers the archive files and the
two source chart registries, creates typed paper_figure rows, links the source
registry records, and imports only verbatim claim rows present in archived
claim/evidence matrices. Missing source paths and rebuild commands remain
explicit pointers; no legacy Slurm job number is joined to a workflow_run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDIT = Path(__file__).resolve().parent.parent
REPO = AUDIT.parents[1]
DB = AUDIT / "provenance_catalog.sqlite"
ARCHIVE = REPO / "results/plots/paper_figure_archive_20261028"
CORE_DB = REPO / "registry/snapshots/migrationbench_asset_registry_core_20261101.sqlite"
PLOT_DB = REPO / "results/plots/registry/migrationbench_asset_registry_20261101.sqlite"
LOCAL_ROOT = "LOCAL-MAIN"
RELEASE_ID = "MB-FIGURE-ARCHIVE-SNAPSHOT-20261028"
STATIC_639_RELEASE_ID = "MB-STATIC-639-HANDOFF-20261014"
DFT_MACE47_RELEASE_ID = "MB-DFT-MACE47-HANDOFF-CANDIDATE-20261023"
STATIC_639_ROOT = REPO / "analysis/multimodel_639_screen_20261014"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def resolve_repo_path(value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        try:
            path.resolve(strict=False).relative_to(REPO.resolve())
            return path.resolve(strict=False)
        except ValueError:
            return None
    return (REPO / path).resolve(strict=False)


def scientific_status(source_status: str) -> str:
    if source_status == "FAILURE_MODE":
        return "FAILURE_MODE"
    if source_status == "RECONSTRUCTION":
        return "RECONSTRUCTION"
    if source_status in ("EXCLUDE_FROM_PUBLICATION", "VERIFIED", "LINEAGE_SUPPORTED"):
        # Keep chart-level source labels verbatim in notes. The archive importer
        # itself does not re-audit all scientific inputs or scheduler records.
        return "PROVENANCE_PARTIAL"
    return "PROVENANCE_PARTIAL"


def action_status(source_action: str, source_status: str) -> str:
    value = (source_action or "").strip().lower()
    if "pending" in value or source_status == "PENDING":
        return "PENDING"
    if "exclude" in value:
        return "EXCLUDE"
    if value == "include":
        return "INCLUDE"
    if "include" in value:
        return "QUALIFY"
    if source_status == "RECONSTRUCTION":
        return "QUALIFY"
    return "PENDING"


def add_registry_source(
    con: sqlite3.Connection,
    *,
    key: str,
    path: Path,
    kind: str,
    captured: str,
    as_of: str | None,
    note: str,
) -> int:
    digest = sha256(path)
    absolute = str(path.resolve())
    existing = con.execute(
        "SELECT source_id,source_path,sha256 FROM registry_source WHERE source_key=?", (key,)
    ).fetchone()
    if existing and existing[1:] != (absolute, digest):
        raise RuntimeError(f"Registry source key changed content/path: {key}")
    con.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
            immutable_snapshot,authority_note,notes)
           VALUES(?,?,?,?,?,?,1,?,?)""",
        (key, absolute, kind, digest, captured, as_of,
         "Captured file snapshot; source remains authoritative only for the entities it contains.", note),
    )
    return con.execute("SELECT source_id FROM registry_source WHERE source_key=?", (key,)).fetchone()[0]


def ensure_artifact(
    con: sqlite3.Connection,
    path: Path,
    *,
    artifact_key: str,
    artifact_kind: str,
    role: str,
    scientific: str,
    retention: str = "PUBLICATION_EVIDENCE",
    note: str = "",
    now: str,
) -> tuple[int, str]:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise RuntimeError(f"Expected a regular file: {path}")
    try:
        relative = path.relative_to(REPO.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"Artifact path is outside catalogued local root: {path}") from exc
    digest = sha256(path)
    size = path.stat().st_size
    prior = con.execute("SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()
    if prior and prior[1:] != (digest, size):
        raise RuntimeError(f"Artifact key changed identity: {artifact_key}")
    con.execute(
        """INSERT OR IGNORE INTO artifact
           (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
            scientific_status,retention_class,notes)
           VALUES('MIGRATIONBENCH',?,?,?,?,?,?,?, ?,?)""",
        (artifact_key, artifact_kind, role, "SHARED_OWNER_CONFIRMED", digest, size,
         scientific, retention, note),
    )
    artifact_id = con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (artifact_key,)).fetchone()[0]
    root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (LOCAL_ROOT,)).fetchone()
    if not root or Path(root[0]).resolve() != REPO.resolve():
        raise RuntimeError("LOCAL-MAIN must resolve to this unified local repository")
    con.execute(
        "INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)",
        (LOCAL_ROOT, relative, now),
    )
    path_id = con.execute(
        "SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (LOCAL_ROOT, relative)
    ).fetchone()[0]
    con.execute(
        """INSERT OR IGNORE INTO artifact_location
           (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
           VALUES(?,?,'PRESENT',?,?)""",
        (artifact_id, path_id, now, now),
    )
    return artifact_id, digest


def add_external_ref(
    con: sqlite3.Connection, source_id: int, entity_type: str, entity_key: str,
    target_type: str, target_key: str, relationship: str, notes: str = "",
) -> None:
    con.execute(
        """INSERT OR IGNORE INTO registry_entity_ref
           (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
           VALUES(?,?,?,?,?,?,?)""",
        (source_id, entity_type, entity_key, target_type, target_key, relationship, notes),
    )


def add_figure_asset(
    con: sqlite3.Connection, figure_id: str, role: str, source_path: str,
    availability: str, relationship: str, notes: str = "", artifact_id: int | None = None,
) -> None:
    con.execute(
        """INSERT OR IGNORE INTO paper_figure_asset
           (figure_id,artifact_id,asset_role,source_path,availability,relationship,notes)
           VALUES(?,?,?,?,?,?,?)""",
        (figure_id, artifact_id, role, source_path, availability, relationship, notes),
    )


def add_claim_evidence(
    con: sqlite3.Connection, claim_id: str, artifact_id: int, evidence_role: str,
    support_status: str, notes: str,
) -> None:
    # SQLite UNIQUE constraints treat NULL run_id values as distinct, so do an
    # explicit NULL-aware existence check before inserting.
    exists = con.execute(
        """SELECT 1 FROM claim_evidence
           WHERE claim_id=? AND artifact_id=? AND run_id IS NULL AND evidence_role=? LIMIT 1""",
        (claim_id, artifact_id, evidence_role),
    ).fetchone()
    if not exists:
        con.execute(
            """INSERT INTO claim_evidence
               (claim_id,artifact_id,run_id,evidence_role,support_status,notes)
               VALUES(?,?,NULL,?,?,?)""",
            (claim_id, artifact_id, evidence_role, support_status, notes),
        )


def main() -> None:
    if not all(p.exists() for p in (DB, ARCHIVE, CORE_DB, PLOT_DB)):
        raise FileNotFoundError("Catalog, figure archive, and both registry snapshots must exist")
    now = now_utc()
    migration = Path(__file__).with_name("009_paper_figure_archive.sql")
    migration_hash = sha256(migration)

    con = sqlite3.connect(DB, timeout=180)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=180000")
    try:
        applied = con.execute("SELECT migration_file FROM schema_migration WHERE version=9").fetchone()
        if applied:
            if applied[0] != migration.name:
                raise RuntimeError("Schema migration version 9 is already assigned to another file")
            if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_figure'").fetchone():
                raise RuntimeError("Migration 009 is recorded but paper_figure is missing")
        else:
            if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_figure'").fetchone():
                raise RuntimeError("paper_figure exists without migration 009; refusing to infer schema state")
            con.executescript(migration.read_text(encoding="utf-8"))
            con.execute(
                "INSERT INTO schema_migration(version,migration_file,applied_utc,description) VALUES(9,?,?,?)",
                (migration.name, now, "Add typed paper figure rows and artifact/pointer links for archived figure provenance."),
            )
            con.commit()

        migration10 = Path(__file__).with_name("010_paper_claim_figure_links.sql")
        applied10 = con.execute("SELECT migration_file FROM schema_migration WHERE version=10").fetchone()
        if applied10:
            if applied10[0] != migration10.name or not con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_claim_figure'"
            ).fetchone():
                raise RuntimeError("Migration 010 state does not match the paper-claim/figure link schema")
        else:
            if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_claim_figure'").fetchone():
                raise RuntimeError("paper_claim_figure exists without migration 010")
            con.executescript(migration10.read_text(encoding="utf-8"))
            con.execute(
                "INSERT INTO schema_migration(version,migration_file,applied_utc,description) VALUES(10,?,?,?)",
                (migration10.name, now, "Add explicit source-backed links between paper claims and figure entities."),
            )
            con.commit()

        con.execute("BEGIN IMMEDIATE")
        root = con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?", (LOCAL_ROOT,)).fetchone()
        if not root or Path(root[0]).resolve() != REPO.resolve():
            raise RuntimeError("Expected storage root LOCAL-MAIN at the unified worktree")

        # Preserve the two dated SQLite snapshots as separate source records.
        core_source = con.execute(
            "SELECT source_id,sha256,source_path FROM registry_source WHERE source_key='migrationbench_asset_registry_core_20261101'"
        ).fetchone()
        plot_source = con.execute(
            "SELECT source_id,sha256,source_path FROM registry_source WHERE source_key='migrationbench_plot_registry_20261101'"
        ).fetchone()
        for existing, path in ((core_source, CORE_DB), (plot_source, PLOT_DB)):
            if not existing or existing[1] != sha256(path) or existing[2] != str(path.resolve()):
                raise RuntimeError(f"Existing registry source does not match on-disk snapshot: {path}")
        core_id, plot_id = core_source[0], plot_source[0]

        # Rebuild only this importer's normalized registry-to-file links so a
        # rerun can correct mappings without deleting independent catalog data.
        con.execute(
            "DELETE FROM registry_entity_ref WHERE source_id IN (?,?) AND entity_type IN (?, ?, ?, ?)",
            (core_id, plot_id, "chart_data_file", "chart_provenance", "chart_data_file_archive_copy", "chart_provenance_archive_copy"),
        )
        con.execute(
            """DELETE FROM paper_figure_asset WHERE relationship IN
               ('REGISTRY_DATA_PATH_RESOLVED','REGISTRY_DATA_PATH_UNRESOLVED',
                'REGISTRY_MANIFEST_RESOLVED','REGISTRY_MANIFEST_POINTER_ONLY',
                'INDEX_POINTER_RESOLVED','INDEX_POINTER_ARCHIVE_COPY')"""
        )

        archive_sources: dict[str, int] = {}
        source_specs = [
            ("paper_figure_reproduction_index_20261028", ARCHIVE / "index/FIGURE_REPRODUCTION_INDEX.tsv", "PAPER_FIGURE_REPRODUCTION_INDEX", "2026-10-28T00:00:00Z"),
            ("paper_figure_rebuild_script_pointers_20261028", ARCHIVE / "scripts/REBUILD_SCRIPT_POINTERS.tsv", "PAPER_FIGURE_REBUILD_POINTERS", "2026-10-28T00:00:00Z"),
            ("paper_figure_source_pointers_20261028", ARCHIVE / "index/SOURCE_POINTERS.tsv", "PAPER_FIGURE_SOURCE_POINTERS", "2026-10-28T00:00:00Z"),
            ("paper_figure_release_selection_20261028", ARCHIVE / "index/release_selection.tsv", "PAPER_FIGURE_SELECTION_SNAPSHOT", "2026-10-28T00:00:00Z"),
            ("paper_figure_candidate_index_20261028", ARCHIVE / "index/FIGURE_CANDIDATES.tsv", "PAPER_FIGURE_CANDIDATE_INDEX", "2026-10-28T00:00:00Z"),
            ("paper_claim_matrix_static_639_20261014", ARCHIVE / "provenance/chart_static_639_family_grouped_20261014/CLAIM_EVIDENCE_MATRIX.tsv", "PAPER_CLAIM_EVIDENCE_MATRIX", "2026-10-14T00:00:00Z"),
            ("paper_claim_matrix_md_fig004_20261016", ARCHIVE / "provenance/chart_md_1ns_fig004_a_to_g_20261016/CLAIM_EVIDENCE_MATRIX.tsv", "PAPER_CLAIM_EVIDENCE_MATRIX", "2026-10-16T00:00:00Z"),
            ("paper_figure_archive_readme_20261028", ARCHIVE / "README.md", "PAPER_FIGURE_ARCHIVE_README", "2026-10-28T00:00:00Z"),
            ("migrationbench_static639_handoff_manifest_20261014", STATIC_639_ROOT / "PAPER_RELEASE_MANIFEST.json", "PAPER_RELEASE_MANIFEST", "2026-10-14T00:00:00Z"),
            ("migrationbench_dft_mace47_handoff_manifest_20261023", ARCHIVE / "provenance/chart_dft_mace47_basin_sensitivity_20261023/PAPER_RELEASE_MANIFEST.json", "PAPER_HANDOFF_CANDIDATE_MANIFEST", "2026-10-23T00:00:00Z"),
        ]
        for key, path, kind, as_of in source_specs:
            archive_sources[key] = add_registry_source(
                con, key=key, path=path, kind=kind, captured=now, as_of=as_of,
                note="Local dated figure or handoff snapshot; preserves declared status and source pointers without upgrading acceptance.")

        # Read both external registries. The plot snapshot is preferred when a
        # chart ID exists in both; both source records remain linked below.
        registry_rows: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {"core": {}, "plot": {}}
        campaign_rows: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {"core": {}, "plot": {}}
        for label, path, source_id in (("core", CORE_DB, core_id), ("plot", PLOT_DB, plot_id)):
            src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            src.row_factory = sqlite3.Row
            for row in src.execute("SELECT * FROM charts"):
                registry_rows[label][row["chart_id"]] = (source_id, dict(row))
            for row in src.execute("SELECT * FROM campaigns"):
                campaign_rows[label][row["campaign_id"]] = (source_id, dict(row))
            src.close()

        index_rows = tsv(ARCHIVE / "index/FIGURE_REPRODUCTION_INDEX.tsv")
        index_by_chart = {row["chart_id"]: row for row in index_rows}
        script_rows = tsv(ARCHIVE / "scripts/REBUILD_SCRIPT_POINTERS.tsv")
        script_by_chart = {row["chart_id"]: row for row in script_rows}

        # Add registry snapshots and all files actually materialized in the
        # archive as first-class artifacts with path and content identity.
        artifact_for_repo_path: dict[str, tuple[int, str]] = {}
        rel_artifacts: dict[str, tuple[int, str]] = {}
        for path, key, kind, role, scientific, note in (
            (CORE_DB, "PAPER-REGISTRY-SNAPSHOT:core-20261101", "SQLITE_REGISTRY_SNAPSHOT", "Immutable MigrationBench core asset registry snapshot", "LINEAGE_SUPPORTED", "Registry snapshot; it does not prove an Overleaf release."),
            (PLOT_DB, "PAPER-REGISTRY-SNAPSHOT:plot-20261101", "SQLITE_REGISTRY_SNAPSHOT", "Immutable MigrationBench plot registry snapshot", "LINEAGE_SUPPORTED", "Registry snapshot; it preserves a separate dated chart-registry version."),
        ):
            aid, digest = ensure_artifact(con, path, artifact_key=key, artifact_kind=kind, role=role,
                                          scientific=scientific, retention="PERMANENT_SOURCE", note=note, now=now)
            rel_artifacts[path.relative_to(REPO).as_posix()] = (aid, digest)
            artifact_for_repo_path[str(path.resolve())] = (aid, digest)

        actual_files: list[Path] = []
        for walk_root, dirs, files in os.walk(ARCHIVE):
            dirs.sort()
            for name in sorted(files):
                path = Path(walk_root) / name
                if path.is_symlink() or not path.is_file():
                    continue
                actual_files.append(path)
        for path in actual_files:
            rel = path.relative_to(REPO).as_posix()
            archive_rel = path.relative_to(ARCHIVE).as_posix()
            chart_id = ""
            top = Path(archive_rel).parts
            if len(top) >= 3 and top[0] in ("figures", "data", "provenance"):
                chart_id = top[1]
            suffix = path.suffix.lower().lstrip(".") or "unknown"
            if top and top[0] == "figures":
                artifact_kind = "PAPER_FIGURE_EXPORT_" + suffix.upper()
                role = "Archived paper-facing figure export"
            elif top and top[0] == "data":
                artifact_kind = "PAPER_FIGURE_DERIVED_DATA_" + suffix.upper()
                role = "Archived figure-derived data or compact evidence table"
            elif top and top[0] == "provenance":
                artifact_kind = "PAPER_FIGURE_PROVENANCE_" + suffix.upper()
                role = "Archived figure provenance or claim evidence"
            elif top and top[0] == "scripts":
                artifact_kind = "PAPER_FIGURE_REBUILD_POINTER_" + suffix.upper()
                role = "Figure rebuild pointer index"
            else:
                artifact_kind = "PAPER_FIGURE_ARCHIVE_METADATA_" + suffix.upper()
                role = "Dated figure archive index or explanatory metadata"
            status_source = index_by_chart.get(chart_id, {}).get("provenance_status", "PROVENANCE_PARTIAL")
            artifact_id, digest = ensure_artifact(
                con, path, artifact_key="PAPER-ARCHIVE-20261028:" + archive_rel,
                artifact_kind=artifact_kind, role=role, scientific=scientific_status(status_source),
                retention="PUBLICATION_EVIDENCE",
                note=(f"Dated local archive member for {chart_id or 'archive metadata'}; source chart status="
                      f"{index_by_chart.get(chart_id, {}).get('chart_status', 'not chart-specific')}; "
                      "not a standalone manuscript-release assertion."), now=now)
            rel_artifacts[rel] = (artifact_id, digest)
            artifact_for_repo_path[str(path.resolve())] = (artifact_id, digest)

        def find_archive_member(chart_id: str, basename: str) -> tuple[Path, tuple[int, str]] | None:
            matches = [p for p in actual_files if p.parent.name == chart_id and p.name == basename]
            if len(matches) == 1:
                p = matches[0]
                return p, rel_artifacts[p.relative_to(REPO).as_posix()]
            return None

        def register_repo_source_if_present(path_text: str, role: str, note: str) -> tuple[Path | None, tuple[int, str] | None]:
            path = resolve_repo_path(path_text)
            if path is None or not path.is_file():
                return path, None
            existing = artifact_for_repo_path.get(str(path.resolve()))
            if existing:
                return path, existing
            try:
                rel = path.relative_to(REPO).as_posix()
            except ValueError:
                return path, None
            aid, digest = ensure_artifact(
                con, path, artifact_key="PAPER-SOURCE:LOCAL-MAIN:" + rel,
                artifact_kind="PAPER_CLAIM_SOURCE_" + (path.suffix.lower().lstrip(".").upper() or "FILE"),
                role=role, scientific="PROVENANCE_PARTIAL", retention="PUBLICATION_EVIDENCE", note=note, now=now)
            artifact_for_repo_path[str(path.resolve())] = (aid, digest)
            rel_artifacts[rel] = (aid, digest)
            return path, (aid, digest)

        def register_static639_source(source_ref: str, role: str, note: str) -> tuple[Path | None, tuple[int, str] | None]:
            if not source_ref or source_ref.lower().startswith("from-scratch"):
                return None, None
            candidate = (STATIC_639_ROOT / source_ref).resolve(strict=False)
            try:
                candidate.relative_to(STATIC_639_ROOT.resolve())
            except ValueError:
                return None, None
            if not candidate.is_file():
                return candidate, None
            existing = artifact_for_repo_path.get(str(candidate.resolve()))
            if existing:
                return candidate, existing
            rel = candidate.relative_to(REPO).as_posix()
            artifact_id, digest = ensure_artifact(
                con, candidate, artifact_key="PAPER-SOURCE:LOCAL-MAIN:" + rel,
                artifact_kind="PAPER_CLAIM_SOURCE_" + (candidate.suffix.lower().lstrip(".").upper() or "FILE"),
                role=role, scientific="PROVENANCE_PARTIAL", retention="PUBLICATION_EVIDENCE",
                note=note + " Source path is relative to the archived static-639 campaign directory.", now=now)
            artifact_for_repo_path[str(candidate.resolve())] = (artifact_id, digest)
            rel_artifacts[rel] = (artifact_id, digest)
            return candidate, (artifact_id, digest)

        # Materialize typed campaigns and registry chart rows, leaving the two
        # dated registry representations separately queryable via source refs.
        external_to_campaign: dict[str, str] = {}
        all_campaign_ids = set(campaign_rows["core"]) | set(campaign_rows["plot"])
        for chart_source in (registry_rows["core"], registry_rows["plot"]):
            all_campaign_ids.update(row.get("campaign_id") for _, row in chart_source.values() if row.get("campaign_id"))
        for ext_id in sorted(x for x in all_campaign_ids if x):
            candidate = campaign_rows["plot"].get(ext_id) or campaign_rows["core"].get(ext_id)
            row = candidate[1] if candidate else {"title": ext_id, "status": "UNKNOWN", "campaign_date": None, "notes": "Referenced by a chart record but absent from the source campaign table."}
            existing = con.execute(
                "SELECT campaign_id FROM campaign WHERE project_id='MIGRATIONBENCH' AND source_campaign_key=?", (ext_id,)
            ).fetchone()
            catalog_campaign = existing[0] if existing else "MBREG-" + ext_id
            source_status = row.get("status", "UNKNOWN")
            status = source_status if source_status in ("ACTIVE", "COMPLETE", "FAILED", "SUPERSEDED", "ARCHIVED", "UNKNOWN") else "UNKNOWN"
            if not existing:
                collision = con.execute("SELECT project_id,source_campaign_key FROM campaign WHERE campaign_id=?", (catalog_campaign,)).fetchone()
                if collision:
                    raise RuntimeError(f"Catalog campaign ID collision: {catalog_campaign}")
                con.execute(
                    """INSERT INTO campaign
                       (campaign_id,project_id,campaign_name,source_campaign_key,start_date,status,notes)
                       VALUES(?,'MIGRATIONBENCH',?,?,?,?,?)""",
                    (catalog_campaign, row.get("title") or ext_id, ext_id, row.get("campaign_date"), status,
                     "Imported from a dated chart registry; source campaign key is preserved. " + str(row.get("notes", ""))),
                )
            external_to_campaign[ext_id] = catalog_campaign
            for label in ("core", "plot"):
                source = campaign_rows[label].get(ext_id)
                if source:
                    add_external_ref(con, source[0], "campaign", ext_id, "CAMPAIGN", catalog_campaign,
                                     "SOURCE_CAMPAIGN_RECORD", f"Source registry={label}; dated registry snapshot.")

        # Chart ids are stable in both registry versions. Prefer plot snapshot
        # fields for the current record, while preserving every source version.
        chart_ids = sorted(set(registry_rows["core"]) | set(registry_rows["plot"]) | set(index_by_chart))
        for chart_id in chart_ids:
            chosen = registry_rows["plot"].get(chart_id) or registry_rows["core"].get(chart_id)
            if not chosen:
                raise RuntimeError(f"Archive index chart absent from both registries: {chart_id}")
            chosen_source_id, source_row = chosen
            idx = index_by_chart.get(chart_id, {})
            campaign_key = source_row.get("campaign_id") or idx.get("campaign_id") or ""
            catalog_campaign = external_to_campaign.get(campaign_key)
            title = idx.get("title") or source_row.get("title") or chart_id
            status = idx.get("chart_status") or source_row.get("chart_status") or "UNKNOWN"
            provenance = idx.get("provenance_status") or "PROVENANCE_PARTIAL"
            boundary = idx.get("notes") or source_row.get("notes") or ""
            processing = idx.get("processing_command") or ""
            con.execute(
                """INSERT OR IGNORE INTO paper_figure
                   (figure_id,project_id,campaign_id,source_campaign_key,title,chart_kind,chart_status,
                    provenance_status,source_registry_source_id,data_query,processing_command,scientific_boundary,notes)
                   VALUES(?,'MIGRATIONBENCH',?,?,?,?,?,?,?,?,?,?,?)""",
                (chart_id, catalog_campaign, campaign_key or None, title,
                 source_row.get("chart_kind") or "unknown", status, provenance, chosen_source_id,
                 source_row.get("data_query") or "", processing,
                 boundary, "Archive index and registry snapshot are historical observations; source-specific status is preserved."),
            )
            existing_figure = con.execute(
                "SELECT project_id,campaign_id,title,chart_kind,chart_status,provenance_status FROM paper_figure WHERE figure_id=?",
                (chart_id,),
            ).fetchone()
            expected = ("MIGRATIONBENCH", catalog_campaign, title, source_row.get("chart_kind") or "unknown", status, provenance)
            if existing_figure != expected:
                raise RuntimeError(f"Existing paper_figure conflicts with source snapshots: {chart_id}")
            for label in ("core", "plot"):
                source_chart = registry_rows[label].get(chart_id)
                if source_chart:
                    source_id, row = source_chart
                    add_external_ref(con, source_id, "chart", chart_id, "PAPER_FIGURE", chart_id,
                                     "SOURCE_CHART_RECORD", f"Registry snapshot={label}; source chart_status={row.get('chart_status')}; source title={row.get('title')}.")
                    if row.get("campaign_id"):
                        target_campaign = external_to_campaign.get(row["campaign_id"], "EXTERNAL:" + row["campaign_id"])
                        add_external_ref(con, source_id, "chart_campaign", chart_id + "|" + row["campaign_id"],
                                         "CAMPAIGN", target_campaign, "CHART_CAMPAIGN_LINK")
            if chart_id in index_by_chart:
                add_external_ref(con, archive_sources["paper_figure_reproduction_index_20261028"],
                                 "figure_index_row", chart_id, "PAPER_FIGURE", chart_id, "ARCHIVE_INDEX_ROW",
                                 "Index status is retained verbatim; archive candidacy does not imply manuscript selection.")

            # Link all physical members of the chart's archived bundle.
            for path in actual_files:
                parts = path.relative_to(ARCHIVE).parts
                if len(parts) >= 3 and parts[0] in ("figures", "data", "provenance") and parts[1] == chart_id:
                    aid, _ = rel_artifacts[path.relative_to(REPO).as_posix()]
                    role = {"figures": "FIGURE_EXPORT", "data": "DERIVED_DATA", "provenance": "CLAIM_MATRIX" if "CLAIM_EVIDENCE_MATRIX" in path.name else "PROVENANCE_MANIFEST"}[parts[0]]
                    add_figure_asset(con, chart_id, role, path.relative_to(REPO).as_posix(), "PRESENT",
                                     "ARCHIVED_BUNDLE_MEMBER", "Physical local archive member; content hash and size are in artifact.", aid)

            # Resolve the index's declared archive paths to local archive copies
            # by chart ID and basename, otherwise retain a non-present pointer.
            for col, role in (("figure_archive_paths", "FIGURE_EXPORT"), ("data_archive_paths", "DERIVED_DATA"), ("manifest_archive_path", "PROVENANCE_MANIFEST")):
                for declared in filter(None, (x.strip() for x in idx.get(col, "").split(";"))):
                    basename = Path(declared).name
                    found = find_archive_member(chart_id, basename)
                    if found:
                        path, (aid, _) = found
                        add_figure_asset(con, chart_id, role, declared, "PRESENT", "INDEX_POINTER_ARCHIVE_COPY",
                                         "Declared archive path resolves to this physical archive copy by chart ID and exact basename: " + path.relative_to(REPO).as_posix() + "; this does not establish equality with the former source checkout.", aid)
                    else:
                        add_figure_asset(con, chart_id, role, declared, "NOT_MATERIALIZED", "INDEX_POINTER_ONLY",
                                         "Declared archive path is not present as a member in this local archive copy.")

            # Preserve canonical source asset and manifest paths independently
            # of the archive copies, including missing locations.
            for path_key, role in (("source_asset_path", "SOURCE_ASSET"), ("source_manifest_path", "SOURCE_MANIFEST")):
                source_path = idx.get(path_key, "")
                if not source_path:
                    continue
                resolved, artifact = register_repo_source_if_present(source_path, "MigrationBench figure source artifact", f"Source pointer from figure index {chart_id}.")
                if artifact:
                    add_figure_asset(con, chart_id, role, source_path, "PRESENT", "SOURCE_PATH_RESOLVED",
                                     "Current local source file; identity recorded on linked artifact.", artifact[0])
                else:
                    add_figure_asset(con, chart_id, role, source_path, "MISSING", "SOURCE_PATH_NOT_FOUND",
                                     "The local source path is not present in the unified worktree; do not treat as an extant input.")

            script = script_by_chart.get(chart_id)
            command = (script or {}).get("script_path_or_command") or processing
            if command:
                script_status = (script or {}).get("script_status", "COMMAND_POINTER_ONLY")
                add_figure_asset(con, chart_id, "REBUILD_METHOD", "command:" + command,
                                 "NOT_MATERIALIZED", "REBUILD_COMMAND_OR_POINTER",
                                 f"script_status={script_status}; command is preserved as a reproduction pointer, not a verified executable.")
                if script:
                    add_external_ref(con, archive_sources["paper_figure_rebuild_script_pointers_20261028"],
                                     "rebuild_pointer", chart_id, "PAPER_FIGURE", chart_id,
                                     "REBUILD_METHOD_POINTER", f"script_status={script_status}")

        # Link the actual static-639 handoff manifest and renderer from the
        # unified source tree, separately from the archive's pointer-only note.
        static_manifest = STATIC_639_ROOT / "PAPER_RELEASE_MANIFEST.json"
        static_manifest_pair = ensure_artifact(
            con, static_manifest, artifact_key="PAPER-SOURCE:LOCAL-MAIN:" + static_manifest.relative_to(REPO).as_posix(),
            artifact_kind="PAPER_RELEASE_MANIFEST", role="Static-639 paper handoff manifest",
            scientific="LINEAGE_SUPPORTED", retention="PUBLICATION_EVIDENCE",
            note="Source manifest says HANDOFF_READY and HANDOFF_READY_NOT_SYNCHRONIZED; it is not an Overleaf release.", now=now)
        artifact_for_repo_path[str(static_manifest.resolve())] = static_manifest_pair
        rel_artifacts[static_manifest.relative_to(REPO).as_posix()] = static_manifest_pair
        renderer = STATIC_639_ROOT / "plot_multimodel_overview.py"
        renderer_pair = ensure_artifact(
            con, renderer, artifact_key="PAPER-SOURCE:LOCAL-MAIN:" + renderer.relative_to(REPO).as_posix(),
            artifact_kind="PAPER_FIGURE_RENDERER", role="Static-639 figure renderer script",
            scientific="PROVENANCE_PARTIAL", retention="PERMANENT_SOURCE",
            note="Renderer exists in the unified source tree; archive script index nevertheless preserves its original command-pointer-only state.", now=now)
        artifact_for_repo_path[str(renderer.resolve())] = renderer_pair
        rel_artifacts[renderer.relative_to(REPO).as_posix()] = renderer_pair
        add_figure_asset(con, "chart_static_639_family_grouped_20261014", "PROVENANCE_MANIFEST",
                         static_manifest.relative_to(REPO).as_posix(), "PRESENT", "STATIC639_HANDOFF_MANIFEST",
                         "Manifest records commit c0aee66 and Overleaf not synchronized.", static_manifest_pair[0])
        add_figure_asset(con, "chart_static_639_family_grouped_20261014", "REBUILD_METHOD",
                         renderer.relative_to(REPO).as_posix(), "PRESENT", "RENDERER_SCRIPT_PRESENT",
                         "Exact renderer file is present and hashed; environment/library path remains in the manifest command.", renderer_pair[0])

        # Keep a second, distinct handoff candidate for DFT-MACE47.
        repository_row = con.execute(
            "SELECT repository_id FROM git_repository WHERE repository_key='local-merge-candidate:MigrationBench:2026-11-22'"
        ).fetchone()
        repository_id = repository_row[0] if repository_row else None
        dft_manifest = ARCHIVE / "provenance/chart_dft_mace47_basin_sensitivity_20261023/PAPER_RELEASE_MANIFEST.json"
        dft_manifest_pair = rel_artifacts[dft_manifest.relative_to(REPO).as_posix()]
        dft_manifest_json = json.loads(dft_manifest.read_text(encoding="utf-8"))
        dft_commit = subprocess.run(
            ["git", "rev-parse", dft_manifest_json["bundle_commit"]], cwd=REPO,
            check=True, text=True, capture_output=True,
        ).stdout.strip()
        current_main = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, text=True, capture_output=True,
        ).stdout.strip()
        dft_is_ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", dft_commit, current_main], cwd=REPO,
        ).returncode == 0
        dft_relation = "is an ancestor of" if dft_is_ancestor else "is not an ancestor of"
        con.execute(
            """INSERT OR IGNORE INTO paper_release
               (release_id,project_id,release_label,manuscript_commit,repository_id,overleaf_version,status,manifest_artifact_id,notes)
               VALUES(?,'MIGRATIONBENCH',?,?,?,?, 'REVIEW',?,?)""",
            (DFT_MACE47_RELEASE_ID, "DFT-MACE47 handoff candidate from 2026-10-23", dft_commit, repository_id,
             None, dft_manifest_pair[0],
             f"Manifest status {dft_manifest_json.get('release_status')}; candidate commit {dft_commit} {dft_relation} unified main {current_main} at catalog import. The archive has FIG3 and derived tables, but its C1-C6 claim matrix is not materialized here; do not join to collaborator DFT-NEB runs."),
        )
        con.execute(
            "UPDATE paper_release SET overleaf_version=NULL,notes=? WHERE release_id=?",
            (f"Manifest status {dft_manifest_json.get('release_status')}; candidate commit {dft_commit} {dft_relation} unified main {current_main} at catalog import. Manifest's PENDING_USER_HANDOFF is a handoff state, not an Overleaf version. The archive has FIG3 and derived tables, but its C1-C6 claim matrix is not materialized here; do not join to collaborator DFT-NEB runs.", DFT_MACE47_RELEASE_ID),
        )

        static_manifest_source = archive_sources["migrationbench_static639_handoff_manifest_20261014"]
        static_manifest_json = json.loads(static_manifest.read_text(encoding="utf-8"))
        static_commit = subprocess.run(
            ["git", "rev-parse", static_manifest_json["github_commit"]], cwd=REPO,
            check=True, text=True, capture_output=True,
        ).stdout.strip()
        static_release_id = STATIC_639_RELEASE_ID
        main_branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=REPO, check=True, text=True, capture_output=True,
        ).stdout.strip()
        if main_branch != "main":
            raise RuntimeError(f"Expected unified repository on main, found {main_branch}")
        static_relation = "is" if subprocess.run(
            ["git", "merge-base", "--is-ancestor", static_commit, current_main], cwd=REPO,
        ).returncode == 0 else "is not"
        con.execute(
            """INSERT OR IGNORE INTO paper_release
               (release_id,project_id,release_label,manuscript_commit,repository_id,overleaf_version,status,manifest_artifact_id,notes)
               VALUES(?,'MIGRATIONBENCH',?,?,?,?, 'REVIEW',?,?)""",
            (static_release_id, "Static-639 handoff candidate from 2026-10-14", static_commit, repository_id,
             static_manifest_json.get("overleaf_version"), static_manifest_pair[0],
             f"Source manifest status {static_manifest_json.get('status')}; Overleaf status {static_manifest_json.get('overleaf_status')}; commit {static_commit} {static_relation} an ancestor of unified main {current_main} at catalog import. The source files exist locally and the source matrix is imported separately; catalog review does not constitute publication acceptance."),
        )
        add_external_ref(con, static_manifest_source, "paper_release_manifest", STATIC_639_RELEASE_ID,
                         "PAPER_RELEASE", STATIC_639_RELEASE_ID, "HANDOFF_CANDIDATE_MANIFEST")
        add_external_ref(con, archive_sources["migrationbench_dft_mace47_handoff_manifest_20261023"],
                         "paper_release_manifest", DFT_MACE47_RELEASE_ID, "PAPER_RELEASE",
                         DFT_MACE47_RELEASE_ID, "HANDOFF_CANDIDATE_MANIFEST",
                         "Manifest declares READY_FOR_REVIEW_NOT_YET_MERGED; claim matrix was not imported.")
        add_figure_asset(con, "chart_dft_mace47_basin_sensitivity_20261023", "PROVENANCE_MANIFEST",
                         dft_manifest.relative_to(REPO).as_posix(), "PRESENT", "DFT_MACE47_CANDIDATE_RELEASE_MANIFEST",
                         "Separate candidate bundle; its C1-C6 claim matrix is not in this local archive copy.", dft_manifest_pair[0])

        # Register every chart registry data/provenance row with an artifact or
        # a clearly labelled external pointer; never join by basename alone if
        # it is ambiguous.
        for label, path, source_id in (("core", CORE_DB, core_id), ("plot", PLOT_DB, plot_id)):
            src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            src.row_factory = sqlite3.Row
            for row in src.execute("SELECT * FROM chart_data_files"):
                chart_id, file_id, local_path = row["chart_id"], row["file_id"], row["local_path"]
                target = None
                resolved, exact = register_repo_source_if_present(local_path, "MigrationBench chart data source", f"Path from {label} chart_data_files row {chart_id}/{file_id}.")
                if exact:
                    target = (exact[0], "PAPER-SOURCE:LOCAL-MAIN:" + resolved.relative_to(REPO).as_posix())
                else:
                    found = find_archive_member(chart_id, Path(local_path).name)
                if not exact and found:
                    archive_key = "PAPER-ARCHIVE-20261028:" + found[0].relative_to(ARCHIVE).as_posix()
                    add_external_ref(con, source_id, "chart_data_file_archive_copy", f"{chart_id}|{file_id}",
                                     "ARTIFACT", archive_key, "ARCHIVE_COPY_CANDIDATE",
                                     f"Archive copy found by chart ID and basename only; original registry path={local_path}; no source/copy hash comparison was possible.")
                target_type, target_key, relation = ("ARTIFACT", target[1], "REGISTRY_DATA_FILE") if target else ("EXTERNAL_POINTER", local_path or f"{chart_id}/{file_id}", "REGISTRY_DATA_POINTER_ONLY")
                add_external_ref(con, source_id, "chart_data_file", f"{chart_id}|{file_id}", target_type, target_key,
                                 relation, f"display_name={row['display_name']}; file_exists={row['file_exists']}; registry={label}")
                if target:
                    add_figure_asset(con, chart_id, "DERIVED_DATA", local_path, "PRESENT", "REGISTRY_DATA_PATH_RESOLVED",
                                     f"Registry={label}; file_id={file_id}; resolved to a hashed local artifact.", target[0])
                else:
                    add_figure_asset(con, chart_id, "DERIVED_DATA", local_path or f"{chart_id}/{file_id}", "EXTERNAL_POINTER",
                                     "REGISTRY_DATA_PATH_UNRESOLVED", f"Registry={label}; file_exists={row['file_exists']}; archived same-basename copies, if any, are linked separately and are not asserted to be byte-identical.")

            for row in src.execute("SELECT * FROM chart_provenance"):
                chart_id, manifest = row["chart_id"], row["manifest_path"]
                target = None
                resolved, exact = register_repo_source_if_present(manifest, "MigrationBench chart provenance manifest", f"Path from {label} chart_provenance row {chart_id}.") if manifest else (None, None)
                if exact:
                    target = (exact[0], "PAPER-SOURCE:LOCAL-MAIN:" + resolved.relative_to(REPO).as_posix())
                else:
                    found = find_archive_member(chart_id, Path(manifest).name) if manifest else None
                if not exact and found:
                    archive_key = "PAPER-ARCHIVE-20261028:" + found[0].relative_to(ARCHIVE).as_posix()
                    add_external_ref(con, source_id, "chart_provenance_archive_copy", chart_id,
                                     "ARTIFACT", archive_key, "ARCHIVE_COPY_CANDIDATE",
                                     f"Archive member found by chart ID and basename only; original manifest path={manifest}; source/copy SHA-256 equality was not asserted.")
                target_type, target_key, relation = ("ARTIFACT", target[1], "REGISTRY_MANIFEST") if target else ("EXTERNAL_POINTER", manifest or f"{chart_id}/manifest", "REGISTRY_MANIFEST_POINTER_ONLY")
                add_external_ref(con, source_id, "chart_provenance", chart_id, target_type, target_key, relation,
                                 f"registry={label}; provenance_status={row['provenance_status']}; manifest_exists={row['manifest_exists']}; output_sha256={row['output_sha256']}")
                if target:
                    add_figure_asset(con, chart_id, "PROVENANCE_MANIFEST", manifest, "PRESENT", "REGISTRY_MANIFEST_RESOLVED",
                                     f"Registry={label}; provenance_status={row['provenance_status']}.", target[0])
                elif manifest:
                    add_figure_asset(con, chart_id, "PROVENANCE_MANIFEST", manifest, "MISSING", "REGISTRY_MANIFEST_POINTER_ONLY",
                                     f"Registry={label}; source manifest path is not present in the unified worktree. Any matching archive file is a separate copy; no byte identity is asserted.")
            src.close()

        # A dated archive catalog row is not an Overleaf/manuscript release.
        manifest_path = ARCHIVE / "index/FIGURE_REPRODUCTION_INDEX.tsv"
        manifest_artifact_id = rel_artifacts[manifest_path.relative_to(REPO).as_posix()][0]
        con.execute(
            """INSERT OR IGNORE INTO paper_release
               (release_id,project_id,release_label,released_utc,status,manifest_artifact_id,notes)
               VALUES(?,'MIGRATIONBENCH',?,NULL,'DRAFT',?,?)""",
            (RELEASE_ID, "2026-10-28 local figure/reproduction archive snapshot",
             manifest_artifact_id,
             "Catalog scope only: not a manuscript or Overleaf release. No synchronized manuscript version was evidenced. Each chart retains its own status; unselected candidates and missing scripts remain explicit."),
        )

        # The static-639 matrix contains nine explicit claims. The MD FIG-004
        # A-G rows are panel descriptions/gates, not manuscript claims; they
        # remain figure assets and are not inflated into paper_claim rows.
        claim_matrix_specs = [
            ("chart_static_639_family_grouped_20261014", ARCHIVE / "provenance/chart_static_639_family_grouped_20261014/CLAIM_EVIDENCE_MATRIX.tsv", "static"),
        ]
        con.execute(
            """DELETE FROM claim_evidence
               WHERE claim_id IN ('FS639-001','FS639-002','FS639-003','FS639-004','FS639-005',
                                  'FS639-006','FS639-007','FS639-008','FS639-009')
                 AND rowid NOT IN (
                   SELECT MIN(rowid) FROM claim_evidence
                   WHERE claim_id IN ('FS639-001','FS639-002','FS639-003','FS639-004','FS639-005',
                                      'FS639-006','FS639-007','FS639-008','FS639-009')
                   GROUP BY claim_id,artifact_id,run_id,evidence_role
                 )"""
        )
        for figure_id, matrix_path, flavor in claim_matrix_specs:
            matrix_rel = matrix_path.relative_to(REPO).as_posix()
            matrix_artifact_id = rel_artifacts[matrix_rel][0]
            matrix_source_key = ("paper_claim_matrix_static_639_20261014" if flavor == "static" else "paper_claim_matrix_md_fig004_20261016")
            matrix_source_id = archive_sources[matrix_source_key]
            rows = tsv(matrix_path)
            for row in rows:
                if flavor == "static":
                    local_id = row["claim_id"]
                    claim_id = local_id
                    label = row["claim"]
                    source_status = row["status"]
                    target = row["manuscript_target"]
                    original_action = row["publication_action"]
                    source_refs = [x.strip() for x in row.get("evidence_artifact", "").split(";") if x.strip()]
                    job_ref = row.get("job_or_run", "")
                    note_lines = [f"verbatim matrix status={source_status}; source publication_action={original_action}; source job_or_run={job_ref}",
                                  "Source path list is copied verbatim below; absent paths are not treated as present or as verified workflow_run IDs."]
                else:
                    local_id = row["panel"]
                    raise RuntimeError("Only static-639 claim rows are imported by this script")
                note_lines.append("source claim-matrix path=" + matrix_rel)
                con.execute(
                    """INSERT OR IGNORE INTO paper_claim
                       (claim_id,release_id,section_or_panel,claim_text,scientific_status,publication_action,notes)
                       VALUES(?,?,?,?,?,?,?)""",
                    (claim_id, static_release_id, target, label, "RECONSTRUCTION" if source_status == "RECONSTRUCTION" else "PROVENANCE_PARTIAL",
                     action_status(original_action, source_status), "\n".join(note_lines)),
                )
                expected_claim = con.execute("SELECT claim_text,section_or_panel FROM paper_claim WHERE claim_id=?", (claim_id,)).fetchone()
                if expected_claim != (label, target):
                    raise RuntimeError(f"Existing paper_claim differs from matrix: {claim_id}")
                add_external_ref(con, matrix_source_id, "claim", local_id, "PAPER_CLAIM", claim_id,
                                 "SOURCE_CLAIM_MATRIX_ROW", "Imported verbatim from the dated matrix; not joined to an unresolved run/job ID.")
                con.execute(
                    """INSERT OR IGNORE INTO paper_claim_source_observation
                       (source_id,source_claim_key,claim_id,source_status,source_publication_action,
                        source_job_or_run,source_target,source_evidence_refs,observed_utc)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (matrix_source_id, local_id, claim_id, source_status, original_action, job_ref, target,
                     json.dumps(source_refs, ensure_ascii=False), now),
                )
                add_claim_evidence(
                    con, claim_id, matrix_artifact_id, "SOURCE_CLAIM_MATRIX", "UNKNOWN",
                    "The archived matrix is claim context; the catalog has not independently assessed scientific support.",
                )
                resolved_refs: list[str] = []
                unresolved_refs: list[str] = []
                for source_ref in source_refs:
                    _, basename = os.path.split(source_ref)
                    resolved: tuple[int, Path] | None = None
                    exact_path = (STATIC_639_ROOT / source_ref).resolve(strict=False)
                    if exact_path and exact_path.is_file():
                        item = register_static639_source(source_ref, "MigrationBench static-screen claim evidence", f"Exact evidence pointer from {local_id}.")
                        if item[1]:
                            resolved = (item[1][0], item[0])
                    if resolved:
                        aid, physical = resolved
                        resolved_refs.append(source_ref + " -> " + physical.relative_to(REPO).as_posix())
                        add_claim_evidence(
                            con, claim_id, aid, "MATRIX_LISTED_ARTIFACT", "UNKNOWN",
                            "Source matrix listed " + source_ref + "; import verifies path and hash only, not scientific support relation.",
                        )
                    else:
                        unresolved_refs.append(source_ref)
                if resolved_refs:
                    con.execute("UPDATE paper_claim SET notes=notes || ? WHERE claim_id=? AND instr(notes,?)=0",
                                ("\nresolved listed artifacts:\n- " + "\n- ".join(resolved_refs), claim_id,
                                 "resolved listed artifacts:"))
                if unresolved_refs:
                    con.execute("UPDATE paper_claim SET notes=notes || ? WHERE claim_id=? AND instr(notes,?)=0",
                                ("\nunresolved matrix-listed paths/pointers:\n- " + "\n- ".join(unresolved_refs), claim_id,
                                 "unresolved matrix-listed paths/pointers:"))

                con.execute(
                    """UPDATE claim_evidence SET support_status='UNKNOWN',notes=?
                       WHERE claim_id=? AND evidence_role='SOURCE_CLAIM_MATRIX'""",
                    ("The source matrix is claim context; the catalog has not independently assessed support.", claim_id),
                )
                con.execute(
                    """UPDATE claim_evidence SET support_status='UNKNOWN',notes=?
                       WHERE claim_id=? AND evidence_role='MATRIX_LISTED_ARTIFACT'""",
                    ("Source matrix listed this artifact; import verifies path and hash only, not its scientific support relation.", claim_id),
                )

        # The matrix explicitly labels FS639-009 as the figure package
        # reconstruction. Do not infer panel-to-claim assignments for FS639-001…008.
        matrix_source_id = archive_sources["paper_claim_matrix_static_639_20261014"]
        con.execute(
            """INSERT OR IGNORE INTO paper_claim_figure
               (claim_id,figure_id,source_id,source_claim_key,relationship,notes)
               VALUES('FS639-009','chart_static_639_family_grouped_20261014',?,'FS639-009','RECONSTRUCTION_OF',?)""",
            (matrix_source_id,
             "Explicit source-matrix link: FS639-009 identifies the four-panel package as a Plot Atlas reconstruction. This figure-level link does not map individual panels."),
        )
        add_external_ref(con, matrix_source_id, "claim_figure_relation", "FS639-009|chart_static_639_family_grouped_20261014",
                         "PAPER_FIGURE", "chart_static_639_family_grouped_20261014",
                         "SOURCE_MATRIX_FIGURE_RELATION", "Source claim key FS639-009; figure-level only, no panel map inferred.")

        if con.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Foreign-key check failed after paper archive import")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    with sqlite3.connect(DB) as check:
        print("migration_009=" + str(check.execute("SELECT count(*) FROM schema_migration WHERE version=9").fetchone()[0]))
        print("migration_010=" + str(check.execute("SELECT count(*) FROM schema_migration WHERE version=10").fetchone()[0]))
        print("paper_figures=" + str(check.execute("SELECT count(*) FROM paper_figure").fetchone()[0]))
        print("figure_asset_links=" + str(check.execute("SELECT count(*) FROM paper_figure_asset").fetchone()[0]))
        print("paper_claims=" + str(check.execute("SELECT count(*) FROM paper_claim").fetchone()[0]))
        print("claim_evidence_links=" + str(check.execute("SELECT count(*) FROM claim_evidence").fetchone()[0]))
        print("claim_figure_links=" + str(check.execute("SELECT count(*) FROM paper_claim_figure").fetchone()[0]))
        print("registry_entity_refs=" + str(check.execute("SELECT count(*) FROM registry_entity_ref").fetchone()[0]))
        print("foreign_key_violations=" + str(len(check.execute("PRAGMA foreign_key_check").fetchall())))


if __name__ == "__main__":
    main()
