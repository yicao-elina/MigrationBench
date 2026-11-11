#!/usr/bin/env python3
"""Build a compact, provenance-preserving Google Drive archive for MigrationBench.

The archive is additive and selective: SQL registries, paper-facing documents,
registered figures/tables, rebuild sources, and compact campaign evidence are
copied. Large raw trajectories, model checkpoints, and scratch outputs are
represented by source pointers and are not duplicated into Drive.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path


SOURCE_ROOT = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench")
DRIVE_ROOT = Path(
    "/Users/alina/Library/CloudStorage/GoogleDrive-yicao.alina@gmail.com/My Drive/26 Fall/MigrationBench"
)
ARCHIVE_NAME = "migrationbench_repro_archive_20261028"
ARCHIVE_ROOT = Path(os.environ.get("MIGRATIONBENCH_ARCHIVE_ROOT", str(DRIVE_ROOT / ARCHIVE_NAME)))
REGISTRY_DIR = SOURCE_ROOT / "analysis/migrationbench_asset_registry_20261025"
REGISTRY_DB = REGISTRY_DIR / "migrationbench_asset_registry.sqlite"

DOC_EXTS = {
    ".csv",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".sh",
    ".slurm",
    ".sql",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".xlsx",
    ".xml",
    ".yaml",
    ".yml",
}
FIG_EXTS = {".pdf", ".png", ".svg"}
SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    "checkpoints",
    "models",
    "model",
    "raw",
    "remote_results",
    "snapshots",
    "trajectories",
    "trajectory",
}
MAX_COMPACT_FILE_BYTES = 25 * 1024 * 1024
COPY_TIMEOUT_SECONDS = 15
BLOCKED_SOURCE_RELS = {
    # This macOS file-provider placeholder blocks low-level reads in the
    # current workspace; retain a pointer rather than creating a blank copy.
    "paper_evidence/templates/section_bidirectional_neb.tex",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_rel(path: Path) -> str:
    return path.relative_to(SOURCE_ROOT).as_posix()


def main() -> None:
    if not SOURCE_ROOT.is_dir():
        raise SystemExit(f"Missing source root: {SOURCE_ROOT}")
    if not DRIVE_ROOT.is_dir():
        raise SystemExit(f"Missing Google Drive directory: {DRIVE_ROOT}")
    if ARCHIVE_ROOT.exists() and any(ARCHIVE_ROOT.iterdir()):
        raise SystemExit(f"Archive already exists and is non-empty: {ARCHIVE_ROOT}")

    for name in (
        "00_INDEX_AND_SQL",
        "01_PAPER_EVIDENCE",
        "02_STAGE_DOCUMENTS",
        "03_FIGURES",
        "04_TABLES_AND_DERIVED_DATA",
        "05_REBUILD_SOURCES",
        "06_SOURCE_POINTERS",
    ):
        (ARCHIVE_ROOT / name).mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, object]] = []
    omitted: list[dict[str, object]] = []
    source_to_archive: dict[str, str] = {}

    def copy_file(
        source: Path,
        destination: Path,
        role: str,
        scientific_status: str = "",
        note: str = "",
        allow_missing: bool = False,
    ) -> str | None:
        source = source.resolve()
        source_key = str(source)
        if not source.is_file():
            row = {
                "source_rel": safe_rel(source) if source.is_relative_to(SOURCE_ROOT) else source_key,
                "role": role,
                "reason": "missing_source",
                "scientific_status": scientific_status,
                "note": note,
            }
            omitted.append(row)
            if allow_missing:
                return None
            return None
        if source.stat().st_size > MAX_COMPACT_FILE_BYTES:
            omitted.append(
                {
                    "source_rel": safe_rel(source),
                    "role": role,
                    "reason": "over_size_limit",
                    "scientific_status": scientific_status,
                    "note": f"{source.stat().st_size} bytes; pointer retained",
                }
            )
            return None

        destination = destination.relative_to(ARCHIVE_ROOT) if destination.is_absolute() else destination
        destination_path = ARCHIVE_ROOT / destination
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists():
            # A destination created by this builder is safe to reuse on a
            # repeated source reference; do not reread a possibly blocked
            # source file merely to compare it.
            pass
        else:
            try:
                subprocess.run(
                    ["/bin/cp", str(source), str(destination_path)],
                    check=True,
                    timeout=COPY_TIMEOUT_SECONDS,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
                destination_path.unlink(missing_ok=True)
                omitted.append(
                    {
                        "source_rel": safe_rel(source),
                        "role": role,
                        "reason": "filesystem_copy_failed_or_timed_out",
                        "scientific_status": scientific_status,
                        "note": str(exc),
                    }
                )
                return None
        archive_rel = destination.as_posix()
        source_to_archive.setdefault(source_key, archive_rel)
        copied.append(
            {
                "archive_rel": archive_rel,
                "source_rel": safe_rel(source),
                "role": role,
                "size_bytes": destination_path.stat().st_size,
                "sha256": sha256(destination_path),
                "scientific_status": scientific_status,
                "note": note,
            }
        )
        return archive_rel

    def copy_compact_tree(source_root: Path, destination_root: Path, role: str) -> None:
        if source_root.is_file():
            copy_file(source_root, destination_root / source_root.name, role, allow_missing=True)
            return
        if not source_root.is_dir():
            omitted.append(
                {
                    "source_rel": safe_rel(source_root) if source_root.is_relative_to(SOURCE_ROOT) else str(source_root),
                    "role": role,
                    "reason": "missing_root",
                    "scientific_status": "",
                    "note": "",
                }
            )
            return
        for path in sorted(source_root.rglob("*")):
            if not path.is_file():
                continue
            if path.is_relative_to(SOURCE_ROOT) and safe_rel(path) in BLOCKED_SOURCE_RELS:
                omitted.append(
                    {
                        "source_rel": safe_rel(path),
                        "role": role,
                        "reason": "filesystem_read_blocked",
                        "scientific_status": "",
                        "note": "Source path retained; no placeholder file created.",
                    }
                )
                continue
            if any(part in SKIP_DIR_NAMES for part in path.relative_to(source_root).parts):
                continue
            if path.suffix.lower() not in DOC_EXTS:
                continue
            if path.stat().st_size > MAX_COMPACT_FILE_BYTES:
                omitted.append(
                    {
                        "source_rel": safe_rel(path),
                        "role": role,
                        "reason": "over_size_limit",
                        "scientific_status": "",
                        "note": f"{path.stat().st_size} bytes",
                    }
                )
                continue
            rel = path.relative_to(source_root)
            copy_file(path, destination_root / rel, role, allow_missing=True)

    with REGISTRY_DB.open("rb"):
        pass
    conn = sqlite3.connect(REGISTRY_DB)
    conn.row_factory = sqlite3.Row
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"Registry integrity check failed: {integrity}")

    # The registry itself is the SQL source of truth. Back it up through the
    # SQLite API (avoids macOS file-provider stalls on individual exports),
    # then preserve the human-facing SQL documents when they can be read.
    registry_archive_root = ARCHIVE_ROOT / "00_INDEX_AND_SQL" / "asset_registry_20261025"
    registry_archive_root.mkdir(parents=True, exist_ok=True)
    archive_db_path = registry_archive_root / "migrationbench_asset_registry.sqlite"
    archive_db_conn = sqlite3.connect(archive_db_path)
    conn.backup(archive_db_conn)
    archive_db_conn.close()
    registry_key_files = [
        "README.md",
        "GOOGLE_SHEETS_LINK.md",
        "schema.sql",
        "query_examples.sql",
        "registry_app.py",
        "registry_data.json",
        "data_quality_report.json",
        "migrationbench_asset_registry.xlsx",
    ]
    for relative_name in registry_key_files:
        copy_file(
            REGISTRY_DIR / relative_name,
            registry_archive_root / relative_name,
            "sql_registry",
            allow_missing=True,
        )
    for path in sorted((REGISTRY_DIR / "updates").glob("*.sql")):
        copy_file(path, registry_archive_root / "updates" / path.name, "sql_registry_update", allow_missing=True)
    secondary_db = SOURCE_ROOT / "data/collaborator_dft_neb/migrationbench_dft_neb.sqlite"
    if secondary_db.is_file():
        secondary_archive_root = ARCHIVE_ROOT / "00_INDEX_AND_SQL" / "collaborator_dft_neb"
        secondary_archive_root.mkdir(parents=True, exist_ok=True)
        secondary_conn = sqlite3.connect(secondary_db)
        secondary_copy = sqlite3.connect(secondary_archive_root / "migrationbench_dft_neb.sqlite")
        secondary_conn.backup(secondary_copy)
        secondary_copy.close()
        secondary_conn.close()
    else:
        omitted.append(
            {
                "source_rel": safe_rel(secondary_db),
                "role": "sql_registry_secondary",
                "reason": "missing_source",
                "scientific_status": "",
                "note": "",
            }
        )
    copy_compact_tree(SOURCE_ROOT / "data/collaborator_dft_neb", ARCHIVE_ROOT / "00_INDEX_AND_SQL" / "collaborator_dft_neb", "sql_registry_secondary")
    copy_compact_tree(SOURCE_ROOT / "paper_evidence", ARCHIVE_ROOT / "01_PAPER_EVIDENCE", "paper_evidence")

    campaign_rows = conn.execute(
        "SELECT campaign_id, title, campaign_path, status FROM campaigns ORDER BY campaign_id"
    ).fetchall()
    campaign_by_id = {row["campaign_id"]: row for row in campaign_rows}

    # Export every registry table/view directly from SQLite so the archive is
    # portable even when one source CSV is not materialized by macOS.
    exports_root = ARCHIVE_ROOT / "00_INDEX_AND_SQL" / "registry_exports"
    exports_root.mkdir(parents=True, exist_ok=True)
    object_rows = conn.execute(
        "SELECT name, type, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY type, name"
    ).fetchall()
    schema_lines: list[str] = ["PRAGMA foreign_keys = ON;", ""]
    for object_row in object_rows:
        if object_row["sql"]:
            schema_lines.append(object_row["sql"].rstrip(";") + ";")
            schema_lines.append("")
        if object_row["type"] not in {"table", "view"}:
            continue
        columns = [item["name"] for item in conn.execute(f'PRAGMA table_info("{object_row["name"]}")').fetchall()]
        data = conn.execute(f'SELECT * FROM "{object_row["name"]}"').fetchall()
        export_path = exports_root / f"{object_row['name']}.csv"
        with export_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for data_row in data:
                writer.writerow([value.decode("utf-8", "replace") if isinstance(value, bytes) else value for value in data_row])
    (ARCHIVE_ROOT / "00_INDEX_AND_SQL" / "ARCHIVE_SCHEMA_FROM_SQLITE.sql").write_text(
        "\n".join(schema_lines) + "\n", encoding="utf-8"
    )
    # The registry has a broad `analysis` locator for Scratch. Use its
    # evidence package instead of recursively copying every analysis folder.
    campaign_roots: dict[str, Path] = {}
    for row in campaign_rows:
        campaign_path = row["campaign_path"]
        if not campaign_path:
            continue
        candidate = SOURCE_ROOT / campaign_path
        if candidate == SOURCE_ROOT / "analysis":
            candidate = SOURCE_ROOT / "analysis/from_scratch_failure_attribution_20261011"
        campaign_roots[row["campaign_id"]] = candidate

    evidence_tokens = (
        "README", "MANIFEST", "REPORT", "CAPTION", "REGISTRY", "MATRIX", "STATUS",
        "FAILURE", "GAP", "AUDIT", "DELIVERY", "TODO", "WORKFLOW", "DISCUSSION",
        "RESULT", "PROVENANCE", "INDEX", "ENVIRONMENT", "SUMMARY", "RELEASE", "CLAIM",
    )
    for campaign_id, root in sorted(campaign_roots.items()):
        if os.environ.get("MIGRATIONBENCH_INCLUDE_CAMPAIGN_DOCS") != "1":
            continue
        destination_root = ARCHIVE_ROOT / "02_STAGE_DOCUMENTS" / "campaigns" / campaign_id
        if root.is_file():
            copy_file(root, destination_root / root.name, f"campaign_documents:{campaign_id}", allow_missing=True)
            continue
        if not root.is_dir():
            omitted.append(
                {
                    "source_rel": safe_rel(root) if root.is_relative_to(SOURCE_ROOT) else str(root),
                    "role": f"campaign_documents:{campaign_id}",
                    "reason": "missing_root",
                    "scientific_status": campaign_by_id[campaign_id]["status"] if campaign_id in campaign_by_id else "",
                    "note": "",
                }
            )
            continue
        # Keep the campaign-level control documents and top-level compact
        # tables. Deep raw/result trees remain represented by the registry,
        # chart data files, and source pointers below.
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in DOC_EXTS:
                continue
            upper_name = path.name.upper()
            if any(token in upper_name for token in evidence_tokens):
                copy_file(path, destination_root / path.name, f"campaign_documents:{campaign_id}", allow_missing=True)
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix.lower() in {".py", ".sh", ".slurm", ".mjs", ".js"}:
                copy_file(
                    path,
                    ARCHIVE_ROOT / "05_REBUILD_SOURCES" / "campaigns" / campaign_id / path.name,
                    f"rebuild_source:{campaign_id}",
                    allow_missing=True,
                )

    # Additional article-facing packages created after the registry snapshot.
    supplemental_roots = [
        "analysis/canonical_model_matrix_20261016",
        "analysis/cr_interlayer_sliding_grouped_20261025",
        "analysis/cr_sb2te3_nafsi_style_20261026",
        "analysis/figure_evidence_audit_20261008",
        "analysis/md_1ns_updated_publication_figure_20261025",
        "analysis/migrationbench_data_universe_20261026",
        "analysis/migrationbench_training_design_figure_20261024",
        "analysis/migrationbench_typical10_render_20261027",
        "analysis/migrationbench_universe_publication_20261026",
        "analysis/model_registry_unified_20261017",
        "analysis/paper_provenance_audit_20261013",
        "analysis/rmse_train_test_20261026",
    ]
    for source_rel in supplemental_roots:
        if os.environ.get("MIGRATIONBENCH_INCLUDE_SUPPLEMENTAL") != "1":
            continue
        root = SOURCE_ROOT / source_rel
        label = Path(source_rel).name
        if not root.is_dir():
            continue
        shallow_paths = list(root.iterdir())
        for child in sorted(root.iterdir()):
            if child.is_dir() and child.name not in SKIP_DIR_NAMES:
                shallow_paths.extend(child.iterdir())
        for path in sorted(shallow_paths):
            if path.is_dir():
                continue
            if path.suffix.lower() in DOC_EXTS:
                upper_name = path.name.upper()
                if any(token in upper_name for token in evidence_tokens):
                    copy_file(
                        path,
                        ARCHIVE_ROOT / "02_STAGE_DOCUMENTS" / "supplemental" / label / path.relative_to(root),
                        f"supplemental_documents:{label}",
                        allow_missing=True,
                    )
        for path in sorted(shallow_paths):
            if not path.is_file() or path.suffix.lower() not in FIG_EXTS:
                continue
            relative_parts = {part.lower() for part in path.relative_to(root).parts}
            if path.suffix.lower() == ".png" and any(
                token in part
                for part in relative_parts
                for token in ("preview", "draft", "ideal", "variant", "packed", "rgba")
            ):
                continue
            copy_file(
                path,
                ARCHIVE_ROOT / "03_FIGURES" / "supplemental" / label / path.relative_to(root),
                f"supplemental_figure:{label}",
                allow_missing=True,
            )

    chart_rows = conn.execute(
        """
        SELECT c.chart_id, c.title, c.chart_kind, c.chart_status, c.asset_path,
               c.asset_exists, c.campaign_id, c.notes,
               cp.manifest_path, cp.processing_command, cp.provenance_status
        FROM charts AS c
        LEFT JOIN chart_provenance AS cp ON cp.chart_id = c.chart_id
        ORDER BY c.chart_id
        """
    ).fetchall()
    data_rows = conn.execute(
        "SELECT chart_id, file_id, display_name, local_path, file_format, file_exists, notes FROM chart_data_files ORDER BY chart_id, file_id"
    ).fetchall()
    data_by_chart: dict[str, list[sqlite3.Row]] = {}
    for row in data_rows:
        data_by_chart.setdefault(row["chart_id"], []).append(row)

    figure_index: list[dict[str, object]] = []
    for row in chart_rows:
        chart_id = row["chart_id"]
        figure_archives: list[str] = []
        asset_path = row["asset_path"] or ""
        if asset_path:
            source = SOURCE_ROOT / asset_path
            if source.is_file():
                siblings = [source]
                for ext in FIG_EXTS:
                    sibling = source.with_suffix(ext)
                    if sibling.is_file() and sibling not in siblings:
                        siblings.append(sibling)
                for figure_source in siblings:
                    archive_rel = copy_file(
                        figure_source,
                        ARCHIVE_ROOT / "03_FIGURES" / chart_id / figure_source.name,
                        f"registered_figure:{chart_id}",
                        row["chart_status"] or "",
                        row["notes"] or "",
                        allow_missing=True,
                    )
                    if archive_rel:
                        figure_archives.append(archive_rel)
            else:
                omitted.append(
                    {
                        "source_rel": asset_path,
                        "role": f"registered_figure:{chart_id}",
                        "reason": "missing_source",
                        "scientific_status": row["chart_status"] or "",
                        "note": row["title"],
                    }
                )

        data_archives: list[str] = []
        for data_row in data_by_chart.get(chart_id, []):
            source = SOURCE_ROOT / data_row["local_path"]
            existing = source_to_archive.get(str(source.resolve())) if source.exists() else None
            archive_rel = existing
            if archive_rel is None:
                archive_rel = copy_file(
                    source,
                    ARCHIVE_ROOT / "04_TABLES_AND_DERIVED_DATA" / chart_id / Path(data_row["local_path"]).name,
                    f"registered_chart_data:{chart_id}",
                    row["chart_status"] or "",
                    data_row["notes"] or "",
                    allow_missing=True,
                )
            if archive_rel:
                data_archives.append(archive_rel)

        manifest_archive = None
        manifest_path = row["manifest_path"] or ""
        if manifest_path:
            manifest_archive = source_to_archive.get(str((SOURCE_ROOT / manifest_path).resolve()))
            if manifest_archive is None:
                manifest_archive = copy_file(
                    SOURCE_ROOT / manifest_path,
                    ARCHIVE_ROOT / "02_STAGE_DOCUMENTS" / "chart_manifests" / chart_id / Path(manifest_path).name,
                    f"chart_manifest:{chart_id}",
                    row["chart_status"] or "",
                    allow_missing=True,
                )

        figure_index.append(
            {
                "chart_id": chart_id,
                "title": row["title"],
                "chart_status": row["chart_status"],
                "provenance_status": row["provenance_status"] or "",
                "campaign_id": row["campaign_id"] or "",
                "figure_archive_paths": ";".join(figure_archives),
                "data_archive_paths": ";".join(data_archives),
                "manifest_archive_path": manifest_archive or "",
                "source_asset_path": asset_path,
                "source_manifest_path": manifest_path,
                "processing_command": row["processing_command"] or "",
                "notes": row["notes"] or "",
            }
        )

    asset_rows = conn.execute(
        "SELECT asset_id, asset_type, display_name, local_path, campaign_id, existence_at_build, scientific_status, notes FROM assets ORDER BY asset_id"
    ).fetchall()
    source_pointers: list[dict[str, object]] = []
    for row in asset_rows:
        source = SOURCE_ROOT / row["local_path"]
        archive_matches = [
            item["archive_rel"]
            for item in copied
            if item["source_rel"] == row["local_path"] or str(item["source_rel"]).startswith(f"{row['local_path'].rstrip('/')}/")
        ]
        source_pointers.append(
            {
                "entity_type": "asset",
                "entity_id": row["asset_id"],
                "display_name": row["display_name"],
                "source_rel": row["local_path"],
                "source_exists": int(source.exists()),
                "scientific_status": row["scientific_status"],
                "campaign_id": row["campaign_id"] or "",
                "archive_locations": ";".join(sorted(set(archive_matches))),
                "copy_mode": "selective_compact" if source.is_dir() else ("copied" if archive_matches else "pointer_only"),
                "notes": row["notes"],
            }
        )
    for row in campaign_rows:
        source = SOURCE_ROOT / (row["campaign_path"] or "")
        source_pointers.append(
            {
                "entity_type": "campaign",
                "entity_id": row["campaign_id"],
                "display_name": row["title"],
                "source_rel": row["campaign_path"] or "",
                "source_exists": int(source.exists()),
                "scientific_status": row["status"],
                "campaign_id": row["campaign_id"],
                "archive_locations": f"02_STAGE_DOCUMENTS/campaigns/{row['campaign_id']}",
                "copy_mode": "selective_compact",
                "notes": "Raw trajectories/checkpoints are not duplicated; see archive manifest and source path.",
            }
        )

    # Record the repository state used to build the archive.
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE_ROOT, text=True).strip()
        git_status = "Not scanned: full repository status is intentionally omitted because the analysis tree contains file-provider placeholders."
    except (OSError, subprocess.CalledProcessError) as exc:
        git_commit = "unavailable"
        git_status = f"unavailable: {exc}"

    write_tsv(
        ARCHIVE_ROOT / "06_SOURCE_POINTERS" / "ARCHIVE_MANIFEST.tsv",
        copied,
        ["archive_rel", "source_rel", "role", "size_bytes", "sha256", "scientific_status", "note"],
    )
    write_tsv(
        ARCHIVE_ROOT / "06_SOURCE_POINTERS" / "OMITTED_OR_POINTER_ONLY.tsv",
        omitted,
        ["source_rel", "role", "reason", "scientific_status", "note"],
    )
    write_tsv(
        ARCHIVE_ROOT / "06_SOURCE_POINTERS" / "SOURCE_POINTERS.tsv",
        source_pointers,
        ["entity_type", "entity_id", "display_name", "source_rel", "source_exists", "scientific_status", "campaign_id", "archive_locations", "copy_mode", "notes"],
    )
    write_tsv(
        ARCHIVE_ROOT / "00_INDEX_AND_SQL" / "FIGURE_REPRODUCTION_INDEX.tsv",
        figure_index,
        ["chart_id", "title", "chart_status", "provenance_status", "campaign_id", "figure_archive_paths", "data_archive_paths", "manifest_archive_path", "source_asset_path", "source_manifest_path", "processing_command", "notes"],
    )

    checksums = ARCHIVE_ROOT / "06_SOURCE_POINTERS" / "CHECKSUMS.sha256"
    with checksums.open("w", encoding="utf-8") as handle:
        for item in sorted(copied, key=lambda row: str(row["archive_rel"])):
            handle.write(f"{item['sha256']}  {item['archive_rel']}\n")

    archive_readme = f"""# MigrationBench reproducibility archive — 2026-10-28

This is a compact, additive archive of the MigrationBench paper evidence and
figure/table rebuild inputs. It was built from:

- source repository: `{SOURCE_ROOT}`
- source commit: `{git_commit}`
- archive build time: `{datetime.now().astimezone().isoformat()}`
- SQL source of truth: `00_INDEX_AND_SQL/asset_registry_20261025/migrationbench_asset_registry.sqlite`

## How to navigate

- `00_INDEX_AND_SQL/` contains the portable SQLite registries, schemas, query examples, exports, and `FIGURE_REPRODUCTION_INDEX.tsv`.
- `01_PAPER_EVIDENCE/` contains the paper-facing candidate/release control plane.
- `02_STAGE_DOCUMENTS/` contains compact campaign and supplemental documents, manifests, captions, status reports, and tables that are not chart-linked.
- `03_FIGURES/` contains registered figure outputs and supplemental article-facing visual artifacts.
- `04_TABLES_AND_DERIVED_DATA/` contains chart-linked source tables when they were not already preserved under a campaign package.
- `05_REBUILD_SOURCES/` contains plotting and packaging scripts from the campaign packages.
- `06_SOURCE_POINTERS/` contains checksums, the complete copied-file manifest, omitted/pointer-only items, and original source locators.

## SQL reproduction

Run `00_INDEX_AND_SQL/asset_registry_20261025/query_examples.sql` against the
copied SQLite database. `00_INDEX_AND_SQL/FIGURE_REPRODUCTION_INDEX.tsv` maps
each registered `chart_id` to its archived figure, data files, manifest, and
processing command. `00_INDEX_AND_SQL/ARCHIVE_QUERIES.sql` contains archive-
oriented helper queries.

The registry deliberately preserves `VERIFIED`, `COMPLETE_WITH_CAVEATS`,
`FINITE_DIAGNOSTIC`, `PARTIAL`, `FAILURE_MODE`, `NEEDS_AUDIT`, and other
scientific states. A copied figure is not thereby promoted to publication
ready. Fixed-path NEB, self-relaxed endpoint-only proxies, warning-bearing DFT,
and failed MD remain separate.

## What is not duplicated

Raw trajectories, full model checkpoints, node-local scratch, and other large
inputs are not copied into Drive. Their original paths and statuses remain in
`06_SOURCE_POINTERS/SOURCE_POINTERS.tsv` and
`06_SOURCE_POINTERS/OMITTED_OR_POINTER_ONLY.tsv`; the archive manifest and
hashes preserve what was actually copied.

Repository status at build time:

```text
{git_status}
```
"""
    (ARCHIVE_ROOT / "README.md").write_text(archive_readme, encoding="utf-8")

    archive_queries = """-- MigrationBench archive helper queries.
-- Run against the copied SQLite database in
-- 00_INDEX_AND_SQL/asset_registry_20261025/.
-- The archive paths for each chart are in FIGURE_REPRODUCTION_INDEX.tsv.

-- Registered charts and their scientific/provenance status.
SELECT c.chart_id, c.title, c.chart_status,
       cp.provenance_status, c.campaign_id, c.asset_path,
       cp.manifest_path, cp.processing_command
FROM charts AS c
LEFT JOIN chart_provenance AS cp ON cp.chart_id = c.chart_id
ORDER BY c.chart_id;

-- Scientific statuses that must remain visible during review.
SELECT scientific_status, COUNT(*) AS n
FROM coverage
GROUP BY scientific_status
ORDER BY scientific_status;

-- Open gaps and recommended next actions.
SELECT priority, gap_id, entity_type, entity_id, gap_type,
       description, recommended_next_action, status
FROM gaps
WHERE status <> 'CLOSED'
ORDER BY CASE priority WHEN 'P0' THEN 0 WHEN 'P1' THEN 1 ELSE 2 END, gap_id;

-- Fixed versus self-relaxed NEB; intentionally not pooled.
SELECT task_id, system_id, campaign_id, model_id,
       expected_count, completed_count, scientific_status, result_summary
FROM coverage
WHERE task_id IN ('fixed_path_neb', 'self_relaxed_neb', 'mic_neb',
                  'interlayer_fixed', 'interlayer_self')
ORDER BY system_id, campaign_id, task_id, model_id;
"""
    (ARCHIVE_ROOT / "00_INDEX_AND_SQL" / "ARCHIVE_QUERIES.sql").write_text(archive_queries, encoding="utf-8")
    # Always provide executable archive-local SQL documents, even if the
    # original text files were not materialized by the macOS file provider.
    (registry_archive_root / "schema.sql").write_text(
        (ARCHIVE_ROOT / "00_INDEX_AND_SQL" / "ARCHIVE_SCHEMA_FROM_SQLITE.sql").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (registry_archive_root / "query_examples.sql").write_text(archive_queries, encoding="utf-8")

    write_tsv(
        ARCHIVE_ROOT / "02_STAGE_DOCUMENTS" / "CAMPAIGN_INDEX.tsv",
        [
            {
                "campaign_id": row["campaign_id"],
                "title": row["title"],
                "source_rel": row["campaign_path"] or "",
                "status": row["status"],
                "archive_note": "Campaign-level source pointer; registered chart/table evidence is indexed by chart_id.",
            }
            for row in campaign_rows
        ],
        ["campaign_id", "title", "source_rel", "status", "archive_note"],
    )

    summary = {
        "archive_root": str(ARCHIVE_ROOT),
        "source_root": str(SOURCE_ROOT),
        "source_commit": git_commit,
        "built_at": datetime.now().astimezone().isoformat(),
        "registry_integrity_check": integrity,
        "campaign_count": len(campaign_rows),
        "chart_count": len(chart_rows),
        "copied_file_count": len(copied),
        "omitted_or_pointer_only_count": len(omitted),
        "copied_bytes": sum(int(item["size_bytes"]) for item in copied),
        "scientific_statuses_preserved": sorted({str(item["scientific_status"]) for item in copied if item["scientific_status"]}),
    }
    (ARCHIVE_ROOT / "06_SOURCE_POINTERS" / "ARCHIVE_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Final inventory covers generated SQL exports and indexes as well as
    # files copied from the source tree.
    final_inventory: list[dict[str, object]] = []
    for path in sorted(ARCHIVE_ROOT.rglob("*")):
        if not path.is_file():
            continue
        final_inventory.append(
            {
                "archive_rel": path.relative_to(ARCHIVE_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_tsv(
        ARCHIVE_ROOT / "06_SOURCE_POINTERS" / "FINAL_FILE_INVENTORY.tsv",
        final_inventory,
        ["archive_rel", "size_bytes", "sha256"],
    )
    with (ARCHIVE_ROOT / "06_SOURCE_POINTERS" / "FINAL_CHECKSUMS.sha256").open("w", encoding="utf-8") as handle:
        for item in final_inventory:
            handle.write(f"{item['sha256']}  {item['archive_rel']}\n")
    summary["final_archive_file_count"] = len(final_inventory)
    summary["final_archive_bytes"] = sum(int(item["size_bytes"]) for item in final_inventory)
    (ARCHIVE_ROOT / "06_SOURCE_POINTERS" / "ARCHIVE_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary_rel = "06_SOURCE_POINTERS/ARCHIVE_SUMMARY.json"
    summary_path = ARCHIVE_ROOT / summary_rel
    updated_inventory: list[dict[str, object]] = []
    for item in final_inventory:
        if item["archive_rel"] == summary_rel:
            item = {
                "archive_rel": summary_rel,
                "size_bytes": summary_path.stat().st_size,
                "sha256": sha256(summary_path),
            }
        updated_inventory.append(item)
    write_tsv(
        ARCHIVE_ROOT / "06_SOURCE_POINTERS" / "FINAL_FILE_INVENTORY.tsv",
        updated_inventory,
        ["archive_rel", "size_bytes", "sha256"],
    )
    with (ARCHIVE_ROOT / "06_SOURCE_POINTERS" / "FINAL_CHECKSUMS.sha256").open("w", encoding="utf-8") as handle:
        for item in updated_inventory:
            handle.write(f"{item['sha256']}  {item['archive_rel']}\n")

    conn.close()
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
