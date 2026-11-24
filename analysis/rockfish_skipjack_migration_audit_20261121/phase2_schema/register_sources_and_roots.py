#!/usr/bin/env python3
"""Register the existing provenance sources and known storage roots.

This is an additive, idempotent loader for provenance_catalog.sqlite. It does
not modify any source registry or contact Rockfish.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDIT_DIR = HERE.parent
REPO_ROOT = HERE.parents[2]
DB_PATH = AUDIT_DIR / "provenance_catalog.sqlite"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


SOURCES = [
    {
        "source_key": "migrationbench_asset_registry_core_20261101",
        "path": "registry/snapshots/migrationbench_asset_registry_core_20261101.sqlite",
        "kind": "SQLITE_PROJECT_ASSET_REGISTRY_SNAPSHOT",
        "as_of": "2026-11-01T23:45:00Z",
        "immutable": 1,
        "authority": "MigrationBench core asset registry snapshot; audit log ends 2026-11-01.",
    },
    {
        "source_key": "migrationbench_plot_registry_20261101",
        "path": "results/plots/registry/migrationbench_asset_registry_20261101.sqlite",
        "kind": "SQLITE_PLOT_GALLERY_REGISTRY_SNAPSHOT",
        "as_of": None,
        "immutable": 1,
        "authority": "Plot/gallery registry snapshot; keep separate from core registry.",
    },
    {
        "source_key": "migrationbench_plot_registry_base_20261028",
        "path": "results/plots/registry/migrationbench_asset_registry_20261101_base_20261028.sqlite",
        "kind": "SQLITE_PLOT_GALLERY_REGISTRY_BASELINE",
        "as_of": "2026-10-28",
        "immutable": 1,
        "authority": "Earlier plot/gallery registry baseline; do not collapse by filename.",
    },
    {
        "source_key": "migrationbench_dft_neb_registry",
        "path": "data/collaborator_dft_neb/migrationbench_dft_neb.sqlite",
        "kind": "SQLITE_DFT_NEB_REGISTRY",
        "as_of": None,
        "immutable": 0,
        "authority": "DFT-NEB calculation and endpoint registry; known artifact/publication-link gaps remain.",
    },
    {
        "source_key": "zero_force_training_provenance",
        "path": "data/processed/cluster/training_provenance/zero_force_training.sqlite",
        "kind": "SQLITE_FRAME_TRAINING_PROVENANCE",
        "as_of": None,
        "immutable": 0,
        "authority": "Frame-level training provenance; not a complete source-run graph.",
    },
    {
        "source_key": "phase1_cleanup_audit_20261122",
        "path": "analysis/rockfish_skipjack_migration_audit_20261121/cleanup/cleanup_audit.sqlite",
        "kind": "SQLITE_CLEANUP_AND_EVIDENCE_LEDGER",
        "as_of": "2026-11-22",
        "immutable": 0,
        "authority": "Prior local and Rockfish cleanup ledger; retained rows are historical evidence, not a current filesystem inventory.",
    },
]

ROOTS = [
    ("RF-DATA-FS", "Rockfish", "data", "/data", "/data", None, None, "OTHER", "READ_ONLY", "Shared multi-petabyte GPFS mount; quota snapshots do not authorize traversal or deletion of other groups' data."),
    ("RF-SCRATCH4-FS", "Rockfish", "scratch4", "/scratch4", "/scratch4", None, None, "OTHER", "READ_ONLY", "Shared multi-petabyte GPFS mount; quota snapshots do not authorize traversal or deletion of other groups' data."),
    ("RF-DATA-YI", "Rockfish", "data", "/data", "/data/pclancy3/yi", "ycao73", "pclancy3", "USER_SUBTREE", "ACTIVE", "Mixed personal project tree; inventory is being refreshed. Do not infer a single project for this root."),
    ("RF-SCRATCH4-YI", "Rockfish", "scratch4", "/scratch4", "/scratch4/pclancy3/yi", "ycao73", "pclancy3", "USER_SUBTREE", "ACTIVE", "User subtree; phase-2 inventory and WSe2 restart review in progress."),
    ("RF-SCRATCH16-YI", "Rockfish", "scratch16", "/scratch16", "/scratch16/pclancy3/yi", "ycao73", "pclancy3", "USER_SUBTREE", "READ_ONLY", "Prior audit has a scoped, incomplete inventory; not covered by current root walk."),
    ("RF-HOME-YI", "Rockfish", "home", "/home", "/home/ycao73", "ycao73", "pclancy3", "USER_SUBTREE", "READ_ONLY", "Home storage; QE pseudopotentials and run support files may be referenced from calculations."),
    ("LOCAL-MAIN", "Local", "APFS", str(REPO_ROOT), str(REPO_ROOT), "alina", "staff", "PROJECT_SUBTREE", "ACTIVE", "Unified local MigrationBench main worktree used for the audit and SQL catalog."),
]


def main() -> None:
    if not DB_PATH.is_file():
        raise SystemExit(f"Catalog DB is missing: {DB_PATH}")
    captured = now_utc()
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        for item in SOURCES:
            path = REPO_ROOT / item["path"]
            if not path.is_file():
                raise SystemExit(f"Missing registry source: {path}")
            identity = (str(path), sha256(path))
            prior = connection.execute(
                "SELECT source_path, sha256 FROM registry_source WHERE source_key = ?",
                (item["source_key"],),
            ).fetchone()
            if prior and prior != identity:
                raise SystemExit(f"Source changed since registration: {item['source_key']}")
            connection.execute(
                """INSERT OR IGNORE INTO registry_source
                   (source_key, source_path, registry_kind, sha256, captured_utc,
                    as_of_utc, immutable_snapshot, authority_note)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    item["source_key"], identity[0], item["kind"], identity[1],
                    captured, item["as_of"], item["immutable"], item["authority"],
                ),
            )

        for values in ROOTS:
            (root_id, cluster, filesystem, mount, scoped, user, group,
             scope_kind, status, notes) = values
            prior = connection.execute(
                """SELECT cluster_name, filesystem_name, mount_path, scoped_path,
                          owner_user, owner_group, scope_kind, status, notes
                   FROM storage_root WHERE root_id = ?""",
                (root_id,),
            ).fetchone()
            expected = (cluster, filesystem, mount, scoped, user, group,
                        scope_kind, status, notes)
            if prior and prior != expected:
                raise SystemExit(f"Storage root changed since registration: {root_id}")
            connection.execute(
                """INSERT OR IGNORE INTO storage_root
                   (root_id, cluster_name, filesystem_name, mount_path, scoped_path,
                    owner_user, owner_group, scope_kind, status, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                values,
            )
        connection.commit()

    with sqlite3.connect(DB_PATH) as connection:
        print(f"Catalog: {DB_PATH}")
        print(f"Registered sources: {connection.execute('SELECT COUNT(*) FROM registry_source').fetchone()[0]}")
        print(f"Registered roots: {connection.execute('SELECT COUNT(*) FROM storage_root').fetchone()[0]}")
        for row in connection.execute(
            "SELECT source_key, sha256, source_path FROM registry_source ORDER BY source_id"
        ):
            print("\t".join(row))


if __name__ == "__main__":
    main()
