#!/usr/bin/env python3
"""Register verified X-FORCE and MigrationBench Rockfish archives in SQLite."""

import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

AUDIT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = AUDIT_ROOT / "phase2_data"
DB = AUDIT_ROOT / "provenance_catalog.sqlite"
STORAGE_ROOT = "RF-SCRATCH16-YI"


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_tsv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def register_source(conn, key, path, kind, authority, notes, captured):
    path = Path(path)
    source_sha = sha256_file(path)
    conn.execute(
        """INSERT OR IGNORE INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
            immutable_snapshot,authority_note,notes)
           VALUES(?,?,?,?,?,?,1,?,?)""",
        (key, str(path), kind, source_sha, captured, captured, authority, notes),
    )
    row = conn.execute("SELECT source_id,sha256 FROM registry_source WHERE source_key=?", (key,)).fetchone()
    if row is None or row["sha256"] != source_sha:
        raise RuntimeError("registry source key is bound to a different snapshot: " + key)
    return row["source_id"], source_sha


def ensure_storage_path(conn, root_id, relative_path, now):
    conn.execute(
        "INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)",
        (root_id, relative_path, now),
    )
    return conn.execute(
        "SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?",
        (root_id, relative_path),
    ).fetchone()[0]


def ensure_artifact(conn, project_id, key, kind, role, sha, size, retention, notes):
    row = conn.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?", (key,)).fetchone()
    if row:
        artifact_id = row[0]
        conn.execute(
            """UPDATE artifact SET project_id=?,artifact_kind=?,semantic_role=?,owner_status='USER_CONFIRMED',
               sha256=?,byte_size=?,scientific_status='PROVENANCE_PARTIAL',retention_class=?,notes=?
               WHERE artifact_id=?""",
            (project_id, kind, role, sha, size, retention, notes, artifact_id),
        )
        return artifact_id
    if sha:
        same = conn.execute(
            "SELECT artifact_id FROM artifact WHERE project_id=? AND sha256=? AND byte_size=? ORDER BY artifact_id LIMIT 1",
            (project_id, sha, size),
        ).fetchone()
        if same:
            return same[0]
    cur = conn.execute(
        """INSERT INTO artifact
           (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
            scientific_status,retention_class,notes)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (project_id, key, kind, role, "USER_CONFIRMED", sha, size,
         "PROVENANCE_PARTIAL", retention, notes),
    )
    return cur.lastrowid


def add_location_and_observation(conn, snapshot_id, artifact_id, root_id, relpath,
                                 size, allocated, sha, now):
    path_id = ensure_storage_path(conn, root_id, relpath, now)
    conn.execute(
        """INSERT OR IGNORE INTO artifact_location
           (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
           VALUES(?,?,'PRESENT',?,?)""",
        (artifact_id, path_id, now, now),
    )
    conn.execute(
        """UPDATE artifact_location SET availability='PRESENT',last_verified_utc=?
           WHERE artifact_id=? AND path_id=?""",
        (now, artifact_id, path_id),
    )
    if snapshot_id:
        conn.execute(
            """INSERT OR IGNORE INTO inventory_observation
               (snapshot_id,path_id,entry_type,logical_bytes,allocated_bytes,sha256,hash_scope,
                observation_status,link_count)
               VALUES(?,?,'FILE',?,?,?,'FULL_FILE','PRESENT',1)""",
            (snapshot_id, path_id, size, allocated, sha),
        )
        conn.execute(
            """UPDATE inventory_observation SET logical_bytes=?,allocated_bytes=?,sha256=?,
               hash_scope='FULL_FILE',observation_status='PRESENT',entry_type='FILE',link_count=1
               WHERE snapshot_id=? AND path_id=?""",
            (size, allocated, sha, snapshot_id, path_id),
        )
    return path_id


def category_metadata(category, path, project_id):
    ext = Path(path).suffix.lower()
    if project_id == "DUALX":
        mapping = {
            "models": ("MODEL_CHECKPOINT", "model checkpoint", "PERMANENT_SOURCE"),
            "trajectories": ("TRAJECTORY", "coordinate trajectory or structure", "REPRODUCIBLE_INPUT"),
            "qe_outputs": ("QE_OUTPUT", "charge-density or volumetric QE output", "PUBLICATION_EVIDENCE"),
            "raw_inputs_logs": ("RUN_INPUT_OR_LOG", "calculation input, script, or scheduler log", "PUBLICATION_EVIDENCE"),
            "datasets": ("DATASET", "source dataset", "REPRODUCIBLE_INPUT"),
            "results_figures": ("FIGURE", "generated result figure", "PUBLICATION_EVIDENCE"),
            "results_json": ("RESULT", "structured experiment result", "COMPACT_RESULT"),
            "reproducibility_scripts": ("SOURCE_CODE", "reproducibility or execution script", "REPRODUCIBLE_INPUT"),
            "source_tree_variants": ("SOURCE_VARIANT", "historical source-tree variant", "PERMANENT_SOURCE"),
            "other_reviewed_source": ("SOURCE", "reviewed supporting source", "PERMANENT_SOURCE"),
        }
        return mapping.get(category, ("ARCHIVED_FILE", category, "UNCLASSIFIED"))
    if ext == ".blend":
        return "BLENDER_SCENE", "editable Blender scene", "PUBLICATION_EVIDENCE"
    if ext == ".png":
        return "HIGH_RESOLUTION_RENDER", "high-resolution publication render", "PUBLICATION_EVIDENCE"
    if ext == ".npz":
        return "REPRODUCIBILITY_ARRAY", "figure or analysis input array", "REPRODUCIBLE_INPUT"
    if ext == ".mp4":
        return "RENDERED_MEDIA", "rendered project media", "PUBLICATION_EVIDENCE"
    return "ARCHIVED_FILE", category, "UNCLASSIFIED"


def register_manifest_artifact(conn, project_id, source_key, source_path, archive_root_rel,
                               root_id, now, run_id=None):
    source_id, sha = register_source(
        conn, source_key, source_path, "ROCKFISH_ARCHIVE_CONTROL_FILE",
        "File hash from local mirror; remote archive path recorded in the SQLite location.",
        "Archive control file; does not imply scientific validation.", now,
    )
    path = Path(source_path)
    artifact_id = ensure_artifact(
        conn, project_id, "ARCHIVE-CONTROL:" + source_key, "ARCHIVE_CONTROL_FILE",
        "archive manifest/status/readme", sha, path.stat().st_size,
        "PUBLICATION_EVIDENCE", "sha256=" + sha + "; " + str(path),
    )
    if archive_root_rel:
        add_location_and_observation(conn, None, artifact_id, root_id, archive_root_rel,
                                     path.stat().st_size, None, sha, now)
    if run_id:
        conn.execute(
            "INSERT OR IGNORE INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes) VALUES(?,?,?,'PROVENANCE',0,?)",
            (run_id, artifact_id, "ARCHIVE_CONTROL", "registry_source_id=" + str(source_id)),
        )
    return artifact_id, source_id


def import_snapshot(conn, *, project_id, run_id, snapshot_id, root_id, items,
                    manifest_sha, manifest_path, now, notes):
    total_bytes = sum(int(r["size_bytes"]) for r in items)
    # `st_blocks` captured immediately after streaming copies was zero for many
    # files on this GPFS mount. Keep per-file allocation unknown; aggregate `du`
    # is recorded in the snapshot notes instead of treating zero as real usage.
    total_allocated = None
    conn.execute(
        """INSERT OR IGNORE INTO inventory_snapshot
           (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,
            files_seen,directories_seen,symlinks_seen,unreadable_count,logical_bytes,allocated_bytes,
            manifest_path,manifest_sha256,notes)
           VALUES(?,?,?,?,?,'archive-import-v1','COMPLETE',?,NULL,NULL,0,?,?,?,?,?)""",
        (snapshot_id, root_id, now, now, "Git index/source snapshot copy with full-file SHA-256 verification",
         len(items), total_bytes, total_allocated, str(manifest_path), manifest_sha, notes),
    )
    conn.execute(
        """UPDATE inventory_snapshot SET finished_utc=?,completion_state='COMPLETE',files_seen=?,
           logical_bytes=?,allocated_bytes=?,manifest_path=?,manifest_sha256=?,notes=? WHERE snapshot_id=?""",
        (now, len(items), total_bytes, total_allocated, str(manifest_path), manifest_sha, notes, snapshot_id),
    )
    for row in items:
        path = row["source_rel"]
        category = row["category"]
        archive_rel = row["archive_rel"]
        kind, role, retention = category_metadata(category, path, project_id)
        sha = row["sha256"]
        size = int(row["size_bytes"])
        key = project_id + "-ROCKFISH-ARCHIVE:20261123:" + path
        notes_text = "source_rel=" + path + "; archive_rel=" + archive_rel + "; category=" + category
        for field in ("source_git_status", "record_class", "git_blob_oid", "source_branch", "source_head", "source_merge_head"):
            if row.get(field):
                notes_text += "; " + field + "=" + row[field]
        artifact_id = ensure_artifact(conn, project_id, key, kind, role, sha, size, retention, notes_text)
        root_relative = ("X-FORCE_data_archive/rockfish_main_snapshot_20261123/" if project_id == "DUALX"
                         else "MigrationBench_data_archive/revision1_git_assets_20261123/") + archive_rel
        path_id = add_location_and_observation(
            conn, snapshot_id, artifact_id, root_id, root_relative, size,
            None, sha, now,
        )
        direction = "MODEL" if kind == "MODEL_CHECKPOINT" else (
            "FIGURE" if kind in ("FIGURE", "HIGH_RESOLUTION_RENDER", "BLENDER_SCENE") else
            "DATASET" if kind in ("TRAJECTORY", "DATASET", "REPRODUCIBILITY_ARRAY") else
            "LOG" if kind == "RUN_INPUT_OR_LOG" else "OUTPUT")
        conn.execute(
            "INSERT OR IGNORE INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes) VALUES(?,?,?,?,?,?)",
            (run_id, artifact_id, category, direction, 0, "archive_path_id=" + str(path_id) + "; full SHA-256 verified"),
        )

        if project_id == "MIGRATIONBENCH":
            old_path = "MigrationBench/" + path
            old_row = conn.execute(
                """SELECT sp.path_id FROM storage_path sp JOIN artifact_location al USING(path_id)
                   WHERE sp.root_id=? AND sp.relative_path=? AND al.artifact_id=?""",
                (root_id, old_path, artifact_id),
            ).fetchone()
            if old_row:
                conn.execute(
                    "UPDATE artifact_location SET availability='MOVED',last_verified_utc=? WHERE artifact_id=? AND path_id=?",
                    (now, artifact_id, old_row[0]),
                )
            for ev in conn.execute(
                """SELECT release_item_id,evidence_role FROM paper_release_item_evidence
                   WHERE artifact_id=? AND source_path=?""", (artifact_id, path)
            ).fetchall():
                archive_evidence_path = "MigrationBench_data_archive/revision1_git_assets_20261123/" + archive_rel
                conn.execute(
                    """INSERT OR IGNORE INTO paper_release_item_evidence
                       (release_item_id,artifact_id,source_path,evidence_role,availability,relationship,notes)
                       VALUES(?,?,?,?,'PRESENT','ARCHIVED_COPY_AFTER_GIT_CLEANUP',?)""",
                    (ev[0], artifact_id, archive_evidence_path, ev[1],
                     "Byte-identical archive copy; SHA-256=" + sha + "; original tree path removed from Git."),
                )


def main():
    now = utc_now()
    x_base = DATA_ROOT / "xforce_archive_manifest.tsv"
    x_supplement = DATA_ROOT / "git_candidate_assets_manifest_20261123.tsv"
    x_status = DATA_ROOT / "xforce_archive_status.json"
    x_provenance = DATA_ROOT / "XFORCE_ROCKFISH_PROVENANCE_MANIFEST_20261123.tsv"
    mb_manifest = DATA_ROOT / "git_large_assets_manifest_20261123.tsv"
    mb_status = DATA_ROOT / "archive_status.json"
    mb_readme = DATA_ROOT / "MIGRATIONBENCH_ROCKFISH_LARGE_ASSET_ARCHIVE_20261123.md"
    for path in (x_base, x_supplement, mb_manifest, mb_readme):
        if not path.is_file():
            raise RuntimeError("required captured evidence file missing: " + str(path))

    x_base_rows = read_tsv(x_base)
    x_supp_rows = read_tsv(x_supplement)
    x_items = []
    for row in x_base_rows:
        x_items.append({
            "source_rel": row["source_rel"], "archive_rel": row["archive_rel"],
            "category": row["category"], "size_bytes": row["size_bytes"],
            "sha256": row["sha256"], "allocated_bytes": row.get("allocated_bytes"),
            "source_git_status": row.get("source_git_status"), "record_class": row.get("record_class"),
        })
    x_items.extend(x_supp_rows)
    x_manifest_sha = sha256_file(x_base)
    x_supp_sha = sha256_file(x_supplement)
    x_status_data = json.loads(x_status.read_text(encoding="utf-8")) if x_status.is_file() else {}
    if x_status_data:
        if len(x_items) != int(x_status_data["aggregate_files"]):
            raise RuntimeError("X-FORCE archive asset count mismatches status")
        if sum(int(r["size_bytes"]) for r in x_items) != int(x_status_data["aggregate_logical_bytes"]):
            raise RuntimeError("X-FORCE archive asset bytes mismatch status")

    mb_items = read_tsv(mb_manifest)
    mb_status_data = json.loads(mb_status.read_text(encoding="utf-8")) if mb_status.is_file() else {}
    if mb_status_data:
        if len(mb_items) != int(mb_status_data["verified_files"]):
            raise RuntimeError("MigrationBench archive asset count mismatches status")
        if sum(int(r["size_bytes"]) for r in mb_items) != int(mb_status_data["logical_bytes"]):
            raise RuntimeError("MigrationBench archive asset bytes mismatch status")

    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        conn.execute("BEGIN IMMEDIATE")
        for project_id in ("MIGRATIONBENCH", "DUALX"):
            if not conn.execute("SELECT 1 FROM project WHERE project_id=?", (project_id,)).fetchone():
                raise RuntimeError("project missing: " + project_id)
        if not conn.execute("SELECT 1 FROM storage_root WHERE root_id=?", (STORAGE_ROOT,)).fetchone():
            raise RuntimeError("storage root missing: " + STORAGE_ROOT)

        conn.execute(
            """INSERT OR IGNORE INTO git_repository
               (repository_key,local_path,remote_url,project_id,ownership_status,canonical_branch,notes)
               VALUES('rockfish:/scratch16/pclancy3/yi/X-FORCE','/scratch16/pclancy3/yi/X-FORCE',
               'https://github.com/yicao-elina/X-FORCE.git','DUALX','USER','main',?)""",
            ("Canonical Rockfish candidate checkout; consolidation PR pending. Archive source tree has dirty legacy origin.",),
        )
        conn.execute(
            """UPDATE git_repository SET project_id='DUALX',ownership_status='USER',notes=notes || ?
               WHERE repository_key='rockfish:/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/X-FORCE'
                 AND project_id IS NULL""",
            ("; linked to DUALX project by 2026-11-23 archive audit",),
        )

        x_run = "XFORCE-ROCKFISH-ARCHIVE-20261123"
        mb_run = "MB-ROCKFISH-LARGE-ASSET-ARCHIVE-20261123"
        for run_id, project_id, relative, notes in (
            (x_run, "DUALX", "X-FORCE_data_archive/rockfish_main_snapshot_20261123",
             "1063 base snapshot files plus 31 Git-index TIFFs; base source checkout was dirty; full-file SHA-256 verified; no WFC/mixing payloads."),
            (mb_run, "MIGRATIONBENCH", "MigrationBench_data_archive/revision1_git_assets_20261123",
             "14 large assets copied directly from the staged Git index and full-file SHA-256 verified before removing them from the Git tree."),
        ):
            conn.execute(
                """INSERT OR IGNORE INTO workflow_run
                   (run_id,project_id,run_label,workflow_type,method,system_key,source_root_id,
                    source_relative_path,scheduler_state,scientific_status,start_utc,end_utc,notes)
                   VALUES(?,?,?,'ARCHIVE_MIGRATION','Git index/source snapshot copy plus full SHA-256 verification',
                    'Rockfish',?,?, 'COMPLETED','PROVENANCE_PARTIAL',?,?,?)""",
                (run_id, project_id, run_id, STORAGE_ROOT, relative, now, now, notes),
            )
            conn.execute(
                """UPDATE workflow_run SET end_utc=?,scheduler_state='COMPLETED',scientific_status='PROVENANCE_PARTIAL',notes=?
                   WHERE run_id=?""", (now, notes, run_id)
            )

        x_sources = [
            ("DUALX-XFORCE-BASE-MANIFEST-20261123", x_base, "Base archive manifest.",
             "X-FORCE_data_archive/rockfish_main_snapshot_20261123/archive_manifest.tsv", x_run),
            ("DUALX-XFORCE-SUPPLEMENT-MANIFEST-20261123", x_supplement, "Git-derived TIFF supplement manifest.",
             "X-FORCE_data_archive/rockfish_main_snapshot_20261123/git_candidate_assets_manifest_20261123.tsv", x_run),
        ]
        if x_status.is_file():
            x_sources.append(("DUALX-XFORCE-ARCHIVE-STATUS-20261123", x_status, "Archive status and counts.",
                              "X-FORCE_data_archive/rockfish_main_snapshot_20261123/archive_status.json", x_run))
        # The detailed source-tree manifest records absolute /data paths and
        # inode/mtime metadata. It stays in the local catalog workspace; a
        # sanitized copy is not needed for the public archive pointer.
        if x_provenance.is_file():
            x_sources.append((
                "DUALX-XFORCE-PROVENANCE-MANIFEST-20261123", x_provenance,
                "Source-tree and variant provenance inventory (local catalog source).",
                "X-FORCE_data_archive/XFORCE_ROCKFISH_PROVENANCE_MANIFEST_20261123.tsv", x_run,
            ))
        x_control_ids = []
        for key, path, notes, rel, run_id in x_sources:
            source_id, source_sha = register_source(
                conn, key, path, "ROCKFISH_ARCHIVE_EVIDENCE",
                "Manifest/status captured from the Rockfish archive; hashes are source-backed.", notes, now)
            aid = ensure_artifact(conn, "DUALX", "ARCHIVE-CONTROL:" + key, "ARCHIVE_CONTROL_FILE",
                                  notes, source_sha, path.stat().st_size, "PUBLICATION_EVIDENCE",
                                  "source_sha256=" + source_sha + "; " + str(path))
            add_location_and_observation(conn, None, aid, STORAGE_ROOT, rel, path.stat().st_size,
                                         path.stat().st_size, source_sha, now)
            conn.execute("INSERT OR IGNORE INTO run_artifact(run_id,artifact_id,role,direction,notes) VALUES(?,?,?,'PROVENANCE',?)",
                         (run_id, aid, "ARCHIVE_CONTROL", "registry_source_id=" + str(source_id)))
            x_control_ids.append(aid)

        mb_sources = [
            ("MB-LARGE-ASSET-MANIFEST-20261123", mb_manifest, "Large Git-index asset manifest.",
             "MigrationBench_data_archive/revision1_git_assets_20261123/git_large_assets_manifest_20261123.tsv", mb_run),
            ("MB-LARGE-ASSET-ARCHIVE-README-20261123", mb_readme, "Archive restore instructions.",
             "MigrationBench_data_archive/revision1_git_assets_20261123/README.md", mb_run),
        ]
        if mb_status.is_file():
            mb_sources.append(("MB-LARGE-ASSET-ARCHIVE-STATUS-20261123", mb_status, "Archive status and counts.",
                               "MigrationBench_data_archive/revision1_git_assets_20261123/archive_status.json", mb_run))
        mb_control_ids = []
        for key, path, notes, rel, run_id in mb_sources:
            source_id, source_sha = register_source(
                conn, key, path, "ROCKFISH_ARCHIVE_EVIDENCE",
                "Manifest/status captured from the Rockfish archive; hashes are source-backed.", notes, now)
            aid = ensure_artifact(conn, "MIGRATIONBENCH", "ARCHIVE-CONTROL:" + key, "ARCHIVE_CONTROL_FILE",
                                  notes, source_sha, path.stat().st_size, "PUBLICATION_EVIDENCE",
                                  "source_sha256=" + source_sha + "; " + str(path))
            add_location_and_observation(conn, None, aid, STORAGE_ROOT, rel, path.stat().st_size,
                                         path.stat().st_size, source_sha, now)
            conn.execute("INSERT OR IGNORE INTO run_artifact(run_id,artifact_id,role,direction,notes) VALUES(?,?,?,'PROVENANCE',?)",
                         (run_id, aid, "ARCHIVE_CONTROL", "registry_source_id=" + str(source_id)))
            mb_control_ids.append(aid)

        x_snapshot = "XFORCE-ROCKFISH-ARCHIVE-20261123"
        mb_snapshot = "MB-ROCKFISH-LARGE-ASSET-ARCHIVE-20261123"
        import_snapshot(
            conn, project_id="DUALX", run_id=x_run, snapshot_id=x_snapshot, root_id=STORAGE_ROOT,
            items=x_items, manifest_sha=x_manifest_sha,
            manifest_path="archive_manifest.tsv; supplement=" + x_supp_sha, now=now,
            notes="1094 payload files: 1063 source snapshot entries and 31 high-resolution Git-index TIFF assets; full-file checksums verified. Source checkout dirty; scientific acceptance not inferred." 
                  + " Per-file allocated bytes are NULL because immediate st_blocks values were inconsistent; `du -s -B1` measured 5573435904 bytes including manifest/status controls. Separate manifest/status artifacts registered.",
        )
        import_snapshot(
            conn, project_id="MIGRATIONBENCH", run_id=mb_run, snapshot_id=mb_snapshot, root_id=STORAGE_ROOT,
            items=mb_items, manifest_sha=sha256_file(mb_manifest), manifest_path=str(mb_manifest), now=now,
            notes="14 large binaries copied from Git index, then removed from current Git working tree; full-file SHA-256 verified. Per-file allocated bytes are NULL because immediate st_blocks values were inconsistent; `du -s -B1` measured 205401600 bytes including the manifest. Duplicate same-content paths are retained as separate locations.",
        )

        # Make archive-control files run outputs in addition to provenance inputs.
        for run_id, ids in ((x_run, x_control_ids), (mb_run, mb_control_ids)):
            for aid in ids:
                conn.execute(
                    "INSERT OR IGNORE INTO run_artifact(run_id,artifact_id,role,direction,notes) VALUES(?,?,?,'PROVENANCE','Archive manifest/status/readme')",
                    (run_id, aid, "ARCHIVE_CONTROL"),
                )

        # Register the repository pointer documentation in the Git tree.
        for project_id, rel, key, sha, size, notes in (
            ("DUALX", "X-FORCE/ROCKFISH_DATA_ARCHIVE.md", "DUALX-ROCKFISH-DATA-ARCHIVE-POINTER-20261123",
             sha256_file(DATA_ROOT / "xforce_gradcam_update" / "ROCKFISH_DATA_ARCHIVE.md"),
             (DATA_ROOT / "xforce_gradcam_update" / "ROCKFISH_DATA_ARCHIVE.md").stat().st_size,
             "Git pointer to Rockfish source and figure archive."),
            ("MIGRATIONBENCH", "MigrationBench/docs/reproducibility/rockfish_large_asset_archive_20261123.md",
             "MB-ROCKFISH-LARGE-ASSET-POINTER-20261123", sha256_file(mb_readme), mb_readme.stat().st_size,
             "Git pointer to Rockfish large asset archive and restore recipe."),
        ):
            aid = ensure_artifact(conn, project_id, key, "ARCHIVE_POINTER", "archive location and restoration instructions",
                                  sha, size, "PUBLICATION_EVIDENCE", notes)
            path_id = ensure_storage_path(conn, STORAGE_ROOT, rel, now)
            conn.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)",
                         (aid, path_id, now, now))

        conn.commit()
        for table in ("artifact", "inventory_snapshot", "inventory_observation", "artifact_location", "paper_release_item_evidence"):
            if table == "inventory_snapshot":
                print(table, [tuple(r) for r in conn.execute("SELECT snapshot_id,files_seen,logical_bytes,completion_state FROM inventory_snapshot WHERE snapshot_id IN (?,?) ORDER BY snapshot_id", (x_snapshot, mb_snapshot)).fetchall()])
            elif table == "inventory_observation":
                print(table, [tuple(r) for r in conn.execute("SELECT snapshot_id,count(*),sum(logical_bytes) FROM inventory_observation WHERE snapshot_id IN (?,?) GROUP BY snapshot_id ORDER BY snapshot_id", (x_snapshot, mb_snapshot)).fetchall()])
            elif table == "artifact_location":
                print(table, conn.execute("SELECT count(*) FROM artifact_location").fetchone()[0])
            elif table == "artifact":
                print(table, [tuple(r) for r in conn.execute("SELECT project_id,count(*) FROM artifact WHERE artifact_key LIKE '%ROCKFISH-ARCHIVE:20261123:%' GROUP BY project_id").fetchall()])
            else:
                print(table, conn.execute("SELECT count(*) FROM paper_release_item_evidence WHERE relationship='ARCHIVED_COPY_AFTER_GIT_CLEANUP'").fetchone()[0])
        print("xforce_base_manifest_sha256", x_manifest_sha)
        print("xforce_supplement_manifest_sha256", x_supp_sha)
        print("migrationbench_large_asset_manifest_sha256", sha256_file(mb_manifest))
        print("migrationbench_release_items", conn.execute("SELECT count(*) FROM paper_release_item WHERE release_id='MB-REVISION1-SUBMITTED-20261110'").fetchone()[0])
        print("migrationbench_evidence_links", conn.execute("SELECT count(*) FROM paper_release_item_evidence WHERE release_item_id IN (SELECT release_item_id FROM paper_release_item WHERE release_id='MB-REVISION1-SUBMITTED-20261110')").fetchone()[0])
        violations = conn.execute("PRAGMA foreign_key_check").fetchmany(10)
        if violations:
            raise RuntimeError("foreign key check failed: " + repr(violations))
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
