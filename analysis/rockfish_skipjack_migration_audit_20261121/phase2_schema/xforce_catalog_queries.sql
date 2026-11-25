-- X-FORCE/DUAL-X source archive and experiment evidence.

-- Current Rockfish archive snapshots, counts, byte totals, and verification notes.
SELECT r.scoped_path, r.cluster_name, s.snapshot_id, s.started_utc, s.finished_utc,
       s.completion_state, s.files_seen, s.logical_bytes, s.allocated_bytes,
       s.manifest_path, s.manifest_sha256, s.notes
FROM inventory_snapshot AS s
JOIN storage_root AS r USING(root_id)
WHERE s.snapshot_id IN ('XFORCE-ROCKFISH-ARCHIVE-20261123',
                        'MB-ROCKFISH-LARGE-ASSET-ARCHIVE-20261123')
ORDER BY s.snapshot_id;

-- Complete X-FORCE archive file list: category, logical size, SHA-256,
-- archive path, allocation observation, and scientific status.
SELECT a.artifact_id, a.artifact_kind, a.semantic_role, a.byte_size,
       a.sha256, a.scientific_status, a.retention_class,
       sp.relative_path AS rockfish_archive_path, io.allocated_bytes,
       io.hash_scope, io.observation_status, a.notes
FROM artifact AS a
JOIN artifact_location AS al USING(artifact_id)
JOIN storage_path AS sp USING(path_id)
JOIN storage_root AS sr USING(root_id)
LEFT JOIN inventory_observation AS io
       ON io.path_id=sp.path_id AND io.snapshot_id='XFORCE-ROCKFISH-ARCHIVE-20261123'
WHERE a.project_id='DUALX'
  AND sp.relative_path LIKE 'X-FORCE_data_archive/rockfish_main_snapshot_20261123/%'
  AND a.artifact_key NOT LIKE 'ARCHIVE-CONTROL:%'
ORDER BY sp.relative_path;

-- X-FORCE models and trajectory inputs for migration planning.
SELECT a.artifact_kind, a.byte_size, a.sha256, a.scientific_status,
       sp.relative_path AS rockfish_archive_path, a.notes
FROM artifact AS a
JOIN artifact_location AS al USING(artifact_id)
JOIN storage_path AS sp USING(path_id)
WHERE a.project_id='DUALX'
  AND (sp.relative_path LIKE 'X-FORCE_data_archive/rockfish_main_snapshot_20261123/models/%'
       OR sp.relative_path LIKE 'X-FORCE_data_archive/rockfish_main_snapshot_20261123/trajectories/%')
ORDER BY a.artifact_kind, sp.relative_path;

-- X-FORCE generated figures and Grad-CAM result JSONs.
SELECT a.artifact_id, a.artifact_kind, a.byte_size, a.sha256,
       sp.relative_path AS rockfish_archive_path, a.notes
FROM artifact AS a
JOIN artifact_location AS al USING(artifact_id)
JOIN storage_path AS sp USING(path_id)
WHERE a.project_id='DUALX'
  AND (sp.relative_path LIKE 'X-FORCE_data_archive/rockfish_main_snapshot_20261123/%results_figures/%'
       OR sp.relative_path LIKE 'X-FORCE_data_archive/rockfish_main_snapshot_20261123/%results_json/%')
ORDER BY sp.relative_path;

-- Large MigrationBench assets moved from Git to the Rockfish archive.
SELECT a.artifact_id, a.artifact_kind, a.semantic_role, a.byte_size,
       a.sha256, al.availability, sp.relative_path AS rockfish_archive_path,
       io.hash_scope, a.notes
FROM artifact AS a
JOIN artifact_location AS al USING(artifact_id)
JOIN storage_path AS sp USING(path_id)
LEFT JOIN inventory_observation AS io
       ON io.path_id=sp.path_id AND io.snapshot_id='MB-ROCKFISH-LARGE-ASSET-ARCHIVE-20261123'
WHERE a.project_id='MIGRATIONBENCH'
  AND sp.relative_path LIKE 'MigrationBench_data_archive/revision1_git_assets_20261123/%'
  AND a.artifact_key NOT LIKE 'ARCHIVE-CONTROL:%'
ORDER BY sp.relative_path;

-- Figure/table result packages in the submitted Revision 1 release, joined to
-- both their source evidence and the file-level Rockfish archive where present.
SELECT i.source_item_id, i.section, i.panel, i.final_asset_path,
       i.final_asset_sha256, i.source_status, i.reproduction_status,
       e.evidence_role, e.source_path, e.availability, e.relationship,
       a.artifact_key, a.sha256 AS evidence_sha256, a.byte_size,
       group_concat(DISTINCT sp.relative_path) AS recorded_locations
FROM paper_release_item AS i
LEFT JOIN paper_release_item_evidence AS e USING(release_item_id)
LEFT JOIN artifact AS a USING(artifact_id)
LEFT JOIN artifact_location AS al USING(artifact_id)
LEFT JOIN storage_path AS sp USING(path_id)
WHERE i.release_id='MB-REVISION1-SUBMITTED-20261110'
GROUP BY i.release_item_id,e.release_item_evidence_id,a.artifact_id
ORDER BY i.section,i.source_item_id,i.panel,e.evidence_role;

-- Repository references and archive-control files for DUAL-X.
SELECT gr.repository_key, gr.local_path, gr.remote_url, gr.canonical_branch,
       a.artifact_key, a.artifact_kind, a.sha256, sp.relative_path, al.availability
FROM git_repository AS gr
LEFT JOIN artifact AS a ON a.project_id='DUALX' AND a.artifact_kind='ARCHIVE_CONTROL_FILE'
LEFT JOIN artifact_location AS al USING(artifact_id)
LEFT JOIN storage_path AS sp USING(path_id)
WHERE gr.project_id='DUALX'
ORDER BY gr.repository_key, a.artifact_key;

-- Archived artifact movement records: source path in the repo and the new
-- archive path can be joined by artifact SHA-256 where the catalog has both.
SELECT a.artifact_id, a.artifact_key, a.sha256, a.byte_size,
       old_sp.relative_path AS prior_repo_path, old_al.availability AS prior_path_state,
       new_sp.relative_path AS archive_path, new_al.availability AS archive_state
FROM artifact AS a
JOIN artifact_location AS old_al ON old_al.artifact_id=a.artifact_id
JOIN storage_path AS old_sp ON old_sp.path_id=old_al.path_id
JOIN artifact_location AS new_al ON new_al.artifact_id=a.artifact_id
JOIN storage_path AS new_sp ON new_sp.path_id=new_al.path_id
WHERE a.project_id='MIGRATIONBENCH'
  AND old_sp.relative_path LIKE 'MigrationBench/%'
  AND old_al.availability='MOVED'
  AND new_sp.relative_path LIKE 'MigrationBench_data_archive/revision1_git_assets_20261123/%'
ORDER BY a.artifact_key;

-- 12. Reconstruct the Rockfish X-FORCE E_gradcam/E17 run evidence package.
-- Model aliases are explicit run_artifact notes; scientific status remains
-- provenance-partial when scheduler/run lineage is unavailable.
SELECT wr.run_id, wr.run_label, wr.workflow_type, wr.method,
       wr.scheduler_state, wr.scientific_status, wr.notes AS run_notes,
       ra.direction, ra.role AS artifact_role, ra.required_for_reproduction,
       ra.notes AS run_artifact_notes, a.artifact_key, a.artifact_kind,
       a.semantic_role, a.sha256, a.byte_size, al.availability,
       sr.scoped_path || '/' || sp.relative_path AS exact_artifact_path
FROM workflow_run AS wr
JOIN run_artifact AS ra ON ra.run_id = wr.run_id
JOIN artifact AS a ON a.artifact_id = ra.artifact_id
LEFT JOIN artifact_location AS al ON al.artifact_id = a.artifact_id
LEFT JOIN storage_path AS sp ON sp.path_id = al.path_id
LEFT JOIN storage_root AS sr ON sr.root_id = sp.root_id
WHERE wr.project_id = 'DUALX'
  AND (wr.run_id LIKE 'XFORCE-EGRADCAM-50-%'
       OR wr.run_id LIKE 'XFORCE-E17%')
ORDER BY wr.run_id, ra.direction, ra.role, a.artifact_key, exact_artifact_path;

-- 13. Query the 2026 DUAL-X MACE-NEB and QE-NEB packages as file-level
-- evidence bundles. Bind :run_id to either run ID for one bundle, or use the
-- two listed IDs together for the full 652-file archive.
SELECT wr.run_id, wr.run_label, wr.workflow_type, wr.scientific_status,
       wr.notes AS provenance_limits,
       ra.direction, ra.role, ra.required_for_reproduction, ra.notes AS link_notes,
       a.artifact_key, a.artifact_kind, a.semantic_role, a.sha256, a.byte_size,
       al.availability, sr.scoped_path || '/' || sp.relative_path AS archive_path
FROM workflow_run AS wr
JOIN run_artifact AS ra ON ra.run_id = wr.run_id
JOIN artifact AS a ON a.artifact_id = ra.artifact_id
LEFT JOIN artifact_location AS al ON al.artifact_id = a.artifact_id
LEFT JOIN storage_path AS sp ON sp.path_id = al.path_id
LEFT JOIN storage_root AS sr ON sr.root_id = sp.root_id
WHERE wr.run_id IN ('DUALX-MACE-PATH-FEATURES-32SYS-20260825',
                    'DUALX-QE-NEB-EM-LT3EV-20260726')
  AND a.artifact_key LIKE 'DUALX-RESEARCH-ARCHIVE:20261123:daily/%'
  AND (:run_id IS NULL OR wr.run_id = :run_id)
ORDER BY wr.run_id, ra.direction, a.artifact_key, archive_path;
