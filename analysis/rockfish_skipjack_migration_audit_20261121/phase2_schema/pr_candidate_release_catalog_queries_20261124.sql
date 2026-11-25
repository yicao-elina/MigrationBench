-- Read-only queries for the additive PR #10/#12/#13 review-candidate import.
-- These release IDs are separate from MB-REVISION1-SUBMITTED-20261110.

-- 1. Review-candidate release identity, source branch, head, and package commit.
SELECT release_id, release_label, status, manuscript_commit,
       overleaf_version, released_utc, manifest_artifact_id, notes
FROM paper_release
WHERE release_id GLOB 'MB-PR10-639-REVIEW-*'
   OR release_id GLOB 'MB-PR12-CRNEB-REVIEW-*'
   OR release_id GLOB 'MB-PR13-ENDPOINT47-REVIEW-*'
ORDER BY release_id;

-- 2. Exact source claim status alongside conservative catalog assessment.
SELECT o.release_id, o.source_claim_id, o.source_row_key,
       o.source_scientific_status, c.scientific_status AS catalog_assessment,
       c.publication_action, o.question, o.primary_artifact,
       o.source_or_table, o.boundary
FROM paper_release_claim_observation AS o
JOIN paper_claim AS c ON c.claim_id = o.catalog_claim_id
WHERE o.release_id GLOB 'MB-PR10-639-REVIEW-*'
   OR o.release_id GLOB 'MB-PR13-ENDPOINT47-REVIEW-*'
ORDER BY o.release_id, o.source_claim_id, o.source_row_key;

-- 3. PR #12 figure candidates remain unselected; no blank-template claims exist.
SELECT figure_id, title, chart_status, provenance_status,
       scientific_boundary, source_registry_source_id
FROM paper_figure
WHERE figure_id GLOB 'MB-PR12-*'
ORDER BY figure_id;

SELECT release_id, count(*) AS claim_rows
FROM paper_claim
WHERE release_id GLOB 'MB-PR12-CRNEB-REVIEW-*'
GROUP BY release_id;

-- 4. Hash and pinned GitHub location for package-file content artifacts.
-- PR #13 manifest declarations that are absent from Git or stale are separate
-- row-level observations; query 8 covers them.
SELECT i.release_id, e.source_path AS pinned_source_path,
       a.sha256, a.byte_size, e.availability, e.relationship, e.notes
FROM paper_release_item AS i
JOIN paper_release_item_evidence AS e USING(release_item_id)
LEFT JOIN artifact AS a ON a.artifact_id = e.artifact_id
WHERE i.source_item_id = 'PACKAGE_FILE_INDEX'
  AND e.evidence_role = 'PACKAGE_FILE'
  AND (i.release_id GLOB 'MB-PR10-639-REVIEW-*'
       OR i.release_id GLOB 'MB-PR12-CRNEB-REVIEW-*'
       OR i.release_id GLOB 'MB-PR13-ENDPOINT47-REVIEW-*')
ORDER BY i.release_id, e.source_path;

-- 5. PR #13 explicit claim-to-figure relations only; table artifacts remain
-- linked through claim_evidence and paper_release_item_evidence.
SELECT c.release_id, f.source_claim_key, f.claim_id, f.figure_id,
       f.relationship, p.title, p.provenance_status
FROM paper_claim_figure AS f
JOIN paper_claim AS c ON c.claim_id = f.claim_id
JOIN paper_figure AS p ON p.figure_id = f.figure_id
WHERE c.release_id GLOB 'MB-PR13-ENDPOINT47-REVIEW-*'
   OR c.release_id GLOB 'MB-PR10-639-REVIEW-*'
ORDER BY c.release_id, c.claim_id, f.figure_id;

-- 6. Package/source snapshot records with their captured bytes and exact URLs.
SELECT s.registry_kind, s.source_key, s.source_path, s.sha256,
       s.captured_utc, s.as_of_utc, s.authority_note
FROM registry_source AS s
WHERE s.source_key LIKE 'migrationbench_pr10:%'
   OR s.source_key LIKE 'migrationbench_pr12:%'
   OR s.source_key LIKE 'migrationbench_pr13:%'
   OR s.source_key LIKE 'migrationbench_pr13_archive_control_%'
   OR s.source_key LIKE 'migrationbench_pr13_git_tree_audit_20261124:%'
   OR s.source_key LIKE 'migrationbench_pr_candidate_capture_manifest_20261124:%'
ORDER BY s.source_key;

-- 7. Confirm the submitted release record is separate and unchanged.
SELECT release_id, release_label, status, manuscript_commit,
       manifest_artifact_id, notes
FROM paper_release
WHERE release_id = 'MB-REVISION1-SUBMITTED-20261110';

-- 8. PR #13 FILE_PROVENANCE row reconciliation status, including manifest-only
-- declarations for paths absent from both pinned Git trees.
SELECT r.relationship AS tree_reconciliation_status, count(*) AS manifest_rows
FROM registry_entity_ref AS r
JOIN paper_release_item AS i
  ON CAST(r.catalog_entity_key AS INTEGER) = i.release_item_id
WHERE i.release_id GLOB 'MB-PR13-ENDPOINT47-REVIEW-*'
  AND r.entity_type = 'PR13_FILE_PROVENANCE_ROW'
GROUP BY r.relationship
ORDER BY r.relationship;

-- 8b. Exact source-manifest row declarations and the linked Git-tree audit.
SELECT i.release_id, r.entity_key AS relative_path,
       r.relationship AS tree_reconciliation_status, r.notes
FROM registry_entity_ref AS r
JOIN paper_release_item AS i
  ON CAST(r.catalog_entity_key AS INTEGER) = i.release_item_id
WHERE i.release_id GLOB 'MB-PR13-ENDPOINT47-REVIEW-*'
  AND r.entity_type = 'PR13_FILE_PROVENANCE_ROW'
ORDER BY r.entity_key;

-- 9. Current Git file artifacts verified against both PR #13 pinned commits.
SELECT i.release_id, e.source_path AS pinned_pr_head_path,
       a.sha256, a.byte_size, e.relationship, e.notes
FROM paper_release_item AS i
JOIN paper_release_item_evidence AS e USING(release_item_id)
JOIN artifact AS a ON a.artifact_id = e.artifact_id
WHERE i.release_id GLOB 'MB-PR13-ENDPOINT47-REVIEW-*'
  AND e.evidence_role = 'PACKAGE_FILE'
ORDER BY e.source_path;

-- 10. Raw evidence present under the dedicated Rockfish archive root, separate
-- from Git-tree availability. These locations have live-verified hashes.
SELECT i.release_id, e.source_path AS rockfish_archive_path,
       a.sha256, a.byte_size, e.availability, e.relationship,
       sp.root_id, sp.relative_path, al.first_verified_utc, e.notes
FROM paper_release_item AS i
JOIN paper_release_item_evidence AS e USING(release_item_id)
JOIN artifact AS a ON a.artifact_id = e.artifact_id
JOIN artifact_location AS al ON al.artifact_id = a.artifact_id
JOIN storage_path AS sp ON sp.path_id = al.path_id
WHERE i.release_id GLOB 'MB-PR13-ENDPOINT47-REVIEW-*'
  AND e.evidence_role = 'ROCKFISH_ARCHIVE_FILE'
  AND sp.root_id = 'RF-SCRATCH16-YI'
  AND e.source_path = ('/scratch16/pclancy3/yi/' || sp.relative_path)
ORDER BY sp.relative_path;

-- 10b. Compact count/byte check for the 1,336 verified Rockfish raw files.
SELECT count(*) AS present_archive_file_rows,
       sum(a.byte_size) AS total_verified_bytes,
       count(DISTINCT a.sha256) AS unique_content_hashes
FROM paper_release_item AS i
JOIN paper_release_item_evidence AS e USING(release_item_id)
JOIN artifact AS a ON a.artifact_id = e.artifact_id
JOIN artifact_location AS al ON al.artifact_id = a.artifact_id
JOIN storage_path AS sp ON sp.path_id = al.path_id
WHERE i.release_id GLOB 'MB-PR13-ENDPOINT47-REVIEW-*'
  AND e.evidence_role = 'ROCKFISH_ARCHIVE_FILE'
  AND sp.root_id = 'RF-SCRATCH16-YI'
  AND e.source_path = ('/scratch16/pclancy3/yi/' || sp.relative_path);

-- 11. Archive verification/control files and the source tar are separately
-- queryable from the raw payload rows.
SELECT i.release_id, e.evidence_role, e.source_path, a.sha256, a.byte_size,
       e.availability, e.relationship, e.notes
FROM paper_release_item AS i
JOIN paper_release_item_evidence AS e USING(release_item_id)
LEFT JOIN artifact AS a ON a.artifact_id = e.artifact_id
WHERE i.release_id GLOB 'MB-PR13-ENDPOINT47-REVIEW-*'
  AND e.evidence_role IN ('ROCKFISH_ARCHIVE_CONTROL_FILE','ROCKFISH_SOURCE_BUNDLE','SOURCE_GIT_BUNDLE')
ORDER BY e.evidence_role, e.source_path;

-- 12. Archived MigrationBench model/vector payloads, hashes, semantic roles,
-- and their stable Rockfish locations.
SELECT a.artifact_id, a.artifact_key, a.artifact_kind, a.semantic_role,
       a.sha256, a.byte_size, a.scientific_status, al.availability,
       sp.root_id, sp.relative_path, a.notes
FROM artifact AS a
JOIN artifact_location AS al ON al.artifact_id = a.artifact_id
JOIN storage_path AS sp ON sp.path_id = al.path_id
WHERE a.project_id = 'MIGRATIONBENCH'
  AND a.artifact_key LIKE 'MIGRATIONBENCH-ROCKFISH-ARCHIVE:20261124:%'
  AND sp.root_id = 'RF-SCRATCH16-YI'
ORDER BY a.artifact_kind, sp.relative_path;

-- 13. Retrieval index for manuscript-alignment, archive, cleanup, and SQL
-- reports registered as audit deliverables in the local unified worktree.
SELECT a.artifact_id, a.artifact_key, a.artifact_kind, a.sha256, a.byte_size,
       al.availability, sp.root_id, sp.relative_path, a.notes
FROM artifact AS a
JOIN artifact_location AS al ON al.artifact_id = a.artifact_id
JOIN storage_path AS sp ON sp.path_id = al.path_id
WHERE a.project_id = 'MIGRATIONBENCH'
  AND a.artifact_key LIKE 'audit_deliverable:%'
  AND (a.artifact_key LIKE '%MANUSCRIPT_REVISION1_ALIGNMENT_AUDIT%'
       OR a.artifact_key LIKE '%revision1_git_large_assets_20261124%'
       OR a.artifact_key LIKE '%pr_release_candidates_20261124%'
       OR a.artifact_key LIKE '%wfc_mix_residual_20261124%')
ORDER BY sp.relative_path;
