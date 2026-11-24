-- Submitted Revision 1 result and evidence index.
SELECT i.source_item_id, i.section, i.panel, i.content, i.source_status,
       i.reproduction_status, i.final_asset_path, i.final_asset_sha256,
       e.evidence_role, e.source_path, e.availability, e.artifact_id,
       e.notes AS evidence_notes
FROM paper_release_item AS i
LEFT JOIN paper_release_item_evidence AS e USING (release_item_id)
WHERE i.release_id = 'MB-REVISION1-SUBMITTED-20261110'
ORDER BY i.section, i.source_item_id, i.panel, e.evidence_role;

-- Source reproducibility labels remain separate from conservative assessment.
SELECT source_item_id, section, item_dir, source_result, assessment_status,
       checks_count, inputs_count, scripts, evidence, problems, notes
FROM paper_reproducibility_record
WHERE release_id = 'MB-REVISION1-SUBMITTED-20261110'
ORDER BY section, source_item_id;

-- Reviewer response claim rows: original status, normalized catalog claim,
-- exact source paths, response locations, and stated limits.
SELECT r.source_claim_id, r.reviewer, r.question, r.source_scientific_status,
       c.scientific_status AS catalog_status, c.publication_action,
       r.primary_artifact, r.source_or_table, r.rebuild_script,
       r.manuscript_location, r.response_location, r.boundary
FROM paper_release_claim_observation AS r
LEFT JOIN paper_claim AS c ON c.claim_id = r.catalog_claim_id
WHERE r.release_id = 'MB-REVISION1-SUBMITTED-20261110'
ORDER BY r.reviewer, r.source_claim_id;

-- The in-gap DFT anchor: source trace is preserved with the source's own
-- SCF-warning / EXCLUDE_FROM_PUBLICATION gate.
SELECT c.claim_id, c.claim_text, c.scientific_status, c.publication_action,
       a.artifact_key, a.sha256 AS artifact_sha256,
       e.evidence_role, e.support_status, e.notes
FROM paper_claim AS c
JOIN claim_evidence AS e USING (claim_id)
JOIN artifact AS a USING (artifact_id)
WHERE c.claim_id = 'MB-R1-MAIN-IN-GAP-DFT-BARRIER-1P52EV';

-- QE restart deletions: evidence, exact file hashes, user authorization,
-- coordinate XML, scheduler context, queue guard, and image coverage.
SELECT ca.run_id, r.run_label, ca.candidate_kind,
       COUNT(*) AS file_count, SUM(a.byte_size) AS logical_bytes,
       SUM(inv.allocated_bytes) AS allocated_bytes,
       MIN(ca.coordinate_nat) AS min_atoms, MAX(ca.coordinate_nat) AS max_atoms,
       SUM(ca.neb_complete_image_coverage) AS files_with_complete_neb_paths,
       MIN(ca.source_queue_capture_utc) AS source_queue_capture_utc,
       GROUP_CONCAT(DISTINCT ca.manifest_restart_required_flag) AS manifest_flags,
       GROUP_CONCAT(DISTINCT ci.deletion_state) AS deletion_states
FROM cleanup_candidate_context AS ca
JOIN artifact AS a USING (artifact_id)
JOIN cleanup_item AS ci USING (cleanup_batch_id, artifact_id)
JOIN workflow_run AS r USING (run_id)
LEFT JOIN inventory_observation AS inv
       ON inv.path_id = (SELECT al.path_id FROM artifact_location AS al
                         WHERE al.artifact_id = a.artifact_id LIMIT 1)
WHERE ca.cleanup_batch_id = 'RF-SCRATCH16-YI-REV1-QE-RESTART-CLEANUP-20261123'
GROUP BY ca.run_id, r.run_label, ca.candidate_kind
ORDER BY ca.run_id, ca.candidate_kind;

-- Compact batch receipt after deletion is finalized.
SELECT b.cleanup_batch_id, b.status, b.expected_bytes, b.actual_bytes,
       r.execution_state, r.deleted_file_count, r.actual_logical_bytes,
       r.actual_allocated_bytes, r.observed_utc, r.notes
FROM cleanup_batch AS b
LEFT JOIN cleanup_execution_receipt AS r USING (cleanup_batch_id)
WHERE b.cleanup_batch_id = 'RF-SCRATCH16-YI-REV1-QE-RESTART-CLEANUP-20261123';

-- Six grouped-A2 MD runs are an explicitly partial subset of the 15-model
-- Fig. 6 / Table S8 / Table R7 cohort. Retrieve their archived result package,
-- hashes, exact Rockfish paths, source relationship, and scope boundary.
SELECT i.source_item_id, i.section, i.panel, i.content,
       e.evidence_role, e.source_path, e.availability, e.relationship, e.notes,
       a.artifact_kind, a.semantic_role, a.sha256, a.byte_size,
       sp.root_id, sp.relative_path, al.last_verified_utc
FROM paper_release_item AS i
JOIN paper_release_item_evidence AS e USING (release_item_id)
LEFT JOIN artifact AS a ON a.artifact_id = e.artifact_id
LEFT JOIN artifact_location AS al ON al.artifact_id = a.artifact_id
LEFT JOIN storage_path AS sp
  ON sp.path_id = al.path_id
 AND sp.root_id = 'RF-SCRATCH16-YI'
 AND sp.relative_path LIKE 'MigrationBench_data_archive/revision1_md_fig6_table_s8_20261124/%'
WHERE i.release_id = 'MB-REVISION1-SUBMITTED-20261110'
  AND i.source_item_id IN ('Fig6','Table S8','Table R7')
  AND e.source_path LIKE '%revision1_md_fig6_table_s8_20261124%'
ORDER BY i.source_item_id, i.section, i.panel, e.evidence_role, e.source_path;
