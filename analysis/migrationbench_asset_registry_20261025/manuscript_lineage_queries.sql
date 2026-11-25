-- Read-only examples for migrations/004_revision1_claim_lineage_graph.sql.
-- Run against a disposable registry copy after applying migrations 001-004.

-- QL01. High-level claim closure counts. A status can remain PARTIAL even when
-- one or more pointers exist; no query result is a scientific validity score.
SELECT closure_status, claim_review_status, scientific_status, COUNT(*) AS n_claims
FROM v_r1_claim_lineage_status
GROUP BY closure_status, claim_review_status, scientific_status
ORDER BY closure_status, claim_review_status, scientific_status;

-- QL02. Claims without an explicit claim->result link, preserving NOT_RUN and
-- other authored statuses instead of interpreting an empty join as a failure.
SELECT c.claim_id, ms.document_role, ms.revision_label, c.section_label,
       c.line_start, c.line_end, c.exact_text, c.evidence_status,
       c.scientific_status, c.claim_review_status
FROM manuscript_claims c
JOIN manuscript_sources ms USING (manuscript_source_id)
LEFT JOIN claim_result_links crl USING (claim_id)
WHERE crl.result_id IS NULL
ORDER BY ms.document_role, c.line_start;

-- QL03. Figure/table/panel/cell slots and their existing legacy registry key.
-- The new graph can coexist with the 52 object rows in rev1_results_index.
SELECT *
FROM v_r1_object_lineage_status
ORDER BY object_kind, object_key, cell_key;

-- QL04. Drill a specific panel/cell into result -> run -> raw/derived artifact.
-- Bind :object_key to a stable manuscript_objects.object_key.
SELECT o.object_key, o.object_kind, o.label,
       r.result_key, r.source_row_key, r.result_grain, r.protocol_key,
       r.evidence_status AS result_evidence_status,
       run.run_key, run.run_kind, run.protocol_key AS run_protocol,
       run.run_status, run.scientific_status AS run_scientific_status,
       run.model_checkpoint_sha256, run.seed, run.sample_key, run.scheduler_job_id,
       a.artifact_key, a.artifact_role, a.source_path, a.path_scope,
       a.sha256, a.byte_size, a.availability_status
FROM manuscript_objects o
JOIN object_result_links orl USING (manuscript_object_id)
JOIN lineage_results r USING (result_id)
LEFT JOIN result_run_links rrl USING (result_id)
LEFT JOIN lineage_runs run USING (run_id)
LEFT JOIN run_artifact_links ral USING (run_id)
LEFT JOIN lineage_artifacts a USING (artifact_id)
WHERE o.object_key = :object_key
ORDER BY r.result_key, run.run_key, a.artifact_role, a.source_path;

-- QL05. A claim and all typed pointers. Do not infer identity from names, seeds,
-- folder labels, or a neighboring panel; require exact SHA for model joins.
SELECT c.claim_id, c.exact_text, c.evidence_status AS claim_evidence_status,
       col.relation AS claim_object_relation, o.object_key,
       crl.relation AS claim_result_relation, r.result_key, r.source_row_key,
       r.model_checkpoint_sha256 AS result_checkpoint_sha256,
       rrl.relation AS result_run_relation, run.run_key,
       run.model_checkpoint_sha256 AS run_checkpoint_sha256,
       ral.artifact_role, a.source_path, a.sha256 AS artifact_sha256,
       CASE WHEN r.model_checkpoint_sha256 <> ''
                  AND r.model_checkpoint_sha256 = run.model_checkpoint_sha256
            THEN 'EXACT_SHA_MATCH'
            WHEN r.model_checkpoint_sha256 = '' OR run.model_checkpoint_sha256 = ''
            THEN 'NOT_LINKED'
            ELSE 'HASH_CONFLICT' END AS checkpoint_join_status
FROM manuscript_claims c
LEFT JOIN claim_object_links col USING (claim_id)
LEFT JOIN manuscript_objects o USING (manuscript_object_id)
LEFT JOIN claim_result_links crl USING (claim_id)
LEFT JOIN lineage_results r USING (result_id)
LEFT JOIN result_run_links rrl USING (result_id)
LEFT JOIN lineage_runs run USING (run_id)
LEFT JOIN run_artifact_links ral USING (run_id)
LEFT JOIN lineage_artifacts a USING (artifact_id)
WHERE c.claim_id = :claim_id
ORDER BY o.object_key, r.result_key, run.run_key, a.source_path;

-- QL06. Open figure/table alignment issues (caption mismatch, source release
-- mismatch, missing export, or scientific-status conflict).
SELECT o.object_key, o.object_kind, o.label, i.issue_kind, i.issue_status,
       i.description, i.resolution_note, i.opened_at
FROM manuscript_object_issues i
JOIN manuscript_objects o USING (manuscript_object_id)
WHERE i.issue_status IN ('OPEN', 'IN_REVIEW')
ORDER BY i.issue_kind, o.object_key;

-- QL07. Keep all run outcome classes visible, including failed, held, partial,
-- not-run and cancelled attempts. Protocol is an explicit field.
SELECT run_kind, protocol_key, run_status, scientific_status,
       COUNT(*) AS n_runs,
       COUNT(DISTINCT model_checkpoint_sha256) AS n_exact_checkpoint_hashes
FROM lineage_runs
GROUP BY run_kind, protocol_key, run_status, scientific_status
ORDER BY run_kind, protocol_key, run_status;

-- QL08. Failure-mode signals and repeatability scope. STABLE is an evidence
-- state recorded by reviewers, not inferred from repeated filenames.
SELECT failure_mode_key, signal_key, detection_status, replication_status,
       replication_scope, evidence_status, COUNT(*) AS n_events
FROM diagnostic_events
GROUP BY failure_mode_key, signal_key, detection_status, replication_status,
         replication_scope, evidence_status
ORDER BY failure_mode_key, signal_key;

-- QL09. A diagnostic->proposed action->next run->outcome chain. Gain remains
-- NOT_EVALUATED unless baseline, follow-up, result, protocol and gate are linked.
SELECT d.failure_mode_key, d.signal_key, d.detection_status,
       d.replication_status, d.replication_scope,
       ald.proposed_action, ald.action_status,
       baseline.run_key AS baseline_run,
       next_run.run_key AS followup_run,
       r.result_key AS followup_result,
       ald.evaluation_protocol, ald.acceptance_gate,
       ald.measured_gain_value, ald.measured_gain_unit,
       ald.measured_gain_status, ald.evidence_status
FROM active_learning_decisions ald
JOIN diagnostic_events d USING (diagnostic_id)
LEFT JOIN lineage_runs baseline ON baseline.run_id=ald.baseline_run_id
LEFT JOIN lineage_runs next_run ON next_run.run_id=ald.next_run_id
LEFT JOIN lineage_results r ON r.result_id=ald.outcome_result_id
ORDER BY d.failure_mode_key, d.signal_key, ald.decision_key;

-- QL10. Exact artifact-hash collisions at multiple storage locations; this
-- surfaces candidates for review but never authorizes deletion automatically.
SELECT sha256, COUNT(DISTINCT source_path) AS n_paths,
       GROUP_CONCAT(source_path, char(10)) AS paths,
       MIN(byte_size) AS min_bytes, MAX(byte_size) AS max_bytes
FROM lineage_artifacts
WHERE sha256 <> ''
GROUP BY sha256
HAVING COUNT(DISTINCT source_path) > 1
ORDER BY n_paths DESC, sha256;
