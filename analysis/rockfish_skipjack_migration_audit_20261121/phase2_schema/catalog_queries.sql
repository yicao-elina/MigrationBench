-- Read-only query patterns for provenance_catalog.sqlite.
-- Bind :project_id, :run_id, :claim_id, or :cleanup_batch_id as needed.

-- 1. Project run and recipe index. One row per run/recipe version; Slurm jobs
-- and run artifacts are separate result sets below to avoid join fan-out.
SELECT p.project_id, p.project_name, wr.run_id, wr.run_label,
       wr.workflow_type, wr.method, wr.scheduler_state, wr.scientific_status,
       wr.source_root_id, wr.source_relative_path,
       rr.recipe_id, rr.recipe_version, rr.recipe_status, rr.exact_command,
       rr.working_directory, rr.executable_version, rr.environment_description,
       rr.module_list, rr.random_seed, rr.validation_note,
       launch.artifact_key AS launch_script_key,
       (SELECT group_concat(r.scoped_path || '/' || s.relative_path, ' | ')
        FROM artifact_location AS l JOIN storage_path AS s USING(path_id)
        JOIN storage_root AS r USING(root_id)
        WHERE l.artifact_id = launch.artifact_id) AS launch_script_paths,
       batch.artifact_key AS slurm_script_key,
       (SELECT group_concat(r.scoped_path || '/' || s.relative_path, ' | ')
        FROM artifact_location AS l JOIN storage_path AS s USING(path_id)
        JOIN storage_root AS r USING(root_id)
        WHERE l.artifact_id = batch.artifact_id) AS slurm_script_paths,
       manifest.artifact_key AS input_manifest_key,
       (SELECT group_concat(r.scoped_path || '/' || s.relative_path, ' | ')
        FROM artifact_location AS l JOIN storage_path AS s USING(path_id)
        JOIN storage_root AS r USING(root_id)
        WHERE l.artifact_id = manifest.artifact_id) AS input_manifest_paths
FROM workflow_run AS wr
JOIN project AS p ON p.project_id = wr.project_id
LEFT JOIN reproduction_recipe AS rr ON rr.run_id = wr.run_id
LEFT JOIN artifact AS launch ON launch.artifact_id = rr.script_artifact_id
LEFT JOIN artifact AS batch ON batch.artifact_id = rr.scheduler_script_artifact_id
LEFT JOIN artifact AS manifest ON manifest.artifact_id = rr.input_manifest_artifact_id
WHERE p.project_id = :project_id
ORDER BY wr.run_id, rr.recipe_version;

-- 1b. Exact run inputs, outputs, logs, and provenance artifacts. One row per
-- run-artifact-location; no recipe or Slurm join is made here.
SELECT wr.run_id, wr.run_label, ra.direction, ra.role,
       ra.required_for_reproduction, a.artifact_key, a.artifact_kind,
       a.semantic_role, a.sha256, a.byte_size, al.availability,
       sr.scoped_path || '/' || sp.relative_path AS artifact_path, ra.notes
FROM workflow_run AS wr
JOIN run_artifact AS ra ON ra.run_id = wr.run_id
JOIN artifact AS a ON a.artifact_id = ra.artifact_id
LEFT JOIN artifact_location AS al ON al.artifact_id = a.artifact_id
LEFT JOIN storage_path AS sp ON sp.path_id = al.path_id
LEFT JOIN storage_root AS sr ON sr.root_id = sp.root_id
WHERE wr.project_id = :project_id
ORDER BY wr.run_id, ra.direction, ra.role, a.artifact_key, artifact_path;

-- 1c. Scheduler history. One row per Slurm job instance with latest state.
SELECT wr.run_id, wr.run_label, sj.cluster_name, sj.job_id,
       sj.array_task_id, sj.submitted_or_started_utc, sj.job_name,
       sj.partition_name, sj.account_name,
       (SELECT so.state FROM slurm_state_observation AS so
        WHERE so.slurm_job_key = sj.slurm_job_key
        ORDER BY so.observed_utc DESC LIMIT 1) AS latest_state,
       (SELECT so.observed_utc FROM slurm_state_observation AS so
        WHERE so.slurm_job_key = sj.slurm_job_key
        ORDER BY so.observed_utc DESC LIMIT 1) AS latest_observation_utc
FROM workflow_run AS wr
JOIN slurm_job AS sj ON sj.run_id = wr.run_id
WHERE wr.project_id = :project_id
ORDER BY wr.run_id, sj.submitted_or_started_utc, sj.job_id, sj.array_task_id;

-- 2. One paper claim through its explicitly registered figure and source assets.
SELECT pc.claim_id, pc.manuscript_path, pc.section_or_panel, pc.claim_text,
       pc.scientific_status, pc.publication_action,
       pf.figure_id, pf.title, pcf.relationship AS claim_figure_relationship,
       pfa.asset_role, pfa.source_path, pfa.availability AS asset_availability,
       a.artifact_key, a.sha256, a.byte_size
FROM paper_claim AS pc
LEFT JOIN paper_claim_figure AS pcf ON pcf.claim_id = pc.claim_id
LEFT JOIN paper_figure AS pf ON pf.figure_id = pcf.figure_id
LEFT JOIN paper_figure_asset AS pfa ON pfa.figure_id = pf.figure_id
LEFT JOIN artifact AS a ON a.artifact_id = pfa.artifact_id
WHERE pc.claim_id = :claim_id
ORDER BY pf.figure_id, pfa.asset_role, pfa.source_path;

-- 3. Cleanup item ledger. One row per item and observed artifact location.
SELECT cb.cleanup_batch_id, cb.status AS batch_status, cb.policy,
       cb.authorization_note, ci.disposition, ci.owner_clearance,
       ci.decision_reason, ci.deletion_state, ci.pre_delete_sha256,
       ci.pre_delete_size_bytes, ci.pre_delete_mtime_ns, ci.deleted_utc,
       ci.actual_bytes, a.artifact_key, a.retention_class, al.availability,
       sr.scoped_path || '/' || sp.relative_path AS artifact_path, ci.notes
FROM cleanup_batch AS cb
JOIN cleanup_item AS ci ON ci.cleanup_batch_id = cb.cleanup_batch_id
JOIN artifact AS a ON a.artifact_id = ci.artifact_id
LEFT JOIN artifact_location AS al ON al.artifact_id = a.artifact_id
LEFT JOIN storage_path AS sp ON sp.path_id = al.path_id
LEFT JOIN storage_root AS sr ON sr.root_id = sp.root_id
WHERE cb.cleanup_batch_id = :cleanup_batch_id
ORDER BY a.artifact_key, artifact_path;

-- 3b. Batch execution receipts and quota/queue evidence. One row per receipt;
-- item rows are intentionally kept in query 3 to prevent N×receipt fan-out.
SELECT rr.cleanup_batch_id, rr.cleanup_execution_receipt_id,
       rr.observed_utc, rr.execution_state, rr.deleted_file_count,
       rr.actual_logical_bytes, rr.actual_allocated_bytes, rr.notes,
       qb.observed_utc AS quota_before_utc, qb.used_bytes AS quota_before_bytes,
       qa.observed_utc AS quota_after_utc, qa.used_bytes AS quota_after_bytes,
       qs.source_path AS queue_capture_path, qs.sha256 AS queue_capture_sha256,
       rs.source_path AS receipt_path, rs.sha256 AS receipt_sha256
FROM cleanup_execution_receipt AS rr
LEFT JOIN quota_observation AS qb
  ON qb.quota_observation_id = rr.quota_before_observation_id
LEFT JOIN quota_observation AS qa
  ON qa.quota_observation_id = rr.quota_after_observation_id
LEFT JOIN registry_source AS qs ON qs.source_id = rr.queue_snapshot_source_id
LEFT JOIN registry_source AS rs ON rs.source_id = rr.receipt_source_id
WHERE rr.cleanup_batch_id = :cleanup_batch_id
ORDER BY rr.observed_utc, rr.cleanup_execution_receipt_id;

-- 4. Most recent inventory snapshot for a storage root and its completeness.
WITH latest AS (
  SELECT root_id, MAX(COALESCE(finished_utc, started_utc)) AS observed_utc
  FROM inventory_snapshot
  GROUP BY root_id
)
SELECT sr.root_id, sr.scoped_path, i.snapshot_id, i.started_utc,
       i.finished_utc, i.completion_state, i.files_seen, i.directories_seen,
       i.symlinks_seen, i.unreadable_count, i.logical_bytes, i.allocated_bytes,
       i.manifest_path, i.manifest_sha256, i.notes
FROM latest AS x
JOIN inventory_snapshot AS i
  ON i.root_id = x.root_id
 AND COALESCE(i.finished_utc, i.started_utc) = x.observed_utc
JOIN storage_root AS sr ON sr.root_id = i.root_id
WHERE sr.root_id = :root_id;

-- 5. Search project artifacts by path or semantic content without inferring
-- ownership or scientific status from the filename.
SELECT p.project_id, p.project_name, a.artifact_key, a.artifact_kind,
       a.semantic_role, a.owner_status, a.scientific_status,
       a.retention_class, a.sha256, a.byte_size, al.availability,
       sr.scoped_path || '/' || sp.relative_path AS artifact_path,
       a.notes
FROM artifact AS a
LEFT JOIN project AS p ON p.project_id = a.project_id
LEFT JOIN artifact_location AS al ON al.artifact_id = a.artifact_id
LEFT JOIN storage_path AS sp ON sp.path_id = al.path_id
LEFT JOIN storage_root AS sr ON sr.root_id = sp.root_id
WHERE a.project_id = :project_id
  AND (a.artifact_key LIKE '%' || :term || '%'
       OR a.semantic_role LIKE '%' || :term || '%'
       OR sp.relative_path LIKE '%' || :term || '%'
       OR a.notes LIKE '%' || :term || '%')
ORDER BY a.retention_class, a.byte_size DESC, a.artifact_key;

-- 6. Exact recipe inputs for one run. One row per recipe input/location;
-- scheduler history is query 1c and run outputs/logs are query 1b.
SELECT wr.run_id, rr.recipe_id, rr.recipe_version, rr.recipe_status,
       rr.exact_command, rr.working_directory, rr.executable_version,
       rr.environment_description, rr.module_list, rr.random_seed,
       ri.input_role, a.artifact_key, a.sha256, a.byte_size,
       al.availability, sr.scoped_path || '/' || sp.relative_path AS input_path
FROM workflow_run AS wr
JOIN reproduction_recipe AS rr ON rr.run_id = wr.run_id
LEFT JOIN recipe_input AS ri ON ri.recipe_id = rr.recipe_id
LEFT JOIN artifact AS a ON a.artifact_id = ri.artifact_id
LEFT JOIN artifact_location AS al ON al.artifact_id = a.artifact_id
LEFT JOIN storage_path AS sp ON sp.path_id = al.path_id
LEFT JOIN storage_root AS sr ON sr.root_id = sp.root_id
WHERE wr.run_id = :run_id
ORDER BY rr.recipe_version, ri.input_role, a.artifact_key, input_path;

-- 7. Claim -> explicitly linked figure -> file/pointer assets. This result set
-- does not imply a producing run; use query 8 only where artifact IDs link.
SELECT pc.claim_id, pc.manuscript_path, pc.section_or_panel, pc.claim_text,
       pc.scientific_status AS claim_status, pc.publication_action,
       pcf.relationship AS claim_figure_relationship,
       pf.figure_id, pf.title, pf.chart_status, pf.provenance_status,
       pfa.figure_asset_id, pfa.asset_role, pfa.relationship AS asset_link,
       pfa.source_path, pfa.availability AS asset_availability,
       a.artifact_key, a.sha256, a.byte_size, al.availability,
       sr.scoped_path || '/' || sp.relative_path AS artifact_path
FROM paper_claim AS pc
LEFT JOIN paper_claim_figure AS pcf ON pcf.claim_id = pc.claim_id
LEFT JOIN paper_figure AS pf ON pf.figure_id = pcf.figure_id
LEFT JOIN paper_figure_asset AS pfa ON pfa.figure_id = pf.figure_id
LEFT JOIN artifact AS a ON a.artifact_id = pfa.artifact_id
LEFT JOIN artifact_location AS al ON al.artifact_id = a.artifact_id
LEFT JOIN storage_path AS sp ON sp.path_id = al.path_id
LEFT JOIN storage_root AS sr ON sr.root_id = sp.root_id
WHERE pc.claim_id = :claim_id
ORDER BY pf.figure_id, pfa.asset_role, pfa.source_path, artifact_path;

-- 8. Follow an exact figure-output artifact through recorded derivation edges
-- to its input artifacts and producing runs. The recursion is cycle-guarded
-- and capped at 20 edges; filename/pointer matches are not used.
WITH RECURSIVE lineage AS (
  SELECT pc.claim_id, pf.figure_id, pfa.figure_asset_id,
         pfa.artifact_id AS artifact_id, NULL AS derivation_id,
         0 AS depth, printf('/%d/', pfa.artifact_id) AS visited
  FROM paper_claim AS pc
  JOIN paper_claim_figure AS pcf ON pcf.claim_id = pc.claim_id
  JOIN paper_figure AS pf ON pf.figure_id = pcf.figure_id
  JOIN paper_figure_asset AS pfa ON pfa.figure_id = pf.figure_id
  WHERE pc.claim_id = :claim_id AND pfa.artifact_id IS NOT NULL
  UNION ALL
  SELECT lineage.claim_id, lineage.figure_id, lineage.figure_asset_id,
         edge.parent_artifact_id, edge.derivation_id,
         lineage.depth + 1,
         lineage.visited || edge.parent_artifact_id || '/'
  FROM lineage
  JOIN derivation_edge AS edge ON edge.child_artifact_id = lineage.artifact_id
  WHERE lineage.depth < 20
    AND instr(lineage.visited, printf('/%d/', edge.parent_artifact_id)) = 0
)
SELECT lineage.claim_id, lineage.figure_id, lineage.figure_asset_id,
       lineage.depth, child.artifact_key AS derived_child_artifact,
       parent.artifact_key AS source_parent_artifact,
       edge.operation, edge.tool_name, edge.tool_version, edge.parameter_json,
       edge.provenance_status AS derivation_status,
       wr.run_id, wr.run_label, wr.workflow_type, wr.scientific_status AS run_status,
       rr.recipe_version, rr.recipe_status, rr.exact_command,
       launch.artifact_key AS launch_script_key,
       batch.artifact_key AS slurm_script_key,
       edge_script.artifact_key AS derivation_script_key,
       rg.commit_sha, rg.binding_role, rg.provenance_status AS git_binding_status
FROM lineage
JOIN derivation_edge AS edge ON edge.derivation_id = lineage.derivation_id
JOIN artifact AS parent ON parent.artifact_id = edge.parent_artifact_id
JOIN artifact AS child ON child.artifact_id = edge.child_artifact_id
LEFT JOIN workflow_run AS wr ON wr.run_id = edge.run_id
LEFT JOIN reproduction_recipe AS rr ON rr.run_id = wr.run_id
LEFT JOIN artifact AS launch ON launch.artifact_id = rr.script_artifact_id
LEFT JOIN artifact AS batch ON batch.artifact_id = rr.scheduler_script_artifact_id
LEFT JOIN artifact AS edge_script ON edge_script.artifact_id = edge.script_artifact_id
LEFT JOIN run_git_binding AS rg ON rg.run_id = wr.run_id
WHERE lineage.derivation_id IS NOT NULL
ORDER BY lineage.figure_id, lineage.figure_asset_id, lineage.depth,
         edge.derivation_id, rg.commit_sha;

-- 9. Coverage gaps in the project-to-run-to-recipe chain. This is a review
-- queue, not an inferred mapping. Each subquery is aggregated before joining
-- so missing relationships remain visible without multiplying rows.
WITH recipe_coverage AS (
  SELECT run_id, COUNT(*) AS recipe_count,
         SUM(exact_command IS NOT NULL AND trim(exact_command) <> '') AS command_count,
         SUM(script_artifact_id IS NOT NULL) AS launch_script_count,
         SUM(scheduler_script_artifact_id IS NOT NULL) AS slurm_script_count,
         SUM(input_manifest_artifact_id IS NOT NULL) AS manifest_count,
         SUM(recipe_status = 'REPRODUCED') AS reproduced_count
  FROM reproduction_recipe
  GROUP BY run_id
), artifact_coverage AS (
  SELECT run_id,
         SUM(direction = 'INPUT') AS input_link_count,
         SUM(direction = 'OUTPUT') AS output_link_count
  FROM run_artifact
  GROUP BY run_id
), slurm_coverage AS (
  SELECT run_id, COUNT(*) AS job_count FROM slurm_job GROUP BY run_id
), git_coverage AS (
  SELECT run_id, COUNT(*) AS binding_count FROM run_git_binding GROUP BY run_id
)
SELECT p.project_id, p.project_name, wr.run_id, wr.run_label,
       wr.workflow_type, wr.scientific_status,
       COALESCE(rc.recipe_count, 0) AS recipe_count,
       COALESCE(rc.command_count, 0) AS recipes_with_exact_command,
       COALESCE(rc.launch_script_count, 0) AS recipes_with_launch_script,
       COALESCE(rc.slurm_script_count, 0) AS recipes_with_slurm_script,
       COALESCE(rc.manifest_count, 0) AS recipes_with_input_manifest,
       COALESCE(rc.reproduced_count, 0) AS reproduced_recipes,
       COALESCE(ac.input_link_count, 0) AS run_input_links,
       COALESCE(ac.output_link_count, 0) AS run_output_links,
       COALESCE(sc.job_count, 0) AS slurm_job_instances,
       COALESCE(gc.binding_count, 0) AS git_commit_bindings
FROM workflow_run AS wr
JOIN project AS p ON p.project_id = wr.project_id
LEFT JOIN recipe_coverage AS rc ON rc.run_id = wr.run_id
LEFT JOIN artifact_coverage AS ac ON ac.run_id = wr.run_id
LEFT JOIN slurm_coverage AS sc ON sc.run_id = wr.run_id
LEFT JOIN git_coverage AS gc ON gc.run_id = wr.run_id
WHERE COALESCE(rc.recipe_count, 0) = 0
   OR COALESCE(rc.command_count, 0) < COALESCE(rc.recipe_count, 0)
   OR COALESCE(rc.launch_script_count, 0) < COALESCE(rc.recipe_count, 0)
   OR COALESCE(rc.slurm_script_count, 0) < COALESCE(rc.recipe_count, 0)
   OR COALESCE(rc.manifest_count, 0) < COALESCE(rc.recipe_count, 0)
   OR COALESCE(ac.input_link_count, 0) = 0
   OR COALESCE(ac.output_link_count, 0) = 0
   OR COALESCE(sc.job_count, 0) = 0
   OR COALESCE(gc.binding_count, 0) = 0
ORDER BY p.project_id, wr.run_id;

-- 9b. Claims with no explicit claim-to-figure relationship. A title, path, or
-- nearby artifact name is deliberately not used to fill this gap.
SELECT pc.claim_id, pc.manuscript_path, pc.section_or_panel, pc.claim_text,
       pc.scientific_status, pc.publication_action
FROM paper_claim AS pc
WHERE NOT EXISTS (
  SELECT 1 FROM paper_claim_figure AS pcf WHERE pcf.claim_id = pc.claim_id
)
ORDER BY pc.claim_id;

-- 10. Start with an exact inventoried path or a path prefix, then follow only
-- explicit artifact, project, run, and recipe links. Unregistered links stay
-- NULL; use :root_id and :relative_prefix (the latter may be an exact path).
WITH latest AS (
  SELECT snapshot_id
  FROM inventory_snapshot
  WHERE root_id = :root_id
  ORDER BY COALESCE(finished_utc, started_utc) DESC, snapshot_id DESC
  LIMIT 1
)
SELECT sr.root_id, sr.scoped_path, i.snapshot_id, i.completion_state,
       sp.relative_path, io.entry_type, io.logical_bytes, io.allocated_bytes,
       io.sha256, io.hash_scope, io.observation_status,
       a.artifact_key, a.artifact_kind, a.semantic_role, a.retention_class,
       p.project_id, p.project_name, wr.run_id, wr.run_label,
       ra.direction, ra.role AS run_artifact_role,
       rr.recipe_id, rr.recipe_version, rr.recipe_status, rr.exact_command,
       rr.working_directory,
       (SELECT group_concat(sj.cluster_name || ':' || sj.job_id ||
                            COALESCE('.' || sj.array_task_id, ''), ' | ')
        FROM slurm_job AS sj WHERE sj.run_id = wr.run_id) AS slurm_jobs
FROM latest AS latest_snapshot
JOIN inventory_snapshot AS i ON i.snapshot_id = latest_snapshot.snapshot_id
JOIN inventory_observation AS io ON io.snapshot_id = i.snapshot_id
JOIN storage_path AS sp ON sp.path_id = io.path_id
JOIN storage_root AS sr ON sr.root_id = sp.root_id
LEFT JOIN artifact_location AS al ON al.path_id = sp.path_id
LEFT JOIN artifact AS a ON a.artifact_id = al.artifact_id
LEFT JOIN project AS p ON p.project_id = a.project_id
LEFT JOIN run_artifact AS ra ON ra.artifact_id = a.artifact_id
LEFT JOIN workflow_run AS wr ON wr.run_id = ra.run_id
LEFT JOIN reproduction_recipe AS rr ON rr.run_id = wr.run_id
WHERE sp.root_id = :root_id
  AND (sp.relative_path = :relative_prefix
       OR sp.relative_path LIKE :relative_prefix || '%')
ORDER BY sp.relative_path, a.artifact_key, wr.run_id, rr.recipe_version;

-- 11. Cleanup evidence sources for a batch, including the guarded executor,
-- independent post-delete scan, and storage/quota readout. The relationships
-- are explicit registry links; this query does not treat filesystem-wide df
-- output as a personal quota measurement.
SELECT cb.cleanup_batch_id, cb.status, cb.expected_bytes, cb.actual_bytes,
       rr.execution_state, rr.deleted_file_count, rr.actual_logical_bytes,
       rr.actual_allocated_bytes, rr.notes AS execution_notes,
       re.relationship, rs.registry_kind, rs.source_path, rs.sha256,
       rs.as_of_utc, rs.notes AS source_notes
FROM cleanup_batch AS cb
LEFT JOIN cleanup_execution_receipt AS rr
  ON rr.cleanup_batch_id = cb.cleanup_batch_id
LEFT JOIN registry_entity_ref AS re
  ON re.entity_type IN ('CLEANUP_BATCH', 'CLEANUP_EVIDENCE')
 AND (re.entity_key = cb.cleanup_batch_id
      OR re.entity_key LIKE cb.cleanup_batch_id || ':%')
 AND re.catalog_entity_type = 'CLEANUP_BATCH'
 AND re.catalog_entity_key = cb.cleanup_batch_id
LEFT JOIN registry_source AS rs ON rs.source_id = re.source_id
WHERE cb.cleanup_batch_id = :cleanup_batch_id
ORDER BY rs.registry_kind, rs.source_path;

-- 12. Summary of the verified Revision 1 QE restart deletion batches.
-- This reports pre-delete logical sizes and state transitions, not a personal
-- quota measurement or a hash of each restart payload.
SELECT cleanup_batch_id, disposition, deletion_state, count(*) AS file_count,
       sum(pre_delete_size_bytes) AS pre_delete_logical_bytes,
       sum(actual_bytes) AS verified_removed_bytes
FROM cleanup_item
WHERE cleanup_batch_id IN (
  'RF-SCRATCH16-YI-REV1-574-RESTART-CLEANUP-20261122-V1',
  'RF-SCRATCH16-YI-REV1-QE-RESTART-CLEANUP-20261123',
  'RF-YI-QE-RESTART-CLEANUP-20261123'
)
GROUP BY cleanup_batch_id, disposition, deletion_state
ORDER BY cleanup_batch_id, disposition, deletion_state;

-- 13. Exact path-level disposition for the later user-directed revision1
-- batch: 3,817 WFC plus 481 mixing restart files.
SELECT ci.cleanup_batch_id, sr.root_id, sr.scoped_path || '/' || sp.relative_path AS full_path,
       a.artifact_kind, ci.disposition, ci.deletion_state,
       ci.pre_delete_size_bytes, ci.actual_bytes, ci.pre_delete_sha256,
       ci.deleted_utc, ci.notes
FROM cleanup_item AS ci
JOIN artifact AS a USING(artifact_id)
JOIN artifact_location AS al USING(artifact_id)
JOIN storage_path AS sp USING(path_id)
JOIN storage_root AS sr USING(root_id)
WHERE ci.cleanup_batch_id = 'RF-SCRATCH16-YI-REV1-QE-RESTART-CLEANUP-20261123'
ORDER BY full_path;

-- 14. Restart paths removed from the 81-atom NEB tree; coordinate evidence
-- remains separate from the restart deletion state.
SELECT ci.cleanup_batch_id, sr.root_id, sr.scoped_path || '/' || sp.relative_path AS full_path,
       a.artifact_kind, ci.disposition, ci.deletion_state,
       ci.pre_delete_size_bytes, ci.actual_bytes, ci.notes
FROM cleanup_item AS ci
JOIN artifact AS a USING(artifact_id)
JOIN artifact_location AS al USING(artifact_id)
JOIN storage_path AS sp USING(path_id)
JOIN storage_root AS sr USING(root_id)
WHERE ci.cleanup_batch_id = 'RF-YI-QE-RESTART-CLEANUP-20261123'
  AND sp.relative_path LIKE '%cr_81atom_neb_path_restart_fix%'
ORDER BY full_path;

-- 15. The eight empty Cr-in-gap QE mixing placeholders were deleted in a
-- separate receipt after the earlier coordinate-supported batch recorded
-- them as HOLD. This query reports the later verified state and exact paths.
SELECT ci.cleanup_batch_id, sr.scoped_path || '/' || sp.relative_path AS full_path,
       a.artifact_kind, ci.disposition, ci.deletion_state,
       ci.pre_delete_sha256, ci.pre_delete_size_bytes, ci.actual_bytes,
       ci.deleted_utc, ci.notes
FROM cleanup_item AS ci
JOIN artifact AS a USING (artifact_id)
JOIN artifact_location AS al USING (artifact_id)
JOIN storage_path AS sp USING (path_id)
JOIN storage_root AS sr USING (root_id)
WHERE ci.cleanup_batch_id = 'RF-YI-QE-ZERO-MIX-CLEANUP-20261124'
ORDER BY full_path;

-- 16. Execution receipt, exact receipt source hash, queue-evidence boundary,
-- and byte total for the zero-byte placeholder cleanup; this is not a
-- personal quota reading. Query 11 also lists the separate post-delete check.
SELECT cb.cleanup_batch_id, cb.status, cb.expected_bytes, cb.actual_bytes,
       r.execution_state, r.deleted_file_count, r.actual_logical_bytes,
       r.actual_allocated_bytes, s.source_path, s.sha256 AS receipt_sha256,
       s.as_of_utc, r.notes
FROM cleanup_batch AS cb
JOIN cleanup_execution_receipt AS r USING (cleanup_batch_id)
JOIN registry_source AS s ON s.source_id = r.receipt_source_id
WHERE cb.cleanup_batch_id = 'RF-YI-QE-ZERO-MIX-CLEANUP-20261124';
