-- QMB01: one row per submitted R1 MD model, not per attempt or DUAL-X task.
SELECT
    mb_model_id,
    family,
    training_seed_or_variant,
    md_status_r02,
    reason,
    last_step,
    last_time_ps,
    target_step,
    raw_frames,
    eligible_frames,
    invalid_frames,
    model_sha256,
    input_sha256,
    figure_row_present,
    reuse_class,
    'MODEL_GRAIN_ONLY' AS snapshot_grain,
    'NOT_A_COMPLETE_TASK_OR_ATTEMPT_MATRIX' AS completion_boundary
FROM mb15_model_coverage
ORDER BY family, mb_model_id;

-- QMB02: exact checkpoint identity plus typed evidence counts; never a validation score.
SELECT
    mb_model_id,
    family,
    model_sha256,
    dualx_model_id_exact_sha,
    dualx_registry_exact_model_ids,
    dualx_r4_model_exact_ids,
    dualx_external_link_pass_rows,
    mb_exact_static_candidate_rows,
    mb_exact_neb_rows,
    dualx_primary_inference_runs,
    dualx_primary_gradcam_runs,
    reuse_class,
    'IDENTITY_MATCH_DOES_NOT_ESTABLISH_RESULT_OR_CLAIM_EQUIVALENCE' AS interpretation_boundary
FROM mb15_model_coverage
WHERE dualx_model_id_exact_sha IS NOT NULL
ORDER BY mb_model_id;

-- QMB03: status counts at model grain, partitioned by family.
SELECT
    family,
    md_status_r02,
    count(*) AS model_count,
    sum(eligible_frames) AS eligible_frames_reported,
    sum(invalid_frames) AS invalid_frames_reported,
    'MODEL_SUMMARY_ROWS_NOT_INDEPENDENT_ATTEMPTS' AS grain_warning
FROM mb15_model_coverage
GROUP BY family, md_status_r02
ORDER BY family, md_status_r02;

-- QMB04: rows with no result-level bridge recorded in this pinned snapshot.
SELECT
    mb_model_id,
    model_sha256,
    reuse_class,
    'NO_RESULT_LEVEL_BRIDGE_IN_THIS_SNAPSHOT' AS snapshot_status,
    'NOT_ASSERTED_ABSENT_FROM_ALL_ARCHIVES' AS search_boundary
FROM mb15_model_coverage
WHERE coalesce(dualx_external_link_pass_rows, 0) = 0
  AND coalesce(mb_exact_static_candidate_rows, 0) = 0
  AND coalesce(mb_exact_neb_rows, 0) = 0
ORDER BY mb_model_id;

-- QMB05: snapshot provenance pins and the declared grain/limitations.
SELECT metadata_key, metadata_value
FROM snapshot_metadata
ORDER BY metadata_key;

-- QMB06: live source-qualified identity/result-link counts across MB15, MB R4, and DUAL-X v14.
-- Identity is exact SHA only; each evidence family remains a separate count.
SELECT
    c.mb_model_id,
    c.model_sha256,
    d.model_id AS dualx_model_id_exact_sha,
    m.model_id AS mb_r4_model_id_exact_sha,
    (SELECT count(*) FROM dx__mb_external_model_links AS l
      WHERE l.dualx_model_id = d.model_id
        AND lower(l.checkpoint_sha256) = lower(c.model_sha256)
        AND l.mapping_status = 'PASS') AS dualx_external_model_link_rows,
    (SELECT count(*) FROM dx__mb_external_neb_runs AS n
      WHERE n.dualx_model_id = d.model_id
        AND lower(n.checkpoint_sha256) = lower(c.model_sha256)) AS path_specific_neb_rows,
    (SELECT count(*) FROM dx__mb_external_static_predictions AS s
      WHERE s.dualx_model_id = d.model_id
        AND lower(s.checkpoint_sha256) = lower(c.model_sha256)) AS prepared_static_candidate_rows,
    CASE WHEN m.model_id IS NULL
           AND NOT EXISTS (
               SELECT 1 FROM dx__mb_external_model_links AS l
               WHERE l.dualx_model_id = d.model_id
                 AND lower(l.checkpoint_sha256) = lower(c.model_sha256)
                 AND l.mapping_status = 'PASS'
           )
           AND NOT EXISTS (
               SELECT 1 FROM dx__mb_external_neb_runs AS n
               WHERE n.dualx_model_id = d.model_id
                 AND lower(n.checkpoint_sha256) = lower(c.model_sha256)
           )
           AND NOT EXISTS (
               SELECT 1 FROM dx__mb_external_static_predictions AS s
               WHERE s.dualx_model_id = d.model_id
                 AND lower(s.checkpoint_sha256) = lower(c.model_sha256)
           )
         THEN 'EXACT_CHECKPOINT_IDENTITY_ONLY_IN_PINNED_MB_BRIDGE'
         ELSE 'EXACT_CHECKPOINT_WITH_TYPED_BRIDGE_ROWS'
    END AS bridge_scope,
    'IDENTITY_AND_LINK_COUNTS_DO_NOT_PROVE_TASK_EQUIVALENCE_OR_CLAIM_SUPPORT' AS interpretation_boundary
FROM mb15_model_coverage AS c
LEFT JOIN dx__models AS d
  ON length(d.checkpoint_sha256) = 64
 AND lower(d.checkpoint_sha256) = lower(c.model_sha256)
LEFT JOIN r4__models AS m
  ON length(m.checkpoint_sha256) = 64
 AND lower(m.checkpoint_sha256) = lower(c.model_sha256)
ORDER BY c.mb_model_id;

-- QMB07: expand exact-SHA-matched MB15 models across the 18 DUAL-X design tasks.
-- These are DUAL-X design/run cells, not MigrationBench task results.
SELECT
    c.mb_model_id,
    c.md_status_r02 AS mb_r1_md_status,
    c.reuse_class AS mb_to_dualx_reuse_class,
    c.model_sha256,
    d.model_id AS dualx_model_id_exact_sha,
    cell.task_id AS dualx_task_id,
    cell.dataset_id AS dualx_dataset_id,
    cell.planned_status AS dualx_planned_status,
    count(DISTINCT r.run_id) AS dualx_run_rows,
    group_concat(DISTINCT r.run_id) AS dualx_run_ids,
    group_concat(DISTINCT r.run_scope || ':' || r.run_type || ':' || r.run_status) AS distinct_dualx_run_states,
    'DUALX_DESIGN_CELL_NOT_MIGRATIONBENCH_TASK_RESULT' AS project_boundary
FROM mb15_model_coverage AS c
LEFT JOIN dx__models AS d
  ON length(d.checkpoint_sha256) = 64
 AND lower(d.checkpoint_sha256) = lower(c.model_sha256)
LEFT JOIN dx__matrix_cells AS cell
  ON cell.model_id = d.model_id
LEFT JOIN dx__runs AS r
  ON r.model_id = cell.model_id
 AND r.task_id = cell.task_id
GROUP BY c.mb_model_id, c.md_status_r02, c.reuse_class, c.model_sha256,
         d.model_id, cell.task_id, cell.dataset_id, cell.planned_status
ORDER BY c.mb_model_id, cell.task_id;
