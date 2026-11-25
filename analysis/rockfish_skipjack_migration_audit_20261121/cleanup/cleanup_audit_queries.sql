-- Completed Rockfish restart cleanup.
SELECT action_id, status, COUNT(*) AS artifacts,
       SUM(size_bytes) AS bytes_removed
FROM cleanup_artifact
GROUP BY action_id, status
ORDER BY action_id;

-- Preserved, hashed evidence associated with each Rockfish cleanup batch.
SELECT r.action_id, COUNT(*) AS evidence_artifacts,
       SUM(r.size_bytes) AS evidence_bytes,
       COUNT(DISTINCT r.run_id) AS runs
FROM retained_evidence AS r
JOIN cleanup_action AS a USING (action_id)
WHERE a.source_root LIKE '/scratch16/%'
GROUP BY r.action_id
ORDER BY r.action_id;

-- Completed local cleanup, grouped by action and category.
SELECT a.action_id, a.rule, a.artifact_count, a.actual_bytes,
       x.category, COUNT(x.artifact_path) AS category_artifacts,
       SUM(x.size_bytes) AS category_bytes
FROM local_cleanup_action AS a
LEFT JOIN local_cleanup_artifact AS x USING (action_id)
GROUP BY a.action_id, a.rule, a.artifact_count, a.actual_bytes, x.category
ORDER BY a.action_id, x.category;

-- Cloud placeholders that have no same-path item in unified main.
SELECT COUNT(*) AS placeholder_paths,
       SUM(legacy_logical_bytes) AS logical_bytes,
       SUM(legacy_allocated_512b_blocks) * 512 AS allocated_bytes
FROM legacy_dataless_placeholder
WHERE path_exists_in_unified_main = 0;

-- The directly inspected File Provider item and materialization observation.
SELECT * FROM local_fileprovider_observation;
