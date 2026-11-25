-- Current metadata-only storage observations for the two authorized roots.
-- The traversal counts cover all descendants; row-level observations are
-- populated for hardlink members and the largest paths only (see snapshot notes).

-- Q1: latest scan totals and coverage notes for the two exact paths.
SELECT s.snapshot_id, r.scoped_path, s.started_utc, s.completion_state,
       s.files_seen, s.directories_seen, s.symlinks_seen, s.unreadable_count,
       s.logical_bytes, s.allocated_bytes, s.manifest_path, s.manifest_sha256,
       s.notes
FROM inventory_snapshot AS s
JOIN storage_root AS r USING(root_id)
WHERE s.snapshot_id IN ('RF-DATA-YI-20261125','RF-SCRATCH4-YI-20261125')
ORDER BY r.scoped_path;

-- Q2: largest `/data/pclancy3/yi` files with existing artifact/provenance links.
SELECT sp.relative_path, io.logical_bytes, io.allocated_bytes,
       a.artifact_key, a.artifact_kind, a.semantic_role, a.scientific_status,
       a.retention_class, a.sha256 AS catalog_sha256,
       COUNT(DISTINCT ra.run_id) AS run_links,
       COUNT(DISTINCT ce.claim_id) AS claim_links
FROM inventory_snapshot AS s
JOIN inventory_observation AS io USING(snapshot_id)
JOIN storage_path AS sp USING(path_id)
JOIN storage_root AS r ON r.root_id=sp.root_id
LEFT JOIN artifact_location AS al ON al.path_id=sp.path_id
LEFT JOIN artifact AS a ON a.artifact_id=al.artifact_id
LEFT JOIN run_artifact AS ra ON ra.artifact_id=a.artifact_id
LEFT JOIN claim_evidence AS ce ON ce.artifact_id=a.artifact_id
WHERE s.snapshot_id='RF-DATA-YI-20261125'
GROUP BY sp.path_id
ORDER BY io.logical_bytes DESC
LIMIT 200;

-- Q3: current in-tree hardlink groups in `/data/pclancy3/yi`. Inode/device are
-- observation metadata; this does not establish duplicate file content.
SELECT io.device, io.inode, COUNT(*) AS paths,
       MIN(io.logical_bytes) AS min_logical_bytes,
       MAX(io.logical_bytes) AS max_logical_bytes,
       SUM(io.allocated_bytes) AS allocated_path_sum
FROM inventory_snapshot AS s
JOIN inventory_observation AS io USING(snapshot_id)
JOIN storage_path AS sp USING(path_id)
WHERE s.snapshot_id='RF-DATA-YI-20261125'
  AND io.device IS NOT NULL AND io.inode IS NOT NULL
GROUP BY io.device, io.inode
HAVING COUNT(*) > 1
ORDER BY allocated_path_sum DESC;
