-- Read-only queries for revision1_manifest_sidecar_r1.sqlite.

-- Q1: inventory of staged entities by row grain.
SELECT 'manifest_observation' AS entity, count(*) AS rows FROM manifest_observation
UNION ALL SELECT 'manifest_control_observation', count(*) FROM manifest_control_observation
UNION ALL SELECT 'manifest_control_field', count(*) FROM manifest_control_field
UNION ALL SELECT 'manifest_path_reference', count(*) FROM manifest_path_reference
UNION ALL SELECT 'path_revalidation', count(*) FROM path_revalidation
UNION ALL SELECT 'artifact', count(*) FROM artifact
UNION ALL SELECT 'artifact_location', count(*) FROM artifact_location;

-- Q2: metadata file revalidation by source root and current status.
SELECT r.scoped_path, m.manifest_kind, m.current_verification_status, count(*) AS file_count
FROM manifest_observation m
JOIN storage_root r USING(root_id)
GROUP BY r.scoped_path, m.manifest_kind, m.current_verification_status
ORDER BY r.scoped_path, m.manifest_kind, m.current_verification_status;

-- Q3: job/retry/attempt/scheduler fields; all remain UNVERIFIED observations.
SELECT f.field_path, f.value_type, f.interpretation_status, count(*) AS observation_count
FROM manifest_control_field f
GROUP BY f.field_path, f.value_type, f.interpretation_status
ORDER BY observation_count DESC, f.field_path;

-- Q4: every candidate path reference, preserving earlier resolution and scope.
SELECT scope_class, previous_resolution, current_verification_status,
       count(*) AS reference_occurrences
FROM manifest_path_reference
GROUP BY scope_class, previous_resolution, current_verification_status
ORDER BY scope_class, previous_resolution, current_verification_status;

-- Q5: personal-root references with revalidated artifact identity where available.
SELECT m.manifest_kind, m.manifest_relpath, f.field_path, f.literal_value,
       f.previous_resolution, f.current_verification_status,
       a.artifact_kind, a.sha256, a.byte_size, a.scientific_status
FROM manifest_path_reference f
JOIN manifest_observation m USING(manifest_observation_id)
LEFT JOIN artifact a ON a.artifact_id=f.target_artifact_id
WHERE f.scope_class IN ('PERSONAL_INPUT_ROOT','PERSONAL_RUN_ROOT')
ORDER BY m.manifest_kind, m.manifest_relpath, f.reference_index;

-- Q6: current target revalidation detail; 13 large target files have size only.
SELECT r.scoped_path, p.relative_path, v.target_kind, v.observed_type,
       v.expected_bytes, v.current_bytes, v.expected_sha256, v.current_sha256,
       v.verification_status
FROM path_revalidation v
JOIN storage_path p USING(path_id)
JOIN storage_root r USING(root_id)
ORDER BY r.scoped_path, p.relative_path;

-- Q7: scientific and scheduler entities intentionally absent in this staging DB.
SELECT name, type FROM sqlite_master
WHERE type IN ('table','view')
  AND name IN ('workflow_run','slurm_job','slurm_state_observation','run_artifact','paper_claim','claim_evidence');
