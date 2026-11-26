-- QCK01: the full 3,027-row owner-scoped metadata inventory; blank hashes remain blank.
SELECT model_dir, seed, job_id, file_class, epoch, artifact_path, size_bytes,
       allocated_bytes, mtime_utc, inode, hardlink_count, disposition,
       sha256, sha256_status
FROM artifact_inventory
ORDER BY seed, file_class, epoch, artifact_path;

-- QCK02: rows/bytes by run, artifact class, disposition, and hash state.
SELECT seed, file_class, disposition, sha256_status,
       COUNT(*) AS file_count,
       SUM(size_bytes) AS logical_bytes,
       SUM(allocated_bytes) AS path_allocated_bytes
FROM artifact_inventory
GROUP BY seed, file_class, disposition, sha256_status
ORDER BY seed, file_class, disposition, sha256_status;

-- QCK03: intermediate/final checkpoints by seed. This is metadata, not content identity.
SELECT seed,
       COUNT(*) AS checkpoint_files,
       SUM(CASE WHEN disposition='HOLD_PENDING_USER_POLICY' THEN 1 ELSE 0 END) AS held_files,
       SUM(CASE WHEN disposition='KEEP_DESIGNATED_FINAL' THEN 1 ELSE 0 END) AS designated_final_files,
       MIN(epoch) AS min_epoch,
       MAX(epoch) AS max_epoch,
       SUM(size_bytes) AS logical_bytes,
       SUM(CASE WHEN sha256='' THEN 1 ELSE 0 END) AS no_hash_files
FROM artifact_inventory
WHERE file_class='checkpoint'
GROUP BY seed
ORDER BY seed;

-- QCK04: exact identities of the 12 artifacts hashed against run provenance, with cross-project matches.
-- A hash match identifies a file/model checkpoint only; it does not establish shared task results.
SELECT a.seed, a.file_class, a.epoch, a.artifact_path, a.sha256,
       a.sha256_status,
       dx.model_id AS dualx_model_id,
       r4.model_id AS migrationbench_r4_model_id
FROM artifact_inventory AS a
LEFT JOIN dx__models AS dx
  ON a.sha256<>'' AND lower(a.sha256)=lower(dx.checkpoint_sha256)
LEFT JOIN r4__models AS r4
  ON a.sha256<>'' AND lower(a.sha256)=lower(r4.checkpoint_sha256)
WHERE a.sha256<>''
ORDER BY a.seed, a.file_class, a.epoch, a.artifact_path;

-- QCK05: scheduler outcomes at the captured time; not a live queue query.
SELECT training_run_key, seed, job_id, scheduler_state, exit_code,
       ended_utc, scheduler_snapshot_utc, source_status_file_sha256
FROM training_run_snapshot
ORDER BY seed;

-- QCK06: exact source pins and retention/measurement boundaries.
SELECT metadata_key, metadata_value
FROM snapshot_metadata
ORDER BY metadata_key;
