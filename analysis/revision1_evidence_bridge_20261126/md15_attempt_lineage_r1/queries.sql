-- QMA01: preserve submitted R02 status at model grain and show closure boundary.
SELECT model_id,family,display_label,r02_status,r02_reason,last_step,last_time_ps,target_step,
       raw_frames,eligible_frames,invalid_frames,model_sha256,input_sha256,run_dir_exists,
       'ONE_ROW_PER_MODEL_NOT_PER_ATTEMPT' AS grain,
       'CANDIDATE_JOBS_ARE_NOT_CONFIRMED_ATTEMPT_LINKS' AS linkage_boundary
FROM model_attempt_audit ORDER BY family,model_id;

-- QMA02: artifact presence and direct hash scope; NULL hash means no direct digest was recorded.
SELECT model_id,evidence_role,evidence_path,observed_state,reported_bytes,observed_sha256,hash_match,source_note
FROM evidence_artifact ORDER BY model_id,evidence_role,evidence_path;

-- QMA03: candidate Slurm IDs recovered from scoped logs, not a confirmed attempt table.
SELECT model_id,job_id,role_note,linkage_status,source_evidence
FROM scheduler_job_candidate ORDER BY model_id,job_id;

-- QMA04: segment order and reported stat sizes; producer-manifest trajectory SHA is missing.
SELECT model_id,parsed_start_step,evidence_path,reported_bytes,sha256_status
FROM trajectory_segment ORDER BY model_id,parsed_start_step,evidence_path;

-- QMA05: explicit missing links, retaining source wording.
SELECT model_id,gap_text FROM missing_link ORDER BY model_id,gap_text;

-- QMA06: run directory and smoke coverage by family/status; these remain model-row counts.
SELECT family,r02_status,count(*) AS model_rows,
       sum(CASE WHEN run_dir_exists='YES' THEN 1 ELSE 0 END) AS registered_run_dirs,
       sum(CASE WHEN smoke_dir IS NOT NULL AND smoke_dir NOT IN ('','NOT_APPLICABLE','NOT_CHECKED') THEN 1 ELSE 0 END) AS smoke_dirs,
       'MODEL_ROWS_NOT_ATTEMPT_COUNTS' AS grain_boundary
FROM model_attempt_audit GROUP BY family,r02_status ORDER BY family,r02_status;

-- QMA07: source and scope pins.
SELECT metadata_key,metadata_value FROM snapshot_metadata ORDER BY metadata_key;
