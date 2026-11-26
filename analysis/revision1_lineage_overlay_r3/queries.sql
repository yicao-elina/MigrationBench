-- Read-only examples for MigrationBench R1 lineage overlay R3.
-- Connect using sqlite3 -readonly migrationbench_r1_lineage_overlay_20261125_r3.sqlite

-- QL13: terminal failure runs, preserving the publication boundary.
SELECT r.run_id, r.model_id, r.model_checkpoint_sha256, r.seed AS training_seed,
       r.sample_key AS md_seed, r.scheduler_job_id, r.run_status,
       r.scientific_status, c.status AS campaign_status, r.notes
FROM lineage_runs r JOIN campaigns c USING(campaign_id)
WHERE c.campaign_id='mace_from_scratch_ablation_md_20261005'
ORDER BY r.run_id;

-- QL14: exact terminal-step facts and linked run identity.
SELECT x.result_key, x.value_json, x.evidence_status, x.scientific_status,
       rr.run_id, rr.relation, rr.evidence_status AS run_link_status
FROM lineage_results x JOIN result_run_links rr USING(result_id)
WHERE x.result_key LIKE 'mace_terminal_failure_r3:%'
ORDER BY x.result_key;

-- QL15: each run's inputs, checkpoints, raw outputs, and derived evidence.
SELECT r.run_id, a.artifact_role, a.path_scope, a.availability_status,
       a.source_path, a.byte_size, a.sha256, l.evidence_status, l.note
FROM lineage_runs r JOIN run_artifact_links l USING(run_id)
JOIN lineage_artifacts a USING(artifact_id)
WHERE r.campaign_id='mace_from_scratch_ablation_md_20261005'
ORDER BY r.run_id, a.artifact_role, a.source_path;

-- QL16: Rockfish/Git postprocess SHA differences; never collapse by basename.
SELECT r.run_id, a.source_path, a.byte_size, a.sha256, l.evidence_status, l.note
FROM lineage_runs r JOIN run_artifact_links l USING(run_id)
JOIN lineage_artifacts a USING(artifact_id)
WHERE r.campaign_id='mace_from_scratch_ablation_md_20261005'
  AND a.artifact_role='DERIVED_DATA'
ORDER BY r.run_id, a.source_path;

-- QL17: counts and integrity boundary for the imported campaign.
SELECT (SELECT count(*) FROM lineage_runs WHERE campaign_id='mace_from_scratch_ablation_md_20261005') AS run_count,
       (SELECT count(*) FROM lineage_results WHERE result_key LIKE 'mace_terminal_failure_r3:%') AS result_count,
       (SELECT count(*) FROM lineage_artifacts WHERE artifact_key LIKE 'mace_terminal_failure_r3:%') AS artifact_count,
       (SELECT count(*) FROM run_artifact_links WHERE run_id IN
          (SELECT run_id FROM lineage_runs WHERE campaign_id='mace_from_scratch_ablation_md_20261005')) AS run_artifact_links,
       (SELECT count(*) FROM result_run_links WHERE result_id IN
          (SELECT result_id FROM lineage_results WHERE result_key LIKE 'mace_terminal_failure_r3:%')) AS result_run_links;
