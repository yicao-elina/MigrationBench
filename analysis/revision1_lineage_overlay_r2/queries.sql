-- QL07: inventory coverage by kind and observation status
SELECT artifact_role, availability_status, COUNT(*) AS n
FROM lineage_artifacts
WHERE artifact_key LIKE 'r1-manifest:%'
GROUP BY artifact_role, availability_status
ORDER BY artifact_role, availability_status;

-- QL08: located legacy QE outputs connected to Table S5, preserving PARTIAL status
SELECT r.result_key, a.source_path, a.byte_size, a.sha256,
       l.evidence_status, l.note
FROM result_artifact_links l
JOIN lineage_results r USING(result_id)
JOIN lineage_artifacts a USING(artifact_id)
WHERE a.artifact_key LIKE 'r1-legacy-qe:%'
ORDER BY r.result_key;

-- QL09: MD continuation attempts and their linked observations; DB run status is intentionally UNKNOWN
SELECT c.title, r.scheduler_job_id, r.scheduler_status, r.run_status,
       r.scientific_status, r.notes
FROM lineage_runs r
JOIN campaigns c USING(campaign_id)
WHERE c.campaign_id IN ('md_continuation_400ps_rockfish','md_continuation_1ns_rockfish')
ORDER BY c.campaign_id, r.scheduler_job_id;

-- QL10: per-attempt MD logs/output paths
SELECT r.scheduler_job_id, a.artifact_role, a.source_path, a.sha256,
       l.evidence_status, l.note
FROM run_artifact_links l
JOIN lineage_runs r USING(run_id)
JOIN lineage_artifacts a USING(artifact_id)
WHERE r.campaign_id IN ('md_continuation_400ps_rockfish','md_continuation_1ns_rockfish')
ORDER BY r.scheduler_job_id, a.artifact_role, a.source_path;

-- QL11: Lmax1 G3 campaign-level diagnostic boundaries (no invented run rows)
SELECT campaign_id, title, campaign_path, status, notes
FROM campaigns
WHERE campaign_id LIKE 'scratch_lmax1_g3%'
ORDER BY campaign_id;

-- QL12: unresolved R1 results remain partial/conflict/missing and have no deterministic run producer
SELECT r.result_key, r.evidence_status, r.scientific_status,
       COUNT(DISTINCT rr.run_id) AS linked_runs,
       SUM(CASE WHEN rr.relation='PRODUCED_BY' THEN 1 ELSE 0 END) AS produced_by_links
FROM lineage_results r
LEFT JOIN result_run_links rr USING(result_id)
GROUP BY r.result_id
HAVING r.evidence_status IN ('PARTIAL','CONFLICT','MISSING')
ORDER BY r.result_key;
