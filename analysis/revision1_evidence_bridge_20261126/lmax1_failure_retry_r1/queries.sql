-- QLM01: per-stage scientific and materialization status; these are collector rows, not publication results.
SELECT stage, materialization_status, run_status, scientific_status, leakage_gate,
       COUNT(*) AS record_count
FROM campaign_record
GROUP BY stage, materialization_status, run_status, scientific_status, leakage_gate
ORDER BY stage, materialization_status, scientific_status, leakage_gate;

-- QLM02: explicit Tier-1 source failures and their retry/perturbation descendant counts.
WITH descendants AS (
  SELECT parent_record_id,
         SUM(CASE WHEN relation_type='RETRY_OF_TIER1_FAILURE' THEN 1 ELSE 0 END) AS planned_retries,
         SUM(CASE WHEN relation_type='PERTURBATION_OF_TIER1_FAILURE' THEN 1 ELSE 0 END) AS perturb_rows
  FROM record_link
  GROUP BY parent_record_id
)
SELECT r.record_id AS source_record_id, r.operator, r.path_id, r.model_id,
       r.run_status AS source_run_status, r.scientific_status AS source_scientific_status,
       COALESCE(d.planned_retries,0) AS planned_retries,
       COALESCE(d.perturb_rows,0) AS perturb_rows
FROM campaign_record AS r
LEFT JOIN descendants AS d ON d.parent_record_id=r.record_id
WHERE r.stage='TIER1_BASELINE' AND r.scientific_status='NOT_CONVERGED'
ORDER BY r.record_id;

-- QLM03: event-level exact source_record_id relationships; no inferred parent joins.
SELECT l.relation_type, l.parent_record_id AS source_record_id,
       l.child_campaign_id AS followup_campaign_id,
       l.child_record_id AS followup_record_id,
       c.materialization_status, c.run_status, c.scientific_status, c.failure_class,
       c.manifest_path, c.manifest_sha256
FROM record_link AS l
JOIN campaign_record AS c
  ON c.campaign_id=l.child_campaign_id AND c.record_id=l.child_record_id
ORDER BY l.relation_type, l.parent_record_id, l.child_record_id;

-- QLM04: planned but missing Tier-1 retry variants; MISSING is not an optimizer failure.
SELECT record_id, source_record_id, operator, path_id, model_id,
       materialization_status, run_status, scientific_status, failure_class,
       raw_row_json
FROM campaign_record
WHERE stage='TIER1_RETRY' AND materialization_status='MISSING'
ORDER BY record_id;

-- QLM05: Tier-2 candidate diagnostics and their pending leakage gate.
SELECT scientific_status, failure_class, leakage_gate, publication_status,
       COUNT(*) AS record_count
FROM campaign_record
WHERE stage='TIER2_DIAGNOSTIC'
GROUP BY scientific_status, failure_class, leakage_gate, publication_status
ORDER BY scientific_status, failure_class;

-- QLM06: compact source files pinned into this snapshot, including source paths and hashes.
SELECT f.campaign_id, s.stage, f.source_role, f.rockfish_path,
       f.size_bytes, f.sha256
FROM source_file AS f
JOIN campaign_stage AS s USING (campaign_id)
ORDER BY f.campaign_id, f.source_role, f.rockfish_path;

-- QLM07: campaign hierarchy and acceptance-scope pins.
SELECT campaign_id, stage, root_path, parent_campaign_id,
       model_registry_sha256, path_registry_sha256, source_ledger_sha256,
       acceptance_scope_sha256, relationship_boundary
FROM campaign_stage
ORDER BY CASE stage WHEN 'TIER1_BASELINE' THEN 1 WHEN 'TIER1_RETRY' THEN 2
                    WHEN 'TIER1_PERTURBATION' THEN 3 ELSE 4 END;
