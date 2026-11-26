-- Read-only examples for MigrationBench R1 lineage overlay R4.
-- Connect with: sqlite3 -readonly migrationbench_r1_lineage_overlay_20261125_r4.sqlite

-- QL21: a result-centric view of source and organized archive evidence.
-- Run-manifest, status, trajectory, input, checkpoint, and derived artifact
-- paths are available from one query; excluded failure status is explicit.
SELECT x.result_key, x.result_kind, x.evidence_status, x.scientific_status,
       r.run_id, r.model_id, r.model_checkpoint_sha256, r.run_status,
       a.artifact_role, a.path_scope, a.availability_status,
       a.source_path, a.byte_size, a.sha256, l.evidence_status AS link_status,
       l.note
FROM lineage_results x
JOIN result_run_links rr ON rr.result_id=x.result_id
JOIN lineage_runs r ON r.run_id=rr.run_id
JOIN run_artifact_links l ON l.run_id=r.run_id
JOIN lineage_artifacts a ON a.artifact_id=l.artifact_id
WHERE r.campaign_id='mace_from_scratch_ablation_md_20261005'
ORDER BY x.result_key, a.artifact_role, a.path_scope, a.source_path;

-- QL22: verify symlink metadata and the dereferenced content as separate rows.
SELECT artifact_id, artifact_role, source_path, path_scope, sha256, byte_size,
       mime_type, availability_status, scientific_status, notes
FROM lineage_artifacts
WHERE source_path IN (
  '/scratch16/pclancy3/yi/mace_from_scratch_ablation_MD_20261005/inputs/common_initial.xyz',
  '/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/0-foundation-mace_mp/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz'
)
ORDER BY path_scope, source_path;

-- QL23: source/archive path pairs and identity equality by SHA256.
SELECT source.sha256, source.byte_size AS source_bytes,
       source.source_path AS observed_source_path,
       archived.byte_size AS archive_bytes,
       archived.source_path AS organized_archive_path,
       CASE WHEN source.sha256=archived.sha256
                  AND source.byte_size=archived.byte_size THEN 'MATCH'
            ELSE 'MISMATCH' END AS identity_check
FROM lineage_artifacts source
JOIN lineage_artifacts archived
  ON archived.sha256=source.sha256
 AND archived.byte_size=source.byte_size
WHERE source.path_scope='SCRATCH16'
  AND archived.path_scope='SCRATCH16'
  AND archived.source_path LIKE '/scratch16/pclancy3/yi/MigrationBench_data_archive/campaigns/cr_sb2te3/mace_from_scratch_ablation_md_20261005/%'
  AND source.source_path NOT LIKE '/scratch16/pclancy3/yi/MigrationBench_data_archive/%'
ORDER BY source.sha256, source.source_path;

-- QL24: result/run terminal states and publication boundary.
SELECT r.run_id, r.model_id, r.scheduler_status, r.run_status,
       r.scientific_status AS run_scientific_status,
       x.result_key, x.value_json, x.evidence_status,
       x.scientific_status AS result_scientific_status
FROM lineage_runs r
JOIN result_run_links rr ON rr.run_id=r.run_id
JOIN lineage_results x ON x.result_id=rr.result_id
WHERE r.campaign_id='mace_from_scratch_ablation_md_20261005'
ORDER BY r.run_id;

-- QL25: Table S5 cell -> result status -> current raw neb.out identity.
SELECT x.source_row_key AS table_cell,
       x.scientific_status AS result_status,
       json_extract(x.value_json,'$.source_scientific_status') AS source_gate_status,
       json_extract(x.value_json,'$.submitted_or_source_value') AS submitted_value_ev,
       a.source_path AS raw_neb_output,
       a.byte_size, a.sha256,
       l.evidence_status AS artifact_link_status, l.note
FROM lineage_results x
JOIN result_artifact_links l ON l.result_id=x.result_id
JOIN lineage_artifacts a ON a.artifact_id=l.artifact_id
WHERE x.result_key LIKE 'R1X:Table S5:table:%'
  AND a.source_path LIKE '%neb.out'
  AND (a.source_path LIKE '/scratch16/%'
       OR a.source_path LIKE 'data/raw_compact/%')
ORDER BY x.source_row_key, a.source_path;

-- QL26: registered result-status counts by Table S5 path identifier.
SELECT substr(x.source_row_key,1,instr(x.source_row_key,' / ')-1) AS path_id,
       x.scientific_status AS result_status,
       json_extract(x.value_json,'$.source_scientific_status') AS source_gate_status,
       count(*) AS table_cells
FROM lineage_results x
WHERE x.result_key LIKE 'R1X:Table S5:table:%'
GROUP BY substr(x.source_row_key,1,instr(x.source_row_key,' / ')-1),
         x.scientific_status,
         json_extract(x.value_json,'$.source_scientific_status')
ORDER BY path_id;
