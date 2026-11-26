-- Read-only exact-identity bridge draft, captured 2026-11-26.
-- Open the v14 DUAL-X SQLite with URI/read-only enabled, then attach the MB R4
-- snapshot read-only as `mb`. Both paths are personal Scratch16 archive paths.
-- Example (Python):
--   con = sqlite3.connect('file:'+DUALX_DB+'?mode=ro', uri=True)
--   con.execute('ATTACH DATABASE ? AS mb', ('file:'+MB_R4_DB+'?mode=ro',))

-- QX01: exact DUAL-X checkpoint SHA against the MB R1 frozen source registry,
-- MB R4 canonical model table, and explicit v14 external-model links.
-- SHA match establishes model identity only; it does not establish task/result
-- equivalence or claim support.
SELECT d.model_id AS dualx_model_id,
       d.checkpoint_sha256,
       GROUP_CONCAT(DISTINCT r.model_id) AS r1_registry_model_ids_exact_sha,
       COUNT(DISTINCT r.model_id) AS r1_registry_model_id_count,
       GROUP_CONCAT(DISTINCT m.model_id) AS mb_current_model_ids_exact_sha,
       GROUP_CONCAT(DISTINCT l.source_id) AS explicit_link_sources,
       COUNT(DISTINCT l.source_id) AS explicit_link_source_count
FROM models d
LEFT JOIN mb.rev1_model_registry r
  ON length(r.sha256_recorded)=64
 AND lower(r.sha256_recorded)=lower(d.checkpoint_sha256)
LEFT JOIN mb.models m
  ON length(m.checkpoint_sha256)=64
 AND lower(m.checkpoint_sha256)=lower(d.checkpoint_sha256)
LEFT JOIN mb_external_model_links l
  ON l.dualx_model_id=d.model_id
 AND lower(l.checkpoint_sha256)=lower(d.checkpoint_sha256)
 AND l.mapping_status='PASS'
GROUP BY d.model_id,d.checkpoint_sha256
ORDER BY d.model_id;

-- QX02: exact-hash static candidate evidence, preserving geometry and row grain.
-- These are prepared static candidates, not DUAL-X held-out R1 frames.
SELECT source_row_key,source_id,migrationbench_model_id,dualx_model_id,
       checkpoint_sha256,task_id,dataset_id,candidate_id,geometry_sha256,
       n_atoms,energy_eV,energy_per_atom_eV,max_force_eV_A,rms_force_eV_A,
       evaluation_status,candidate_reason,source_campaign_id,source_file
FROM mb_external_static_predictions
WHERE length(checkpoint_sha256)=64
  AND dualx_model_id IS NOT NULL
ORDER BY dualx_model_id,task_id,source_row_key;

-- QX03: exact-checkpoint NEB records. Keep each path, attempt, operator, and
-- publication disposition separate; never aggregate these into a model score.
SELECT record_key,source_id,migrationbench_model_id,dualx_model_id,
       checkpoint_sha256,task_id,operator,system_id,path_id,campaign_id,
       attempt_id,run_manifest_path,run_manifest_sha256,
       initial_structure_sha256,final_structure_sha256,
       scientific_status,publication_status,optimizer_converged,
       final_max_neb_residual_eV_A,fmax_target_eV_A,evidence_scope
FROM mb_external_neb_runs
WHERE model_hash_verification='PASS_EXACT_SHA256'
ORDER BY dualx_model_id,task_id,path_id,attempt_id;

-- QX04: all 414 registered DUAL-X design cells, left joined to actual run rows.
-- A cell with no run stays present; repeated inference/GradCAM runs remain rows.
SELECT c.model_id,c.task_id,c.dataset_id,c.planned_status,
       r.run_id,r.run_scope,r.run_type,r.run_status,r.checkpoint_sha256,
       r.input_sha256,r.output_manifest_path,r.output_manifest_sha256,
       r.failure_reason
FROM matrix_cells c
LEFT JOIN runs r
  ON r.model_id=c.model_id AND r.task_id=c.task_id
ORDER BY c.model_id,c.task_id,r.run_scope,r.run_type,r.run_id;

-- QX05: source coverage status is a registry/coverage relation, not a metric.
SELECT source_id,mapping_status,COUNT(*) AS model_rows,
       COUNT(DISTINCT dualx_model_id) AS distinct_dualx_models
FROM mb_external_model_coverage
GROUP BY source_id,mapping_status
ORDER BY source_id,mapping_status;
