-- Candidate -> MD frame/run -> source checkpoint -> evaluated checkpoints.
SELECT candidate_id, execution_alias, source_label, training_seed, md_velocity_seed,
       run_key, step, time_ps, source_window, md_status,
       source_md_checkpoint_sha256, evaluated_model_id, evaluated_checkpoint_sha256,
       evaluation_status, energy_eV, max_force_eV_A, input_sha256
FROM v_candidate_evaluation_trace
WHERE candidate_id = 'dftcand_0049'
ORDER BY evaluated_model_id;

-- Claims and their resolved evidence files and figure-panel links.
SELECT * FROM v_claim_evidence_trace
WHERE claim_id IN ('C001','C003','C004','C005','C009','C013')
ORDER BY claim_id, evidence_ref_verbatim;

SELECT * FROM v_claim_figure_panel_trace
WHERE claim_id IN ('C001','C003','C004','C013')
ORDER BY claim_id, panel_id;

-- Verify the static evaluation matrix is exactly one row per candidate/checkpoint.
SELECT COUNT(*) AS n_rows,
       COUNT(DISTINCT candidate_id) AS n_candidates,
       COUNT(DISTINCT checkpoint_sha256) AS n_checkpoints,
       SUM(evaluation_status='success') AS n_success
FROM model_evaluation;

-- Observe DFT preparation state without treating it as a completed reference result.
SELECT b.batch_id, b.batch_status, b.no_dft_submitted,
       COUNT(m.candidate_id) AS n_members, COUNT(DISTINCT m.dft_status) AS n_dft_statuses
FROM selection_batch b LEFT JOIN selection_member m USING(batch_id)
GROUP BY b.batch_id, b.batch_status, b.no_dft_submitted
ORDER BY b.batch_id;

-- Source files whose manifest-relative evidence references did not resolve.
SELECT claim_id, evidence_ref_verbatim, resolution_status
FROM claim_evidence
WHERE resolution_status IN ('AMBIGUOUS_BASENAME','UNRESOLVED_OR_EXTERNAL')
ORDER BY claim_id, evidence_ref_verbatim;

-- Historical submission/resource-gate statements are evidence records, not live job status.
SELECT scope, status_as_reported, time_scope, interpretation_boundary
FROM campaign_status_evidence
ORDER BY status_evidence_id;
