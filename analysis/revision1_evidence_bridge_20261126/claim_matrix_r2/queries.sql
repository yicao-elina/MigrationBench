-- QCE05: Figure 4 R4 summaries → pinned R05 group rows; summary grain only.
-- A NULL profile_data_line with MISSING_OR_EXCLUDED identifies a missing row,
-- not zero attempts. No attempt/run linkage or scientific acceptance is asserted.
SELECT crosswalk_key, r4_result_id, r4_result_key, panel, group_key,
       observable, path_key, path_class, profile_data_line, coverage_data_line,
       match_status, n_models, n_profile_rows, r05_scientific_statuses,
       r05_publication_statuses, source_boundary, r4_protocol_key,
       r4_evidence_status, r4_scientific_status, mapping_grain,
       run_attempt_linkage, scientific_acceptance, source_r05_key
FROM fig4_r05_summary_crosswalk
ORDER BY CASE panel WHEN 'a' THEN 1 WHEN 'b' THEN 2 WHEN 'c' THEN 3
                    WHEN 'd' THEN 4 WHEN 'e' THEN 5 WHEN 'f' THEN 6 END,
         CASE group_key WHEN 'foundation' THEN 1 WHEN 'ft_600k' THEN 2
                        WHEN 'ft_multit' THEN 3 WHEN 'scratch_600k' THEN 4
                        WHEN 'scratch_multit' THEN 5 END;
