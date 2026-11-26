-- QF501: submitted Figure 5 model × task cells with explicit metric/status provenance.
SELECT row_id,row_label,family,strategy,model_sha256,column_id,column_label,task_block,metric_type,
       raw_value_eV,predicted_value_eV,error_eV,absolute_error_eV,n_available,n_expected,
       coverage_fraction,aggregation,status_summary,scientific_status,publication_status,
       source_artifact,matrix_status,
       'FIGURE5_MEASURED_DATA_NOT_COMPLETE_MODEL_BY_TASK_MATRIX' AS scope_boundary
FROM fig5_matrix_cell ORDER BY family,row_id,column_id;

-- QF502: status and count by task; diagnostic/failure rows remain visible.
SELECT column_id,column_label,publication_status,scientific_status,count(*) AS row_count,
       'FIGURE5_SOURCE_ROWS_NOT_INDEPENDENT_ATTEMPTS' AS grain_boundary
FROM fig5_matrix_cell GROUP BY column_id,column_label,publication_status,scientific_status
ORDER BY column_id,publication_status,scientific_status;

-- QF503: model checkpoint identity and task coverage, retaining source matrix status.
SELECT row_id,row_label,model_sha256,count(*) AS task_rows,
       sum(CASE WHEN publication_status='PRIMARY_EVIDENCE' THEN 1 ELSE 0 END) AS primary_rows,
       sum(CASE WHEN publication_status='DIAGNOSTIC_ONLY' THEN 1 ELSE 0 END) AS diagnostic_only_rows,
       sum(CASE WHEN publication_status='SUPPORTING_EVIDENCE' THEN 1 ELSE 0 END) AS supporting_rows,
       group_concat(column_id, ',') AS task_ids,
       '15_MODELS_X_8_SUBMITTED_FIGURE5_TASKS_ONLY' AS coverage_boundary
FROM fig5_matrix_cell GROUP BY row_id,row_label,model_sha256 ORDER BY row_id;

-- QF504: matrix source and processing pins.
SELECT metadata_key,metadata_value FROM snapshot_metadata ORDER BY metadata_key;
