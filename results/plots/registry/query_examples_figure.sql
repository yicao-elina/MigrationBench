-- Figure-level registry queries for review and release checks.

-- Catalog all charts with their campaign and provenance status.
SELECT chart_id, title, chart_status, campaign_id, asset_path
FROM v_chart_catalog
ORDER BY chart_id;

-- Inspect the user-certified R05 chart and all linked files.
SELECT c.chart_id, c.title, c.chart_status, c.asset_path,
       p.provenance_status, p.manifest_path, p.output_sha256
FROM charts AS c
JOIN chart_provenance AS p USING (chart_id)
WHERE c.chart_id = 'chart_cr_three_path_scratch600k_r05_20261101';

SELECT chart_id, panel_id, panel_label, panel_title, task_id, dataset_id
FROM chart_panels
WHERE chart_id = 'chart_cr_three_path_scratch600k_r05_20261101'
ORDER BY panel_id;

SELECT chart_id, file_id, display_name, local_path, file_format, file_exists
FROM chart_data_files
WHERE chart_id = 'chart_cr_three_path_scratch600k_r05_20261101'
ORDER BY file_id;

-- Keep fixed and endpoint-only self-relaxed datasets separate.
SELECT dataset_id, display_name, task_id, record_count, scientific_status,
       dft_reference_status, notes
FROM datasets
WHERE dataset_id IN ('scratch600k_cr81_fixed_r05', 'scratch600k_cr81_self_r05')
ORDER BY task_id;

-- Integrity check to run after any migration.
PRAGMA foreign_key_check;
