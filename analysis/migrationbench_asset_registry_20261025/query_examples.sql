-- MigrationBench Asset Registry: common read-only queries.
-- Run these in the local UI SQL tab or with sqlite3 migrationbench_asset_registry.sqlite.

-- 0) Standardized model names, plot labels, subgroup, and alias coverage.
SELECT m.model_id, mm.preferred_name, mm.plot_label, mm.major_class,
       mm.subgroup_label, GROUP_CONCAT(DISTINCT ma.alias) AS aliases
FROM models AS m
JOIN model_name_map AS mm ON mm.model_id = m.model_id
LEFT JOIN model_aliases AS ma ON ma.model_id = m.model_id
GROUP BY m.model_id, mm.preferred_name, mm.plot_label, mm.major_class, mm.subgroup_label
ORDER BY mm.major_class, mm.subgroup_label, mm.plot_label;

-- 0b) Group-level aliases (use these for family shorthand such as Scratch-600K).
SELECT mga.alias, mga.model_group_id, mg.display_name, mga.alias_context
FROM model_aliases AS mga
JOIN model_groups AS mg ON mg.model_group_id = mga.model_group_id
ORDER BY mg.sort_order, mga.preferred DESC, mga.alias;

-- 1) Current Foundation taxonomy, including explicit fine-tuned Foundation status.
SELECT m.model_id, m.display_name, mf.subfamily, m.training_regime,
       m.lineage_status, m.scientific_status, m.notes
FROM models AS m
JOIN model_families AS mf ON mf.family_id = m.family_id
WHERE mf.major_class = 'Foundation'
ORDER BY mf.subfamily, m.model_id;

-- 2) Which models are complete/partial/missing for a task group?
SELECT preferred_name, plot_label, major_class, subgroup_label,
       task_group_id, task_group_name, coverage_state,
       expected_task_count, covered_task_count, coverage_rows, completed_rows, failure_rows,
       task_ids, run_statuses
FROM v_model_task_summary
WHERE task_group_id = 'mlip_md'
ORDER BY major_class, subgroup_label, plot_label;

-- 3) Models missing at least one task in the MLIP-NEB group.
SELECT preferred_name, plot_label, coverage_state, task_ids, run_statuses
FROM v_model_task_summary
WHERE task_group_id = 'mlip_neb' AND coverage_state <> 'COMPLETE'
ORDER BY major_class, plot_label;

-- 4) Every chart/panel related to a task group and standardized model label.
SELECT DISTINCT chart_id, title, chart_status, task_group_id, task_group_name,
       model_id, preferred_name, plot_label, model_group_id, model_group_name,
       panel_id, panel_label, system_id, dataset_id
FROM v_chart_catalog
WHERE task_group_id = 'mlip_interlayer'
ORDER BY chart_id, panel_id, plot_label;

-- 5) All figures related to one canonical model or plot label.
SELECT DISTINCT chart_id, title, chart_status, campaign_id, task_group_name,
       preferred_name, plot_label, model_group_name
FROM v_chart_catalog
WHERE model_id = 'foundation_mh0'
   OR plot_label = 'MACE-MH-0'
   OR preferred_name = 'MACE-MH-0'
ORDER BY chart_id;

-- 6) Search by a human alias and resolve it without changing canonical IDs.
SELECT alias, model_id, model_group_id, alias_context, preferred
FROM model_aliases
WHERE lower(alias) LIKE lower('%MACE-MH%')
   OR lower(alias) LIKE lower('%600K%')
ORDER BY alias;

-- 2) Which model/task/system combinations are not scientifically accepted?
SELECT c.coverage_id, c.model_id, c.task_id, c.system_id, c.campaign_id,
       c.run_status, c.scientific_status, c.dft_reference_status, c.result_summary
FROM coverage AS c
WHERE c.scientific_status NOT IN ('VERIFIED', 'LINEAGE_SUPPORTED', 'COMPLETE')
ORDER BY c.scientific_status, c.task_id, c.model_id;

-- 3) Fixed versus self-relaxed NEB, intentionally never pooled.
SELECT c.task_id, c.system_id, c.campaign_id, c.model_id,
       c.expected_count, c.completed_count, c.scientific_status, c.result_summary
FROM coverage AS c
WHERE c.task_id IN ('fixed_path_neb', 'self_relaxed_neb', 'mic_neb', 'interlayer_fixed', 'interlayer_self')
ORDER BY c.system_id, c.campaign_id, c.task_id, c.model_id;

-- 4) OOD explosion coverage and the unresolved locator gap.
SELECT c.coverage_id, c.scope_label, c.run_status, c.scientific_status,
       c.result_summary, g.gap_id, g.recommended_next_action
FROM coverage AS c
LEFT JOIN gaps AS g ON g.entity_id = c.dataset_id
WHERE c.task_id = 'ood_explosion';

-- 5) Open repair queue, highest priority first.
SELECT priority, gap_id, entity_type, entity_id, gap_type,
       description, recommended_next_action, status
FROM gaps
WHERE status <> 'CLOSED'
ORDER BY CASE priority WHEN 'P0' THEN 0 WHEN 'P1' THEN 1 ELSE 2 END, gap_id;

-- 6) Compact coverage status heatmap source.
SELECT task_id, scientific_status, COUNT(*) AS records
FROM coverage
GROUP BY task_id, scientific_status
ORDER BY task_id, scientific_status;
