PRAGMA foreign_keys = ON;

CREATE TABLE metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE sources (
  source_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  source_path TEXT NOT NULL,
  source_kind TEXT NOT NULL,
  evidence_level TEXT NOT NULL,
  as_of_date TEXT,
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE model_families (
  family_id TEXT PRIMARY KEY,
  major_class TEXT NOT NULL,
  subfamily TEXT NOT NULL,
  description TEXT NOT NULL
);

CREATE TABLE model_groups (
  model_group_id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  major_class TEXT NOT NULL,
  group_level TEXT NOT NULL,
  sort_order INTEGER NOT NULL DEFAULT 0,
  description TEXT NOT NULL DEFAULT ''
);

CREATE TABLE models (
  model_id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  family_id TEXT NOT NULL REFERENCES model_families(family_id),
  identity_kind TEXT NOT NULL,
  training_regime TEXT NOT NULL,
  training_data_label TEXT,
  training_temperature_label TEXT,
  training_seed TEXT,
  representation_label TEXT,
  checkpoint_sha256 TEXT,
  artifact_path TEXT,
  lineage_status TEXT NOT NULL,
  release_status TEXT NOT NULL,
  scientific_status TEXT NOT NULL,
  active_scope TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE model_aliases (
  alias_id INTEGER PRIMARY KEY,
  model_id TEXT REFERENCES models(model_id),
  model_group_id TEXT REFERENCES model_groups(model_group_id),
  alias TEXT NOT NULL,
  alias_context TEXT NOT NULL,
  preferred INTEGER NOT NULL DEFAULT 0,
  CHECK(model_id IS NOT NULL OR model_group_id IS NOT NULL),
  UNIQUE(model_id, model_group_id, alias, alias_context)
);

CREATE TABLE model_group_members (
  model_group_id TEXT NOT NULL REFERENCES model_groups(model_group_id),
  model_id TEXT NOT NULL REFERENCES models(model_id),
  membership_role TEXT NOT NULL,
  PRIMARY KEY (model_group_id, model_id)
);

CREATE TABLE model_name_map (
  model_id TEXT PRIMARY KEY REFERENCES models(model_id),
  preferred_name TEXT NOT NULL,
  plot_label TEXT NOT NULL,
  short_label TEXT NOT NULL,
  major_class TEXT NOT NULL,
  subgroup_label TEXT NOT NULL,
  search_text TEXT NOT NULL,
  naming_status TEXT NOT NULL DEFAULT 'STANDARDIZED',
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE task_groups (
  task_group_id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  task_layer TEXT NOT NULL,
  sort_order INTEGER NOT NULL DEFAULT 0,
  description TEXT NOT NULL DEFAULT ''
);

CREATE TABLE tasks (
  task_id TEXT PRIMARY KEY,
  category TEXT NOT NULL,
  operator TEXT NOT NULL,
  comparison_layer TEXT NOT NULL,
  description TEXT NOT NULL,
  publication_boundary TEXT NOT NULL
);

CREATE TABLE task_group_members (
  task_group_id TEXT NOT NULL REFERENCES task_groups(task_group_id),
  task_id TEXT NOT NULL REFERENCES tasks(task_id),
  membership_role TEXT NOT NULL DEFAULT 'canonical_task',
  PRIMARY KEY (task_group_id, task_id)
);

CREATE TABLE task_aliases (
  alias_id INTEGER PRIMARY KEY,
  task_id TEXT NOT NULL REFERENCES tasks(task_id),
  alias TEXT NOT NULL,
  alias_context TEXT NOT NULL,
  preferred INTEGER NOT NULL DEFAULT 0,
  UNIQUE(task_id, alias, alias_context)
);

CREATE TABLE task_group_aliases (
  alias_id INTEGER PRIMARY KEY,
  task_group_id TEXT NOT NULL REFERENCES task_groups(task_group_id),
  alias TEXT NOT NULL,
  alias_context TEXT NOT NULL,
  preferred INTEGER NOT NULL DEFAULT 0,
  UNIQUE(task_group_id, alias, alias_context)
);

CREATE TABLE systems (
  system_id TEXT PRIMARY KEY,
  material_family TEXT NOT NULL,
  chemistry TEXT NOT NULL,
  atom_count INTEGER,
  defect_or_geometry TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE campaigns (
  campaign_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  campaign_path TEXT,
  source_id TEXT REFERENCES sources(source_id),
  campaign_date TEXT,
  status TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE datasets (
  dataset_id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  data_layer TEXT NOT NULL,
  task_id TEXT REFERENCES tasks(task_id),
  system_id TEXT REFERENCES systems(system_id),
  campaign_id TEXT REFERENCES campaigns(campaign_id),
  record_count INTEGER,
  unit_of_record TEXT,
  scientific_status TEXT NOT NULL,
  dft_reference_status TEXT,
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE coverage (
  coverage_id TEXT PRIMARY KEY,
  model_id TEXT REFERENCES models(model_id),
  task_id TEXT NOT NULL REFERENCES tasks(task_id),
  system_id TEXT REFERENCES systems(system_id),
  campaign_id TEXT REFERENCES campaigns(campaign_id),
  dataset_id TEXT REFERENCES datasets(dataset_id),
  scope_label TEXT NOT NULL,
  expected_count INTEGER,
  completed_count INTEGER,
  run_status TEXT NOT NULL,
  scientific_status TEXT NOT NULL,
  dft_reference_status TEXT,
  result_summary TEXT NOT NULL DEFAULT '',
  source_id TEXT REFERENCES sources(source_id)
);

CREATE TABLE assets (
  asset_id TEXT PRIMARY KEY,
  asset_type TEXT NOT NULL,
  display_name TEXT NOT NULL,
  local_path TEXT NOT NULL,
  campaign_id TEXT REFERENCES campaigns(campaign_id),
  source_id TEXT REFERENCES sources(source_id),
  existence_at_build INTEGER NOT NULL,
  scientific_status TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE charts (
  chart_id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  chart_kind TEXT NOT NULL,
  chart_status TEXT NOT NULL,
  asset_path TEXT NOT NULL DEFAULT '',
  asset_mime TEXT NOT NULL DEFAULT 'application/octet-stream',
  asset_exists INTEGER NOT NULL DEFAULT 0,
  campaign_id TEXT REFERENCES campaigns(campaign_id),
  source_id TEXT REFERENCES sources(source_id),
  data_query TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE chart_panels (
  chart_id TEXT NOT NULL REFERENCES charts(chart_id),
  panel_id TEXT NOT NULL,
  panel_label TEXT NOT NULL,
  panel_title TEXT NOT NULL,
  model_id TEXT REFERENCES models(model_id),
  task_id TEXT REFERENCES tasks(task_id),
  system_id TEXT REFERENCES systems(system_id),
  dataset_id TEXT REFERENCES datasets(dataset_id),
  data_query TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (chart_id, panel_id)
);

CREATE TABLE chart_links (
  chart_id TEXT NOT NULL REFERENCES charts(chart_id),
  entity_type TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  role TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (chart_id, entity_type, entity_id, role)
);

CREATE TABLE chart_data_files (
  chart_id TEXT NOT NULL REFERENCES charts(chart_id),
  file_id TEXT NOT NULL,
  display_name TEXT NOT NULL,
  local_path TEXT NOT NULL,
  file_format TEXT NOT NULL,
  file_exists INTEGER NOT NULL DEFAULT 0,
  notes TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (chart_id, file_id)
);

CREATE TABLE chart_provenance (
  chart_id TEXT PRIMARY KEY REFERENCES charts(chart_id),
  manifest_path TEXT NOT NULL DEFAULT '',
  manifest_exists INTEGER NOT NULL DEFAULT 0,
  output_sha256 TEXT,
  input_file_hashes TEXT NOT NULL DEFAULT '{}',
  processing_command TEXT NOT NULL DEFAULT '',
  provenance_status TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE gaps (
  gap_id TEXT PRIMARY KEY,
  priority TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  entity_id TEXT,
  gap_type TEXT NOT NULL,
  description TEXT NOT NULL,
  recommended_next_action TEXT NOT NULL,
  status TEXT NOT NULL,
  source_id TEXT REFERENCES sources(source_id)
);

CREATE TABLE audit_log (
  audit_id INTEGER PRIMARY KEY,
  checked_at TEXT NOT NULL,
  check_name TEXT NOT NULL,
  result TEXT NOT NULL,
  details TEXT NOT NULL
);

CREATE INDEX idx_models_family ON models(family_id);
CREATE INDEX idx_model_groups_major ON model_groups(major_class, sort_order);
CREATE INDEX idx_model_group_members_model ON model_group_members(model_id);
CREATE INDEX idx_model_aliases_alias ON model_aliases(alias);
CREATE INDEX idx_coverage_model_task ON coverage(model_id, task_id);
CREATE INDEX idx_coverage_system ON coverage(system_id);
CREATE INDEX idx_datasets_task ON datasets(task_id);
CREATE INDEX idx_task_groups_layer ON task_groups(task_layer, sort_order);
CREATE INDEX idx_task_group_members_task ON task_group_members(task_id);
CREATE INDEX idx_task_aliases_alias ON task_aliases(alias);
CREATE INDEX idx_gaps_priority ON gaps(priority, status);
CREATE INDEX idx_charts_campaign ON charts(campaign_id);
CREATE INDEX idx_chart_links_entity ON chart_links(entity_type, entity_id);
CREATE INDEX idx_chart_panels_task ON chart_panels(task_id);
CREATE INDEX idx_chart_provenance_status ON chart_provenance(provenance_status);

CREATE VIEW v_chart_catalog AS
WITH link_entities AS (
  SELECT
    chart_id,
    MAX(CASE WHEN entity_type = 'model' THEN entity_id END) AS linked_model_id,
    MAX(CASE WHEN entity_type = 'task' THEN entity_id END) AS linked_task_id,
    MAX(CASE WHEN entity_type = 'system' THEN entity_id END) AS linked_system_id,
    MAX(CASE WHEN entity_type = 'dataset' THEN entity_id END) AS linked_dataset_id
  FROM chart_links
  GROUP BY chart_id
), chart_rows AS (
  SELECT
    c.chart_id, c.title, c.chart_kind, c.chart_status, c.asset_path,
    c.asset_mime, c.asset_exists, c.campaign_id, c.source_id, c.notes,
    cp.panel_id, cp.panel_label, cp.panel_title,
    COALESCE(cp.model_id, le.linked_model_id) AS model_id,
    COALESCE(cp.task_id, le.linked_task_id) AS task_id,
    COALESCE(cp.system_id, le.linked_system_id) AS system_id,
    COALESCE(cp.dataset_id, le.linked_dataset_id) AS dataset_id
  FROM charts c
  LEFT JOIN chart_panels cp ON cp.chart_id = c.chart_id
  LEFT JOIN link_entities le ON le.chart_id = c.chart_id
)
SELECT DISTINCT
  cr.chart_id, cr.title, cr.chart_kind, cr.chart_status, cr.asset_path,
  cr.asset_mime, cr.asset_exists, cr.campaign_id, cr.source_id, cr.notes,
  cr.panel_id, cr.panel_label, cr.panel_title, cr.model_id, mm.preferred_name,
  mm.plot_label, mm.short_label, mm.major_class, mm.subgroup_label,
  cr.task_id, tg.task_group_id, tg.display_name AS task_group_name,
  tg.task_layer, cr.system_id, cr.dataset_id
FROM chart_rows cr
LEFT JOIN model_name_map mm ON mm.model_id = cr.model_id
LEFT JOIN task_group_members tgm ON tgm.task_id = cr.task_id
LEFT JOIN task_groups tg ON tg.task_group_id = tgm.task_group_id;

CREATE VIEW v_model_task_coverage AS
SELECT
  cov.coverage_id, cov.model_id, COALESCE(mm.preferred_name, m.display_name) AS preferred_name,
  mm.plot_label, mm.short_label, mm.major_class, mm.subgroup_label,
  cov.task_id, t.category, t.operator,
  tgm.task_group_id, tg.display_name AS task_group_name, tg.task_layer,
  cov.system_id, cov.campaign_id, cov.dataset_id, cov.scope_label,
  cov.expected_count, cov.completed_count, cov.run_status,
  cov.scientific_status, cov.dft_reference_status, cov.result_summary,
  cov.source_id
FROM coverage cov
LEFT JOIN models m ON m.model_id = cov.model_id
LEFT JOIN model_name_map mm ON mm.model_id = cov.model_id
JOIN tasks t ON t.task_id = cov.task_id
JOIN task_group_members tgm ON tgm.task_id = cov.task_id
JOIN task_groups tg ON tg.task_group_id = tgm.task_group_id;

CREATE VIEW v_model_task_summary AS
SELECT
  m.model_id, COALESCE(mm.preferred_name, m.display_name) AS preferred_name,
  mm.plot_label, mm.short_label, mm.major_class, mm.subgroup_label,
  tg.task_group_id, tg.display_name AS task_group_name, tg.task_layer,
  COUNT(DISTINCT tgm.task_id) AS expected_task_count,
  COUNT(DISTINCT cov.coverage_id) AS coverage_rows,
  COUNT(DISTINCT CASE WHEN cov.run_status LIKE 'COMPLETE%' OR cov.scientific_status LIKE 'COMPLETE%' THEN cov.coverage_id END) AS completed_rows,
  COUNT(DISTINCT CASE WHEN cov.run_status LIKE '%FAIL%' OR cov.scientific_status LIKE '%FAIL%' OR cov.scientific_status LIKE '%EXCLUDE%' THEN cov.coverage_id END) AS failure_rows,
  GROUP_CONCAT(DISTINCT tgm.task_id) AS task_ids,
  GROUP_CONCAT(DISTINCT cov.run_status) AS run_statuses,
  CASE
    WHEN COUNT(DISTINCT cov.coverage_id) = 0 THEN 'MISSING'
    WHEN COUNT(DISTINCT CASE WHEN cov.run_status LIKE 'COMPLETE%' OR cov.scientific_status LIKE 'COMPLETE%' THEN cov.coverage_id END) = COUNT(DISTINCT cov.coverage_id) THEN 'COMPLETE'
    ELSE 'PARTIAL'
  END AS coverage_state
FROM models m
LEFT JOIN model_name_map mm ON mm.model_id = m.model_id
CROSS JOIN task_groups tg
LEFT JOIN task_group_members tgm ON tgm.task_group_id = tg.task_group_id
LEFT JOIN coverage cov ON cov.model_id = m.model_id AND cov.task_id = tgm.task_id
GROUP BY m.model_id, mm.preferred_name, mm.plot_label, mm.short_label,
         mm.major_class, mm.subgroup_label, tg.task_group_id,
         tg.display_name, tg.task_layer;
