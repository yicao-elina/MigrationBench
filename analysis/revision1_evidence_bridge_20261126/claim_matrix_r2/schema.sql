PRAGMA foreign_keys=ON;

-- These additive tables live beside the unchanged R1 claim inventory tables.
-- Do not add rows to snapshot_metadata: QCE04 must remain byte-for-byte equal.
CREATE TABLE fig4_crosswalk_metadata (
  metadata_key TEXT PRIMARY KEY,
  metadata_value TEXT NOT NULL
);

CREATE TABLE fig4_r05_summary_crosswalk (
  crosswalk_key TEXT PRIMARY KEY,
  r4_result_id TEXT NOT NULL,
  r4_result_key TEXT NOT NULL UNIQUE,
  panel TEXT NOT NULL CHECK (panel IN ('a','b','c','d','e','f')),
  group_key TEXT NOT NULL CHECK (group_key IN ('foundation','ft_600k','ft_multit','scratch_600k','scratch_multit')),
  observable TEXT NOT NULL CHECK (observable IN ('FIXED_PATH_SINGLE_POINT','SELF_RELAXED_ENDPOINT_ONLY_NEB')),
  path_key TEXT NOT NULL CHECK (path_key IN ('in_gap','gap_to_ql','within_ql')),
  path_class TEXT NOT NULL,
  profile_data_line INTEGER,
  coverage_data_line INTEGER NOT NULL,
  match_status TEXT NOT NULL CHECK (match_status IN ('PRESENT','MISSING_OR_EXCLUDED')),
  n_models INTEGER NOT NULL CHECK (n_models >= 0),
  n_profile_rows INTEGER NOT NULL CHECK (n_profile_rows >= 0),
  r05_scientific_statuses TEXT NOT NULL,
  r05_publication_statuses TEXT NOT NULL,
  source_boundary TEXT NOT NULL,
  r4_protocol_key TEXT NOT NULL,
  r4_evidence_status TEXT NOT NULL,
  r4_scientific_status TEXT NOT NULL,
  mapping_grain TEXT NOT NULL CHECK (mapping_grain='SUMMARY_TO_R05_GROUP_ROW'),
  run_attempt_linkage TEXT NOT NULL CHECK (run_attempt_linkage='NOT_ESTABLISHED'),
  scientific_acceptance TEXT NOT NULL CHECK (scientific_acceptance='NOT_ASSESSED'),
  source_r05_key TEXT NOT NULL,
  UNIQUE (observable, path_key, group_key)
);

CREATE INDEX idx_fig4_crosswalk_panel ON fig4_r05_summary_crosswalk(panel, group_key);
CREATE INDEX idx_fig4_crosswalk_status ON fig4_r05_summary_crosswalk(match_status, observable, path_key);

PRAGMA user_version=1;
