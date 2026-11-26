PRAGMA foreign_keys=ON;
PRAGMA application_id=1296183605;
PRAGMA user_version=1;

CREATE TABLE snapshot_metadata (
  metadata_key TEXT PRIMARY KEY,
  metadata_value TEXT NOT NULL
);

CREATE TABLE campaign_stage (
  campaign_id TEXT PRIMARY KEY,
  stage TEXT NOT NULL,
  root_path TEXT NOT NULL,
  parent_campaign_id TEXT,
  model_registry_sha256 TEXT NOT NULL,
  path_registry_sha256 TEXT NOT NULL,
  source_ledger_sha256 TEXT NOT NULL,
  acceptance_scope_sha256 TEXT NOT NULL,
  acceptance_scope_json TEXT NOT NULL,
  relationship_boundary TEXT NOT NULL,
  FOREIGN KEY(parent_campaign_id) REFERENCES campaign_stage(campaign_id)
);

CREATE TABLE source_file (
  relative_path TEXT PRIMARY KEY,
  campaign_id TEXT NOT NULL,
  rockfish_path TEXT NOT NULL UNIQUE,
  size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
  sha256 TEXT NOT NULL CHECK(length(sha256)=64),
  source_role TEXT NOT NULL,
  FOREIGN KEY(campaign_id) REFERENCES campaign_stage(campaign_id)
);

CREATE TABLE campaign_record (
  campaign_id TEXT NOT NULL,
  record_id TEXT NOT NULL,
  stage TEXT NOT NULL,
  source_row_number INTEGER NOT NULL,
  source_record_id TEXT NOT NULL,
  source_job_index TEXT NOT NULL,
  operator TEXT NOT NULL,
  path_id TEXT NOT NULL,
  model_id TEXT NOT NULL,
  materialization_status TEXT NOT NULL CHECK(materialization_status IN ('PRESENT_IN_COLLECTOR','MISSING')),
  run_status TEXT NOT NULL,
  scientific_status TEXT NOT NULL,
  failure_class TEXT NOT NULL,
  leakage_gate TEXT NOT NULL,
  manifest_path TEXT NOT NULL,
  manifest_sha256 TEXT NOT NULL,
  publication_status TEXT NOT NULL,
  raw_row_json TEXT NOT NULL,
  PRIMARY KEY(campaign_id, record_id),
  FOREIGN KEY(campaign_id) REFERENCES campaign_stage(campaign_id)
);

CREATE TABLE record_link (
  parent_campaign_id TEXT NOT NULL,
  parent_record_id TEXT NOT NULL,
  child_campaign_id TEXT NOT NULL,
  child_record_id TEXT NOT NULL,
  relation_type TEXT NOT NULL CHECK(relation_type IN ('RETRY_OF_TIER1_FAILURE','PERTURBATION_OF_TIER1_FAILURE')),
  source_key_field TEXT NOT NULL CHECK(source_key_field='source_record_id'),
  PRIMARY KEY(child_campaign_id, child_record_id),
  FOREIGN KEY(parent_campaign_id, parent_record_id) REFERENCES campaign_record(campaign_id, record_id),
  FOREIGN KEY(child_campaign_id, child_record_id) REFERENCES campaign_record(campaign_id, record_id)
);

CREATE INDEX idx_campaign_record_status ON campaign_record(campaign_id, scientific_status, materialization_status);
CREATE INDEX idx_campaign_record_source ON campaign_record(source_record_id);
CREATE INDEX idx_record_link_parent ON record_link(parent_campaign_id, parent_record_id);
CREATE INDEX idx_source_file_stage ON source_file(campaign_id, source_role);
