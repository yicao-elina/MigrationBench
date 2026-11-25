-- Source-faithful Revision 1 indexes and scheduler/cleanup crosswalks.
-- Exact source status strings stay separate from catalog assessments.

CREATE TABLE IF NOT EXISTS paper_release_item (
  release_item_id INTEGER PRIMARY KEY,
  release_id TEXT NOT NULL REFERENCES paper_release(release_id),
  source_item_id TEXT NOT NULL,
  section TEXT NOT NULL DEFAULT '',
  panel TEXT NOT NULL DEFAULT '',
  content TEXT NOT NULL,
  final_asset_path TEXT,
  final_asset_sha256 TEXT,
  generator TEXT,
  plotted_data TEXT,
  upstream_data TEXT,
  raw_location TEXT,
  model_refs TEXT,
  calculation_type TEXT,
  source_status TEXT NOT NULL DEFAULT '',
  notes TEXT NOT NULL DEFAULT '',
  item_dir TEXT,
  reproduce_script TEXT,
  reproduction_status TEXT,
  UNIQUE(release_id, source_item_id, section, panel)
);

CREATE TABLE IF NOT EXISTS paper_release_item_evidence (
  release_item_evidence_id INTEGER PRIMARY KEY,
  release_item_id INTEGER NOT NULL REFERENCES paper_release_item(release_item_id),
  artifact_id INTEGER REFERENCES artifact(artifact_id),
  source_path TEXT NOT NULL,
  evidence_role TEXT NOT NULL,
  availability TEXT NOT NULL CHECK (availability IN ('PRESENT','MISSING','EXTERNAL_POINTER','NOT_MATERIALIZED')),
  relationship TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT '',
  UNIQUE(release_item_id, evidence_role, source_path)
);

CREATE TABLE IF NOT EXISTS paper_reproducibility_record (
  reproducibility_record_id INTEGER PRIMARY KEY,
  release_id TEXT NOT NULL REFERENCES paper_release(release_id),
  source_item_id TEXT NOT NULL,
  section TEXT NOT NULL DEFAULT '',
  item_dir TEXT NOT NULL DEFAULT '',
  source_result TEXT NOT NULL,
  scripts TEXT NOT NULL DEFAULT '',
  evidence TEXT NOT NULL DEFAULT '',
  checks_count INTEGER,
  inputs_count INTEGER,
  problems TEXT NOT NULL DEFAULT '',
  linked_release_item_id INTEGER REFERENCES paper_release_item(release_item_id),
  assessment_status TEXT NOT NULL DEFAULT 'PROVENANCE_PARTIAL'
    CHECK (assessment_status IN ('VERIFIED','LINEAGE_SUPPORTED','PROVENANCE_PARTIAL','RECONSTRUCTION','EXCLUDE_FROM_PUBLICATION','FAILURE_MODE','UNKNOWN')),
  notes TEXT NOT NULL DEFAULT '',
  UNIQUE(release_id, source_item_id, section, item_dir)
);

CREATE TABLE IF NOT EXISTS paper_release_claim_observation (
  release_id TEXT NOT NULL REFERENCES paper_release(release_id),
  source_claim_id TEXT NOT NULL,
  reviewer TEXT NOT NULL DEFAULT '',
  question TEXT NOT NULL,
  source_scientific_status TEXT NOT NULL DEFAULT '',
  primary_artifact TEXT NOT NULL DEFAULT '',
  source_or_table TEXT NOT NULL DEFAULT '',
  rebuild_script TEXT NOT NULL DEFAULT '',
  manuscript_location TEXT NOT NULL DEFAULT '',
  response_location TEXT NOT NULL DEFAULT '',
  boundary TEXT NOT NULL DEFAULT '',
  catalog_claim_id TEXT REFERENCES paper_claim(claim_id),
  notes TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(release_id, source_claim_id)
);

CREATE TABLE IF NOT EXISTS scheduler_job_source_observation (
  scheduler_observation_id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES workflow_run(run_id),
  cluster_name TEXT NOT NULL,
  job_id TEXT NOT NULL,
  job_name TEXT NOT NULL DEFAULT '',
  sacct_state TEXT NOT NULL DEFAULT '',
  exit_code TEXT NOT NULL DEFAULT '',
  submit_time_raw TEXT NOT NULL DEFAULT '',
  start_time_raw TEXT NOT NULL DEFAULT '',
  end_time_raw TEXT NOT NULL DEFAULT '',
  workdir TEXT NOT NULL DEFAULT '',
  observed_utc TEXT NOT NULL,
  source_artifact_id INTEGER REFERENCES artifact(artifact_id),
  notes TEXT NOT NULL DEFAULT '',
  UNIQUE(run_id, cluster_name, job_id, observed_utc)
);

CREATE TABLE IF NOT EXISTS cleanup_candidate_context (
  cleanup_batch_id TEXT NOT NULL REFERENCES cleanup_batch(cleanup_batch_id),
  artifact_id INTEGER NOT NULL REFERENCES artifact(artifact_id),
  run_id TEXT NOT NULL REFERENCES workflow_run(run_id),
  candidate_kind TEXT NOT NULL,
  manifest_paths TEXT NOT NULL DEFAULT '',
  manifest_present INTEGER NOT NULL CHECK (manifest_present IN (0,1)),
  manifest_restart_required_flag TEXT NOT NULL DEFAULT '',
  manifest_restart_required_basis TEXT NOT NULL DEFAULT '',
  input_paths TEXT NOT NULL DEFAULT '',
  output_log_paths TEXT NOT NULL DEFAULT '',
  coordinate_evidence_path TEXT NOT NULL,
  coordinate_evidence_sha256 TEXT NOT NULL,
  coordinate_xml_parse TEXT NOT NULL,
  coordinate_nat INTEGER,
  coordinate_evidence_kind TEXT NOT NULL,
  neb_path_evidence TEXT NOT NULL DEFAULT '',
  neb_image_ids TEXT NOT NULL DEFAULT '',
  neb_complete_image_coverage INTEGER NOT NULL CHECK (neb_complete_image_coverage IN (0,1)),
  source_queue_capture_utc TEXT NOT NULL,
  source_queue_status TEXT NOT NULL,
  source_queue_workdir_overlap INTEGER NOT NULL CHECK (source_queue_workdir_overlap IN (0,1)),
  disposition_reason TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(cleanup_batch_id, artifact_id)
);

CREATE INDEX IF NOT EXISTS idx_release_item_lookup
  ON paper_release_item(release_id, source_item_id, section, panel);
CREATE INDEX IF NOT EXISTS idx_release_evidence_path
  ON paper_release_item_evidence(source_path);
CREATE INDEX IF NOT EXISTS idx_release_claim_status
  ON paper_release_claim_observation(release_id, source_scientific_status);
CREATE INDEX IF NOT EXISTS idx_scheduler_observation_job
  ON scheduler_job_source_observation(cluster_name, job_id, observed_utc);
CREATE INDEX IF NOT EXISTS idx_cleanup_context_run
  ON cleanup_candidate_context(run_id, candidate_kind);

INSERT OR IGNORE INTO schema_migration(version, migration_file, applied_utc, description)
VALUES (12, '012_revision1_release_and_scheduler_evidence.sql',
        strftime('%Y-%m-%dT%H:%M:%fZ','now'),
        'Preserve submitted paper result/reproducibility indexes and QE scheduler/cleanup run context with exact source statuses');
