-- Isolated, additive sidecar schema for Revision 1 manifest metadata.
-- Core project/storage/artifact tables mirror phase2_schema/001_provenance_catalog.sql.
PRAGMA foreign_keys = ON;
PRAGMA user_version = 1;

CREATE TABLE registry_source (
  source_id INTEGER PRIMARY KEY,
  source_key TEXT NOT NULL UNIQUE,
  source_path TEXT NOT NULL,
  registry_kind TEXT NOT NULL,
  sha256 TEXT,
  captured_utc TEXT NOT NULL,
  as_of_utc TEXT,
  immutable_snapshot INTEGER NOT NULL DEFAULT 0 CHECK (immutable_snapshot IN (0,1)),
  authority_note TEXT NOT NULL DEFAULT '',
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE project (
  project_id TEXT PRIMARY KEY,
  project_name TEXT NOT NULL,
  owner_scope TEXT NOT NULL CHECK (owner_scope IN ('USER','SHARED','THIRD_PARTY','UNKNOWN')),
  description TEXT NOT NULL DEFAULT '',
  canonical_source_id INTEGER REFERENCES registry_source(source_id),
  status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE','HISTORICAL','ARCHIVED','UNKNOWN'))
);

CREATE TABLE storage_root (
  root_id TEXT PRIMARY KEY,
  cluster_name TEXT NOT NULL,
  filesystem_name TEXT NOT NULL,
  mount_path TEXT NOT NULL,
  scoped_path TEXT NOT NULL UNIQUE,
  owner_user TEXT,
  owner_group TEXT,
  scope_kind TEXT NOT NULL CHECK (scope_kind IN ('USER_SUBTREE','PROJECT_SUBTREE','GROUP_SUBTREE','OTHER')),
  project_id TEXT REFERENCES project(project_id),
  status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE','READ_ONLY','MIGRATED','RETIRED','UNKNOWN')),
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE storage_path (
  path_id INTEGER PRIMARY KEY,
  root_id TEXT NOT NULL REFERENCES storage_root(root_id),
  relative_path TEXT NOT NULL,
  first_seen_utc TEXT NOT NULL,
  UNIQUE(root_id, relative_path)
);

CREATE TABLE artifact (
  artifact_id INTEGER PRIMARY KEY,
  project_id TEXT REFERENCES project(project_id),
  artifact_key TEXT NOT NULL UNIQUE,
  artifact_kind TEXT NOT NULL,
  semantic_role TEXT NOT NULL,
  owner_status TEXT NOT NULL DEFAULT 'UNKNOWN' CHECK (owner_status IN ('USER_CONFIRMED','SHARED_OWNER_CONFIRMED','THIRD_PARTY','UNKNOWN')),
  sha256 TEXT,
  byte_size INTEGER CHECK (byte_size IS NULL OR byte_size >= 0),
  scientific_status TEXT NOT NULL DEFAULT 'UNKNOWN' CHECK (scientific_status IN ('VERIFIED','LINEAGE_SUPPORTED','PROVENANCE_PARTIAL','RECONSTRUCTION','EXCLUDE_FROM_PUBLICATION','FAILURE_MODE','UNKNOWN')),
  retention_class TEXT NOT NULL DEFAULT 'UNCLASSIFIED' CHECK (retention_class IN ('PERMANENT_SOURCE','PUBLICATION_EVIDENCE','REPRODUCIBLE_INPUT','COMPACT_RESULT','REPRODUCIBLE_INTERMEDIATE','CACHE','DUPLICATE_COPY','TEMPORARY','UNCLASSIFIED')),
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE artifact_location (
  location_id INTEGER PRIMARY KEY,
  artifact_id INTEGER NOT NULL REFERENCES artifact(artifact_id),
  path_id INTEGER NOT NULL REFERENCES storage_path(path_id),
  availability TEXT NOT NULL DEFAULT 'PRESENT' CHECK (availability IN ('PRESENT','ARCHIVED','MOVED','REMOVED','MISSING','PLANNED')),
  first_verified_utc TEXT,
  last_verified_utc TEXT,
  UNIQUE(artifact_id, path_id)
);

CREATE TABLE sidecar_source (
  source_key TEXT PRIMARY KEY,
  local_path TEXT NOT NULL,
  sha256 TEXT NOT NULL CHECK (length(sha256)=64),
  byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
  source_kind TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE revalidation_snapshot (
  snapshot_id TEXT PRIMARY KEY,
  targets_sha256 TEXT NOT NULL CHECK (length(targets_sha256)=64),
  results_sha256 TEXT NOT NULL CHECK (length(results_sha256)=64),
  receipt_sha256 TEXT NOT NULL CHECK (length(receipt_sha256)=64),
  captured_utc TEXT NOT NULL,
  host_name TEXT NOT NULL,
  python_version TEXT NOT NULL,
  target_count INTEGER NOT NULL CHECK (target_count >= 0),
  method TEXT NOT NULL,
  scope_note TEXT NOT NULL
);

CREATE TABLE manifest_observation (
  manifest_observation_id INTEGER PRIMARY KEY,
  root_id TEXT NOT NULL REFERENCES storage_root(root_id),
  path_id INTEGER NOT NULL REFERENCES storage_path(path_id),
  artifact_id INTEGER NOT NULL REFERENCES artifact(artifact_id),
  manifest_kind TEXT NOT NULL CHECK (manifest_kind IN ('input','run')),
  top_level_dir TEXT NOT NULL,
  manifest_relpath TEXT NOT NULL,
  selector TEXT NOT NULL,
  recorded_scan_sha256 TEXT NOT NULL CHECK (length(recorded_scan_sha256)=64),
  recorded_scan_bytes INTEGER NOT NULL CHECK (recorded_scan_bytes >= 0),
  scan_read_status TEXT NOT NULL,
  current_verification_status TEXT NOT NULL,
  current_artifact_id INTEGER REFERENCES artifact(artifact_id),
  revalidation_snapshot_id TEXT NOT NULL REFERENCES revalidation_snapshot(snapshot_id),
  UNIQUE(root_id, manifest_relpath)
);

CREATE TABLE manifest_control_observation (
  control_observation_id INTEGER PRIMARY KEY,
  manifest_observation_id INTEGER NOT NULL UNIQUE REFERENCES manifest_observation(manifest_observation_id),
  field_count INTEGER NOT NULL CHECK (field_count > 0),
  field_key_count INTEGER NOT NULL CHECK (field_key_count > 0),
  validation_status TEXT NOT NULL DEFAULT 'UNVERIFIED' CHECK (validation_status='UNVERIFIED'),
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE manifest_control_field (
  control_field_id INTEGER PRIMARY KEY,
  control_observation_id INTEGER NOT NULL REFERENCES manifest_control_observation(control_observation_id),
  field_path TEXT NOT NULL,
  raw_value TEXT NOT NULL,
  value_type TEXT NOT NULL,
  interpretation_status TEXT NOT NULL DEFAULT 'UNVERIFIED' CHECK (interpretation_status='UNVERIFIED')
);

CREATE TABLE manifest_path_reference (
  reference_id INTEGER PRIMARY KEY,
  manifest_observation_id INTEGER NOT NULL REFERENCES manifest_observation(manifest_observation_id),
  reference_index INTEGER NOT NULL,
  field_path TEXT NOT NULL,
  literal_value TEXT NOT NULL,
  previous_resolution TEXT NOT NULL,
  previous_bytes INTEGER,
  previous_target_sha256 TEXT,
  previous_hash_status TEXT NOT NULL,
  scope_class TEXT NOT NULL CHECK (scope_class IN ('PERSONAL_INPUT_ROOT','PERSONAL_RUN_ROOT','OUTSIDE_PERSONAL_ROOTS','UNRESOLVED_RELATIVE','INVALID_PATH')),
  target_path_id INTEGER REFERENCES storage_path(path_id),
  target_artifact_id INTEGER REFERENCES artifact(artifact_id),
  current_verification_status TEXT NOT NULL DEFAULT 'NOT_RECHECKED',
  notes TEXT NOT NULL DEFAULT '',
  UNIQUE(manifest_observation_id, reference_index)
);

CREATE TABLE path_revalidation (
  revalidation_id INTEGER PRIMARY KEY,
  snapshot_id TEXT NOT NULL REFERENCES revalidation_snapshot(snapshot_id),
  path_id INTEGER NOT NULL REFERENCES storage_path(path_id),
  target_kind TEXT NOT NULL CHECK (target_kind IN ('METADATA','REFERENCE')),
  observed_type TEXT NOT NULL,
  current_bytes INTEGER CHECK (current_bytes IS NULL OR current_bytes >= 0),
  current_sha256 TEXT,
  expected_bytes INTEGER CHECK (expected_bytes IS NULL OR expected_bytes >= 0),
  expected_sha256 TEXT,
  mtime_ns TEXT,
  verification_status TEXT NOT NULL,
  symlink_target TEXT,
  error_text TEXT NOT NULL DEFAULT '',
  UNIQUE(snapshot_id, path_id)
);

CREATE TABLE sidecar_import (
  import_id TEXT PRIMARY KEY,
  source_scan_sha256 TEXT NOT NULL CHECK (length(source_scan_sha256)=64),
  importer_sha256 TEXT NOT NULL CHECK (length(importer_sha256)=64),
  schema_sha256 TEXT NOT NULL CHECK (length(schema_sha256)=64),
  canonical_main_commit TEXT NOT NULL,
  imported_utc TEXT NOT NULL,
  metadata_observation_count INTEGER NOT NULL,
  control_observation_count INTEGER NOT NULL,
  control_field_count INTEGER NOT NULL,
  path_reference_count INTEGER NOT NULL,
  revalidation_path_count INTEGER NOT NULL,
  quick_check TEXT NOT NULL,
  foreign_key_violations INTEGER NOT NULL,
  scientific_boundary TEXT NOT NULL
);

CREATE INDEX idx_manifest_kind ON manifest_observation(manifest_kind);
CREATE INDEX idx_manifest_sha ON manifest_observation(recorded_scan_sha256);
CREATE INDEX idx_control_field_path ON manifest_control_field(field_path);
CREATE INDEX idx_path_reference_scope ON manifest_path_reference(scope_class, current_verification_status);
CREATE INDEX idx_path_reference_value ON manifest_path_reference(literal_value);
CREATE INDEX idx_path_revalidation_status ON path_revalidation(verification_status);
