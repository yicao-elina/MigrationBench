-- Phase 2 additive provenance catalog. Apply to a NEW SQLite database.
-- Existing registries and cleanup_audit.sqlite are source records; do not overwrite them.
PRAGMA foreign_keys = ON;

CREATE TABLE schema_migration (
  version INTEGER PRIMARY KEY,
  migration_file TEXT NOT NULL UNIQUE,
  applied_utc TEXT NOT NULL,
  description TEXT NOT NULL
);

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
  status TEXT NOT NULL DEFAULT 'ACTIVE'
    CHECK (status IN ('ACTIVE','HISTORICAL','ARCHIVED','UNKNOWN'))
);

CREATE TABLE campaign (
  campaign_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES project(project_id),
  campaign_name TEXT NOT NULL,
  source_campaign_key TEXT,
  start_date TEXT,
  end_date TEXT,
  status TEXT NOT NULL DEFAULT 'UNKNOWN'
    CHECK (status IN ('ACTIVE','COMPLETE','FAILED','SUPERSEDED','ARCHIVED','UNKNOWN')),
  notes TEXT NOT NULL DEFAULT '',
  UNIQUE(project_id, source_campaign_key)
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
  status TEXT NOT NULL DEFAULT 'ACTIVE'
    CHECK (status IN ('ACTIVE','READ_ONLY','MIGRATED','RETIRED','UNKNOWN')),
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE quota_observation (
  quota_observation_id INTEGER PRIMARY KEY,
  root_id TEXT NOT NULL REFERENCES storage_root(root_id),
  observed_utc TEXT NOT NULL,
  quota_subject_kind TEXT NOT NULL CHECK (quota_subject_kind IN ('USER','GROUP','FILESYSTEM')),
  quota_subject TEXT NOT NULL,
  used_bytes INTEGER CHECK (used_bytes IS NULL OR used_bytes >= 0),
  quota_bytes INTEGER CHECK (quota_bytes IS NULL OR quota_bytes >= 0),
  hard_limit_bytes INTEGER CHECK (hard_limit_bytes IS NULL OR hard_limit_bytes >= 0),
  in_doubt_bytes INTEGER CHECK (in_doubt_bytes IS NULL OR in_doubt_bytes >= 0),
  grace_state TEXT,
  source_command TEXT NOT NULL,
  raw_output_path TEXT,
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE inventory_snapshot (
  snapshot_id TEXT PRIMARY KEY,
  root_id TEXT NOT NULL REFERENCES storage_root(root_id),
  started_utc TEXT NOT NULL,
  finished_utc TEXT,
  method TEXT NOT NULL,
  tool_version TEXT,
  completion_state TEXT NOT NULL
    CHECK (completion_state IN ('COMPLETE','PARTIAL','FAILED')),
  files_seen INTEGER,
  directories_seen INTEGER,
  symlinks_seen INTEGER,
  unreadable_count INTEGER NOT NULL DEFAULT 0,
  logical_bytes INTEGER,
  allocated_bytes INTEGER,
  manifest_path TEXT,
  manifest_sha256 TEXT,
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE storage_path (
  path_id INTEGER PRIMARY KEY,
  root_id TEXT NOT NULL REFERENCES storage_root(root_id),
  relative_path TEXT NOT NULL,
  first_seen_utc TEXT NOT NULL,
  UNIQUE(root_id, relative_path)
);

CREATE TABLE inventory_observation (
  observation_id INTEGER PRIMARY KEY,
  snapshot_id TEXT NOT NULL REFERENCES inventory_snapshot(snapshot_id),
  path_id INTEGER NOT NULL REFERENCES storage_path(path_id),
  entry_type TEXT NOT NULL CHECK (entry_type IN ('FILE','DIRECTORY','SYMLINK','OTHER')),
  uid INTEGER,
  gid INTEGER,
  mode_octal TEXT,
  logical_bytes INTEGER CHECK (logical_bytes IS NULL OR logical_bytes >= 0),
  allocated_bytes INTEGER CHECK (allocated_bytes IS NULL OR allocated_bytes >= 0),
  mtime_ns INTEGER,
  inode TEXT,
  device TEXT,
  symlink_target TEXT,
  sha256 TEXT,
  hash_scope TEXT NOT NULL DEFAULT 'NONE'
    CHECK (hash_scope IN ('NONE','FULL_FILE','CONTENT_SAMPLE','MANIFEST_ONLY')),
  observation_status TEXT NOT NULL DEFAULT 'PRESENT'
    CHECK (observation_status IN ('PRESENT','UNREADABLE','VANISHED_DURING_SCAN','EXCLUDED')),
  UNIQUE(snapshot_id, path_id)
);

CREATE TABLE workflow_run (
  run_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES project(project_id),
  campaign_id TEXT REFERENCES campaign(campaign_id),
  run_label TEXT NOT NULL,
  workflow_type TEXT NOT NULL,
  method TEXT,
  system_key TEXT,
  source_root_id TEXT REFERENCES storage_root(root_id),
  source_relative_path TEXT,
  scheduler_state TEXT NOT NULL DEFAULT 'UNKNOWN'
    CHECK (scheduler_state IN ('PENDING','RUNNING','COMPLETED','FAILED','CANCELLED','TIMEOUT','UNKNOWN')),
  scientific_status TEXT NOT NULL DEFAULT 'PROVENANCE_PARTIAL'
    CHECK (scientific_status IN ('VERIFIED','LINEAGE_SUPPORTED','PROVENANCE_PARTIAL','RECONSTRUCTION','EXCLUDE_FROM_PUBLICATION','FAILURE_MODE','UNKNOWN')),
  start_utc TEXT,
  end_utc TEXT,
  supersedes_run_id TEXT REFERENCES workflow_run(run_id),
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE slurm_job (
  slurm_job_key INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES workflow_run(run_id),
  cluster_name TEXT NOT NULL,
  job_id TEXT NOT NULL,
  array_task_id TEXT NOT NULL DEFAULT '',
  submitted_or_started_utc TEXT NOT NULL,
  job_name TEXT,
  partition_name TEXT,
  account_name TEXT,
  UNIQUE(cluster_name, job_id, array_task_id, submitted_or_started_utc)
);

CREATE TABLE slurm_state_observation (
  state_observation_id INTEGER PRIMARY KEY,
  slurm_job_key INTEGER NOT NULL REFERENCES slurm_job(slurm_job_key),
  observed_utc TEXT NOT NULL,
  state TEXT NOT NULL,
  exit_code TEXT,
  start_utc TEXT,
  end_utc TEXT,
  elapsed_seconds INTEGER,
  source_command TEXT NOT NULL,
  raw_record_path TEXT,
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE artifact (
  artifact_id INTEGER PRIMARY KEY,
  project_id TEXT REFERENCES project(project_id),
  artifact_key TEXT NOT NULL UNIQUE,
  artifact_kind TEXT NOT NULL,
  semantic_role TEXT NOT NULL,
  owner_status TEXT NOT NULL DEFAULT 'UNKNOWN'
    CHECK (owner_status IN ('USER_CONFIRMED','SHARED_OWNER_CONFIRMED','THIRD_PARTY','UNKNOWN')),
  sha256 TEXT,
  byte_size INTEGER CHECK (byte_size IS NULL OR byte_size >= 0),
  scientific_status TEXT NOT NULL DEFAULT 'UNKNOWN'
    CHECK (scientific_status IN ('VERIFIED','LINEAGE_SUPPORTED','PROVENANCE_PARTIAL','RECONSTRUCTION','EXCLUDE_FROM_PUBLICATION','FAILURE_MODE','UNKNOWN')),
  retention_class TEXT NOT NULL DEFAULT 'UNCLASSIFIED'
    CHECK (retention_class IN ('PERMANENT_SOURCE','PUBLICATION_EVIDENCE','REPRODUCIBLE_INPUT','COMPACT_RESULT','REPRODUCIBLE_INTERMEDIATE','CACHE','DUPLICATE_COPY','TEMPORARY','UNCLASSIFIED')),
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE artifact_location (
  location_id INTEGER PRIMARY KEY,
  artifact_id INTEGER NOT NULL REFERENCES artifact(artifact_id),
  path_id INTEGER NOT NULL REFERENCES storage_path(path_id),
  availability TEXT NOT NULL DEFAULT 'PRESENT'
    CHECK (availability IN ('PRESENT','ARCHIVED','MOVED','REMOVED','MISSING','PLANNED')),
  first_verified_utc TEXT,
  last_verified_utc TEXT,
  UNIQUE(artifact_id, path_id)
);

CREATE TABLE run_artifact (
  run_id TEXT NOT NULL REFERENCES workflow_run(run_id),
  artifact_id INTEGER NOT NULL REFERENCES artifact(artifact_id),
  role TEXT NOT NULL,
  direction TEXT NOT NULL CHECK (direction IN ('INPUT','OUTPUT','INTERMEDIATE','LOG','PROVENANCE','ENVIRONMENT','MODEL','DATASET','FIGURE','TABLE')),
  required_for_reproduction INTEGER NOT NULL DEFAULT 0 CHECK (required_for_reproduction IN (0,1)),
  notes TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(run_id, artifact_id, role)
);

CREATE TABLE run_parent (
  child_run_id TEXT NOT NULL REFERENCES workflow_run(run_id),
  parent_run_id TEXT NOT NULL REFERENCES workflow_run(run_id),
  relationship TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(child_run_id, parent_run_id, relationship),
  CHECK(child_run_id <> parent_run_id)
);

CREATE TABLE reproduction_recipe (
  recipe_id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES workflow_run(run_id),
  recipe_version TEXT NOT NULL,
  exact_command TEXT NOT NULL,
  working_directory TEXT NOT NULL,
  executable_version TEXT,
  environment_description TEXT,
  module_list TEXT,
  random_seed TEXT,
  script_artifact_id INTEGER REFERENCES artifact(artifact_id),
  scheduler_script_artifact_id INTEGER REFERENCES artifact(artifact_id),
  input_manifest_artifact_id INTEGER REFERENCES artifact(artifact_id),
  recipe_status TEXT NOT NULL DEFAULT 'DOCUMENTED'
    CHECK (recipe_status IN ('DOCUMENTED','REVIEWED','REPRODUCED','INCOMPLETE','BLOCKED')),
  last_reproduced_utc TEXT,
  validation_note TEXT NOT NULL DEFAULT '',
  UNIQUE(run_id, recipe_version)
);

CREATE TABLE recipe_input (
  recipe_id INTEGER NOT NULL REFERENCES reproduction_recipe(recipe_id),
  artifact_id INTEGER NOT NULL REFERENCES artifact(artifact_id),
  input_role TEXT NOT NULL,
  PRIMARY KEY(recipe_id, artifact_id, input_role)
);

CREATE TABLE derivation_edge (
  derivation_id INTEGER PRIMARY KEY,
  parent_artifact_id INTEGER NOT NULL REFERENCES artifact(artifact_id),
  child_artifact_id INTEGER NOT NULL REFERENCES artifact(artifact_id),
  run_id TEXT REFERENCES workflow_run(run_id),
  operation TEXT NOT NULL,
  tool_name TEXT,
  tool_version TEXT,
  script_artifact_id INTEGER REFERENCES artifact(artifact_id),
  parameter_json TEXT,
  provenance_status TEXT NOT NULL DEFAULT 'PROVENANCE_PARTIAL'
    CHECK (provenance_status IN ('VERIFIED','LINEAGE_SUPPORTED','PROVENANCE_PARTIAL','RECONSTRUCTION','UNKNOWN')),
  notes TEXT NOT NULL DEFAULT '',
  CHECK(parent_artifact_id <> child_artifact_id)
);

CREATE TABLE quality_gate (
  run_id TEXT NOT NULL REFERENCES workflow_run(run_id),
  gate_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('PASS','FAIL','NOT_RUN','UNKNOWN','NOT_APPLICABLE')),
  checked_utc TEXT,
  evidence_artifact_id INTEGER REFERENCES artifact(artifact_id),
  rule_description TEXT,
  notes TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(run_id, gate_id)
);

CREATE TABLE paper_release (
  release_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES project(project_id),
  release_label TEXT NOT NULL,
  manuscript_commit TEXT,
  repository_id INTEGER REFERENCES git_repository(repository_id),
  overleaf_version TEXT,
  released_utc TEXT,
  status TEXT NOT NULL DEFAULT 'DRAFT'
    CHECK (status IN ('DRAFT','REVIEW','RELEASED','SUPERSEDED','UNKNOWN')),
  manifest_artifact_id INTEGER REFERENCES artifact(artifact_id),
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE paper_claim (
  claim_id TEXT PRIMARY KEY,
  release_id TEXT REFERENCES paper_release(release_id),
  manuscript_path TEXT,
  section_or_panel TEXT,
  claim_text TEXT NOT NULL,
  scientific_status TEXT NOT NULL DEFAULT 'PROVENANCE_PARTIAL'
    CHECK (scientific_status IN ('VERIFIED','LINEAGE_SUPPORTED','PROVENANCE_PARTIAL','RECONSTRUCTION','EXCLUDE_FROM_PUBLICATION','FAILURE_MODE','UNKNOWN')),
  publication_action TEXT NOT NULL DEFAULT 'PENDING'
    CHECK (publication_action IN ('INCLUDE','QUALIFY','EXCLUDE','PENDING')),
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE claim_evidence (
  claim_id TEXT NOT NULL REFERENCES paper_claim(claim_id),
  artifact_id INTEGER REFERENCES artifact(artifact_id),
  run_id TEXT REFERENCES workflow_run(run_id),
  evidence_role TEXT NOT NULL,
  support_status TEXT NOT NULL DEFAULT 'UNKNOWN'
    CHECK (support_status IN ('SUPPORTS','QUALIFIES','CONTRADICTS','EXCLUDED','UNKNOWN')),
  notes TEXT NOT NULL DEFAULT '',
  CHECK(artifact_id IS NOT NULL OR run_id IS NOT NULL),
  UNIQUE(claim_id, artifact_id, run_id, evidence_role)
);

CREATE TABLE registry_entity_ref (
  external_ref_id INTEGER PRIMARY KEY,
  source_id INTEGER NOT NULL REFERENCES registry_source(source_id),
  entity_type TEXT NOT NULL,
  entity_key TEXT NOT NULL,
  catalog_entity_type TEXT NOT NULL,
  catalog_entity_key TEXT NOT NULL,
  relationship TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT '',
  UNIQUE(source_id, entity_type, entity_key, catalog_entity_type, catalog_entity_key, relationship)
);

CREATE TABLE git_repository (
  repository_id INTEGER PRIMARY KEY,
  repository_key TEXT NOT NULL UNIQUE,
  local_path TEXT,
  remote_url TEXT,
  project_id TEXT REFERENCES project(project_id),
  ownership_status TEXT NOT NULL DEFAULT 'UNKNOWN'
    CHECK (ownership_status IN ('USER','SHARED','THIRD_PARTY','UNKNOWN')),
  canonical_branch TEXT,
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE git_ref_observation (
  git_ref_observation_id INTEGER PRIMARY KEY,
  repository_id INTEGER NOT NULL REFERENCES git_repository(repository_id),
  observed_utc TEXT NOT NULL,
  ref_name TEXT NOT NULL,
  ref_kind TEXT NOT NULL CHECK (ref_kind IN ('BRANCH','TAG','REMOTE_BRANCH','DETACHED')),
  commit_sha TEXT NOT NULL,
  upstream_ref TEXT,
  is_merged_to_canonical INTEGER CHECK (is_merged_to_canonical IN (0,1)),
  dirty_worktree INTEGER CHECK (dirty_worktree IN (0,1)),
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE run_git_binding (
  run_id TEXT NOT NULL REFERENCES workflow_run(run_id),
  repository_id INTEGER NOT NULL REFERENCES git_repository(repository_id),
  commit_sha TEXT NOT NULL,
  binding_role TEXT NOT NULL DEFAULT 'CODE',
  evidence_artifact_id INTEGER REFERENCES artifact(artifact_id),
  provenance_status TEXT NOT NULL DEFAULT 'PROVENANCE_PARTIAL'
    CHECK (provenance_status IN ('VERIFIED','LINEAGE_SUPPORTED','PROVENANCE_PARTIAL','UNKNOWN')),
  PRIMARY KEY(run_id, repository_id, commit_sha, binding_role)
);

CREATE TABLE duplicate_group (
  duplicate_group_id INTEGER PRIMARY KEY,
  detection_run_id TEXT NOT NULL,
  match_method TEXT NOT NULL CHECK (match_method IN ('SHA256','BYTE_COMPARE','INODE','SIZE_AND_NAME','HEURISTIC')),
  verification_status TEXT NOT NULL DEFAULT 'UNVERIFIED'
    CHECK (verification_status IN ('VERIFIED_IDENTICAL','LIKELY','NOT_IDENTICAL','UNVERIFIED')),
  disposition TEXT NOT NULL DEFAULT 'PENDING'
    CHECK (disposition IN ('KEEP_BOTH','KEEP_CANONICAL','ARCHIVE_COPY','DELETE_COPY','PENDING')),
  rationale TEXT NOT NULL DEFAULT ''
);

CREATE TABLE duplicate_member (
  duplicate_group_id INTEGER NOT NULL REFERENCES duplicate_group(duplicate_group_id),
  artifact_id INTEGER NOT NULL REFERENCES artifact(artifact_id),
  canonical_member INTEGER NOT NULL DEFAULT 0 CHECK (canonical_member IN (0,1)),
  PRIMARY KEY(duplicate_group_id, artifact_id)
);

CREATE TABLE cleanup_batch (
  cleanup_batch_id TEXT PRIMARY KEY,
  created_utc TEXT NOT NULL,
  source_root_id TEXT REFERENCES storage_root(root_id),
  policy TEXT NOT NULL,
  rationale TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'PLANNED'
    CHECK (status IN ('PLANNED','EVIDENCE_READY','EXECUTING','COMPLETED','PARTIAL','CANCELLED')),
  authorization_note TEXT NOT NULL DEFAULT '',
  expected_bytes INTEGER,
  actual_bytes INTEGER,
  completed_utc TEXT,
  evidence_artifact_id INTEGER REFERENCES artifact(artifact_id)
);

CREATE TABLE cleanup_item (
  cleanup_batch_id TEXT NOT NULL REFERENCES cleanup_batch(cleanup_batch_id),
  artifact_id INTEGER NOT NULL REFERENCES artifact(artifact_id),
  disposition TEXT NOT NULL CHECK (disposition IN ('DELETE','RETAIN','HOLD','ARCHIVE')),
  decision_reason TEXT NOT NULL,
  owner_clearance TEXT NOT NULL DEFAULT 'UNKNOWN'
    CHECK (owner_clearance IN ('CLEARED','NOT_REQUIRED','UNKNOWN','BLOCKED')),
  pre_delete_sha256 TEXT,
  pre_delete_size_bytes INTEGER,
  pre_delete_mtime_ns INTEGER,
  deletion_state TEXT NOT NULL DEFAULT 'PLANNED'
    CHECK (deletion_state IN ('PLANNED','EVIDENCE_READY','DELETED','VERIFIED','FAILED','HELD')),
  deleted_utc TEXT,
  actual_bytes INTEGER,
  legacy_cleanup_source_id INTEGER REFERENCES registry_source(source_id),
  legacy_cleanup_table TEXT,
  legacy_cleanup_row_key TEXT,
  notes TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(cleanup_batch_id, artifact_id)
);

-- A delete must have explicit ownership clearance. Shared/collaborator/unknown
-- content stays held unless a later migration adds a specific owner release.
CREATE TRIGGER cleanup_owner_guard_insert
BEFORE INSERT ON cleanup_item
WHEN NEW.disposition = 'DELETE' AND
     (NEW.owner_clearance NOT IN ('CLEARED','NOT_REQUIRED') OR
      (SELECT owner_status FROM artifact WHERE artifact_id = NEW.artifact_id) <> 'USER_CONFIRMED')
BEGIN
  SELECT RAISE(ABORT, 'delete blocked: ownership is not cleared and user-confirmed');
END;

CREATE TRIGGER cleanup_owner_guard_update
BEFORE UPDATE OF disposition, owner_clearance, deletion_state ON cleanup_item
WHEN NEW.disposition = 'DELETE' AND
     (NEW.owner_clearance NOT IN ('CLEARED','NOT_REQUIRED') OR
      (SELECT owner_status FROM artifact WHERE artifact_id = NEW.artifact_id) <> 'USER_CONFIRMED')
BEGIN
  SELECT RAISE(ABORT, 'delete blocked: ownership is not cleared and user-confirmed');
END;

-- Reproducible calculation intermediates can only be staged for deletion after
-- a documented recipe exists, scripts and input manifest are catalogued, and
-- essential input/output evidence has been linked. Human review still decides.
CREATE TRIGGER cleanup_reproducibility_guard_insert
BEFORE INSERT ON cleanup_item
WHEN NEW.disposition = 'DELETE' AND
     (SELECT retention_class FROM artifact WHERE artifact_id = NEW.artifact_id) = 'REPRODUCIBLE_INTERMEDIATE' AND
     NOT EXISTS (
       SELECT 1
       FROM run_artifact ra
       JOIN reproduction_recipe rr ON rr.run_id = ra.run_id
       WHERE ra.artifact_id = NEW.artifact_id
         AND ra.direction = 'INTERMEDIATE'
         AND rr.recipe_status IN ('DOCUMENTED','REVIEWED','REPRODUCED')
         AND rr.script_artifact_id IS NOT NULL
         AND rr.scheduler_script_artifact_id IS NOT NULL
         AND rr.input_manifest_artifact_id IS NOT NULL
         AND EXISTS (SELECT 1 FROM artifact_location l WHERE l.artifact_id = rr.script_artifact_id AND l.availability IN ('PRESENT','ARCHIVED'))
         AND EXISTS (SELECT 1 FROM artifact_location l WHERE l.artifact_id = rr.scheduler_script_artifact_id AND l.availability IN ('PRESENT','ARCHIVED'))
         AND EXISTS (SELECT 1 FROM artifact_location l WHERE l.artifact_id = rr.input_manifest_artifact_id AND l.availability IN ('PRESENT','ARCHIVED'))
         AND EXISTS (SELECT 1 FROM run_artifact x JOIN artifact_location l USING (artifact_id) WHERE x.run_id = ra.run_id AND x.direction = 'INPUT' AND l.availability IN ('PRESENT','ARCHIVED'))
         AND EXISTS (SELECT 1 FROM run_artifact x JOIN artifact_location l USING (artifact_id) WHERE x.run_id = ra.run_id AND x.direction = 'OUTPUT' AND l.availability IN ('PRESENT','ARCHIVED'))
     )
BEGIN
  SELECT RAISE(ABORT, 'delete blocked: reproducible-intermediate evidence chain incomplete');
END;

CREATE TRIGGER cleanup_reproducibility_guard_update
BEFORE UPDATE OF disposition, deletion_state ON cleanup_item
WHEN NEW.disposition = 'DELETE' AND
     (SELECT retention_class FROM artifact WHERE artifact_id = NEW.artifact_id) = 'REPRODUCIBLE_INTERMEDIATE' AND
     NOT EXISTS (
       SELECT 1
       FROM run_artifact ra
       JOIN reproduction_recipe rr ON rr.run_id = ra.run_id
       WHERE ra.artifact_id = NEW.artifact_id
         AND ra.direction = 'INTERMEDIATE'
         AND rr.recipe_status IN ('DOCUMENTED','REVIEWED','REPRODUCED')
         AND rr.script_artifact_id IS NOT NULL
         AND rr.scheduler_script_artifact_id IS NOT NULL
         AND rr.input_manifest_artifact_id IS NOT NULL
         AND EXISTS (SELECT 1 FROM artifact_location l WHERE l.artifact_id = rr.script_artifact_id AND l.availability IN ('PRESENT','ARCHIVED'))
         AND EXISTS (SELECT 1 FROM artifact_location l WHERE l.artifact_id = rr.scheduler_script_artifact_id AND l.availability IN ('PRESENT','ARCHIVED'))
         AND EXISTS (SELECT 1 FROM artifact_location l WHERE l.artifact_id = rr.input_manifest_artifact_id AND l.availability IN ('PRESENT','ARCHIVED'))
         AND EXISTS (SELECT 1 FROM run_artifact x JOIN artifact_location l USING (artifact_id) WHERE x.run_id = ra.run_id AND x.direction = 'INPUT' AND l.availability IN ('PRESENT','ARCHIVED'))
         AND EXISTS (SELECT 1 FROM run_artifact x JOIN artifact_location l USING (artifact_id) WHERE x.run_id = ra.run_id AND x.direction = 'OUTPUT' AND l.availability IN ('PRESENT','ARCHIVED'))
     )
BEGIN
  SELECT RAISE(ABORT, 'delete blocked: reproducible-intermediate evidence chain incomplete');
END;

CREATE INDEX idx_path_root_rel ON storage_path(root_id, relative_path);
CREATE INDEX idx_obs_snapshot ON inventory_observation(snapshot_id, observation_status);
CREATE INDEX idx_run_project_campaign ON workflow_run(project_id, campaign_id, scientific_status);
CREATE INDEX idx_job_run ON slurm_job(run_id);
CREATE INDEX idx_artifact_hash ON artifact(sha256);
CREATE INDEX idx_artifact_retention ON artifact(retention_class, owner_status);
CREATE INDEX idx_run_artifact_role ON run_artifact(run_id, role, direction);
CREATE INDEX idx_claim_status ON paper_claim(scientific_status, publication_action);
CREATE INDEX idx_cleanup_state ON cleanup_item(cleanup_batch_id, disposition, deletion_state);

INSERT INTO schema_migration(version, migration_file, applied_utc, description)
VALUES (1, '001_provenance_catalog.sql', strftime('%Y-%m-%dT%H:%M:%fZ','now'),
        'Create additive cross-project storage, run, artifact, paper, Git, duplicate, and cleanup catalog');
