PRAGMA foreign_keys=ON;
CREATE TABLE snapshot_metadata (
  metadata_key TEXT PRIMARY KEY,
  metadata_value TEXT NOT NULL
);
CREATE TABLE training_run_snapshot (
  training_run_key TEXT PRIMARY KEY,
  seed INTEGER NOT NULL UNIQUE,
  job_id TEXT NOT NULL UNIQUE,
  scheduler_state TEXT NOT NULL,
  exit_code TEXT NOT NULL,
  ended_utc TEXT NOT NULL,
  scheduler_snapshot_utc TEXT NOT NULL,
  source_status_file_sha256 TEXT NOT NULL
);
CREATE TABLE artifact_inventory (
  artifact_path TEXT PRIMARY KEY,
  model_dir TEXT NOT NULL,
  seed INTEGER NOT NULL,
  job_id TEXT NOT NULL,
  file_class TEXT NOT NULL,
  epoch INTEGER,
  size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
  allocated_bytes INTEGER NOT NULL CHECK(allocated_bytes >= 0),
  mtime_utc TEXT NOT NULL,
  inode TEXT NOT NULL,
  hardlink_count INTEGER NOT NULL CHECK(hardlink_count >= 1),
  disposition TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  sha256_status TEXT NOT NULL,
  FOREIGN KEY(seed) REFERENCES training_run_snapshot(seed),
  CHECK(file_class IN ('checkpoint','model_export','compiled_export','provenance_manifest','metrics_or_preflight','training_log','debug_log','training_diagnostic_plot')),
  CHECK(disposition IN ('KEEP','KEEP_DESIGNATED_FINAL','HOLD_PENDING_USER_POLICY')),
  CHECK(sha256_status IN ('NOT_COMPUTED','HASH_NOT_COMPUTED','SHA256_VERIFIED_AGAINST_final_model_provenance_json'))
);
CREATE INDEX idx_checkpoint_artifact_seed_epoch ON artifact_inventory(seed, file_class, epoch);
CREATE INDEX idx_checkpoint_artifact_hash ON artifact_inventory(sha256) WHERE sha256 <> '';
CREATE INDEX idx_checkpoint_artifact_disposition ON artifact_inventory(disposition, file_class);
PRAGMA user_version=1;
