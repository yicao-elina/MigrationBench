-- Preserve unknown allocated-byte totals as NULL rather than reporting zero.
PRAGMA foreign_keys=OFF;
BEGIN IMMEDIATE;

ALTER TABLE cleanup_execution_receipt RENAME TO cleanup_execution_receipt_old;

CREATE TABLE cleanup_execution_receipt (
  cleanup_execution_receipt_id INTEGER PRIMARY KEY,
  cleanup_batch_id TEXT NOT NULL REFERENCES cleanup_batch(cleanup_batch_id),
  receipt_source_id INTEGER NOT NULL REFERENCES registry_source(source_id),
  observed_utc TEXT NOT NULL,
  execution_state TEXT NOT NULL CHECK (execution_state IN ('PRECHECKED','COMPLETED','PARTIAL','FAILED')),
  deleted_file_count INTEGER NOT NULL DEFAULT 0 CHECK (deleted_file_count >= 0),
  actual_logical_bytes INTEGER NOT NULL DEFAULT 0 CHECK (actual_logical_bytes >= 0),
  actual_allocated_bytes INTEGER CHECK (actual_allocated_bytes IS NULL OR actual_allocated_bytes >= 0),
  quota_before_observation_id INTEGER REFERENCES quota_observation(quota_observation_id),
  quota_after_observation_id INTEGER REFERENCES quota_observation(quota_observation_id),
  queue_snapshot_source_id INTEGER REFERENCES registry_source(source_id),
  notes TEXT NOT NULL DEFAULT '',
  UNIQUE(cleanup_batch_id, receipt_source_id)
);

INSERT INTO cleanup_execution_receipt (
  cleanup_execution_receipt_id,cleanup_batch_id,receipt_source_id,observed_utc,execution_state,
  deleted_file_count,actual_logical_bytes,actual_allocated_bytes,quota_before_observation_id,
  quota_after_observation_id,queue_snapshot_source_id,notes
)
SELECT
  cleanup_execution_receipt_id,cleanup_batch_id,receipt_source_id,observed_utc,execution_state,
  deleted_file_count,actual_logical_bytes,actual_allocated_bytes,quota_before_observation_id,
  quota_after_observation_id,queue_snapshot_source_id,notes
FROM cleanup_execution_receipt_old;

DROP TABLE cleanup_execution_receipt_old;
CREATE INDEX idx_cleanup_receipt_batch_time
  ON cleanup_execution_receipt(cleanup_batch_id, observed_utc);

INSERT INTO schema_migration(version,migration_file,applied_utc,description)
VALUES (15,'015_cleanup_receipt_allocated_bytes_nullable.sql',
        strftime('%Y-%m-%dT%H:%M:%fZ','now'),
        'Allow cleanup receipts to represent unmeasured allocated bytes as NULL');

COMMIT;
PRAGMA foreign_keys=ON;
