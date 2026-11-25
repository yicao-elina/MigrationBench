-- Explicit, queryable redirects for same-path metadata aliases.
-- Alias artifact rows and historical locations remain available for audit.
CREATE TABLE IF NOT EXISTS artifact_alias (
  alias_artifact_id INTEGER PRIMARY KEY REFERENCES artifact(artifact_id),
  canonical_artifact_id INTEGER NOT NULL REFERENCES artifact(artifact_id),
  alias_reason TEXT NOT NULL,
  evidence_artifact_id INTEGER REFERENCES artifact(artifact_id),
  recorded_utc TEXT NOT NULL,
  CHECK(alias_artifact_id <> canonical_artifact_id)
);

CREATE INDEX IF NOT EXISTS idx_artifact_alias_canonical
  ON artifact_alias(canonical_artifact_id);

INSERT INTO schema_migration(version,migration_file,applied_utc,description)
VALUES(16,'016_artifact_alias_redirects.sql',strftime('%Y-%m-%dT%H:%M:%fZ','now'),
       'Add explicit canonical redirects for same-path artifact aliases while retaining historical rows and locations.')
ON CONFLICT(version) DO NOTHING;
