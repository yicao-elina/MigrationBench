-- Keep multiple reviewer questions that share one source claim label as distinct rows.
BEGIN IMMEDIATE;

ALTER TABLE paper_release_claim_observation RENAME TO paper_release_claim_observation_v12;

CREATE TABLE paper_release_claim_observation (
  release_id TEXT NOT NULL REFERENCES paper_release(release_id),
  source_row_key TEXT NOT NULL,
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
  PRIMARY KEY(release_id, source_row_key)
);

INSERT INTO paper_release_claim_observation (
  release_id, source_row_key, source_claim_id, reviewer, question,
  source_scientific_status, primary_artifact, source_or_table, rebuild_script,
  manuscript_location, response_location, boundary, catalog_claim_id, notes
)
SELECT release_id, source_claim_id || '#legacy', source_claim_id, reviewer, question,
       source_scientific_status, primary_artifact, source_or_table, rebuild_script,
       manuscript_location, response_location, boundary, catalog_claim_id, notes
FROM paper_release_claim_observation_v12;

DROP TABLE paper_release_claim_observation_v12;
CREATE INDEX IF NOT EXISTS idx_release_claim_status
  ON paper_release_claim_observation(release_id, source_scientific_status);

INSERT INTO schema_migration(version, migration_file, applied_utc, description)
VALUES (13, '013_duplicate_source_claim_rows.sql', strftime('%Y-%m-%dT%H:%M:%fZ','now'),
        'Retain multiple distinct reviewer questions under a repeated source claim label');

COMMIT;
