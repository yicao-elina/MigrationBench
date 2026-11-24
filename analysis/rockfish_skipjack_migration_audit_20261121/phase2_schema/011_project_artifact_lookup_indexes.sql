-- Speed up project-scoped artifact search and reverse path lookup.
-- These indexes preserve all artifact and location rows; no provenance data
-- or identity rules change.
BEGIN IMMEDIATE;

CREATE INDEX IF NOT EXISTS idx_artifact_project_key
    ON artifact(project_id, artifact_key);

CREATE INDEX IF NOT EXISTS idx_artifact_location_path_artifact
    ON artifact_location(path_id, artifact_id);

INSERT INTO schema_migration(version, migration_file, applied_utc, description)
VALUES(11, '011_project_artifact_lookup_indexes.sql', CURRENT_TIMESTAMP,
       'Add indexes for project-scoped artifact search and reverse artifact-location lookup.');

COMMIT;
