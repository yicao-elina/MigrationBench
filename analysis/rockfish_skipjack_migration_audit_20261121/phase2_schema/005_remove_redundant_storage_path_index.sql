-- Drop the ordinary duplicate of storage_path's UNIQUE(root_id,relative_path)
-- auto-index. The UNIQUE constraint remains the integrity guard and lookup index.
-- Apply only after the active inventory importer releases the catalog lock.

DROP INDEX IF EXISTS idx_path_root_rel;

INSERT INTO schema_migration(version, migration_file, applied_utc, description)
VALUES (5, '005_remove_redundant_storage_path_index.sql',
        strftime('%Y-%m-%dT%H:%M:%fZ','now'),
        'Remove duplicate non-unique storage_path index while retaining the UNIQUE(root_id,relative_path) auto-index');
