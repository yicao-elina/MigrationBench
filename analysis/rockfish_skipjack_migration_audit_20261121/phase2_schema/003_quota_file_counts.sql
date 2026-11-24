-- Keep inode/file counts alongside block quotas for precise storage reviews.
ALTER TABLE quota_observation ADD COLUMN files_used INTEGER;
ALTER TABLE quota_observation ADD COLUMN files_quota INTEGER;
ALTER TABLE quota_observation ADD COLUMN files_limit INTEGER;
ALTER TABLE quota_observation ADD COLUMN files_in_doubt INTEGER;

INSERT INTO schema_migration(version, migration_file, applied_utc, description)
VALUES (3, '003_quota_file_counts.sql', strftime('%Y-%m-%dT%H:%M:%fZ','now'),
        'Record GPFS user/group file counts and file-count quotas with byte quotas');
