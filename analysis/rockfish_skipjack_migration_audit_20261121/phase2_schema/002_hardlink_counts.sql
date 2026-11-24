-- Preserve GPFS/local hard-link counts for reuse and cache ownership checks.
ALTER TABLE inventory_observation ADD COLUMN link_count INTEGER;
CREATE INDEX idx_obs_device_inode ON inventory_observation(device, inode);

INSERT INTO schema_migration(version, migration_file, applied_utc, description)
VALUES (2, '002_hardlink_counts.sql', strftime('%Y-%m-%dT%H:%M:%fZ','now'),
        'Record filesystem link counts and index device/inode identity for hard-link analysis');
