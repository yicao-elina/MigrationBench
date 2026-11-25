-- Immutable, local-only merge/fast-forward results. This records branch
-- integration separately from a GitHub push or deployment.

CREATE TABLE git_merge_observation (
  git_merge_observation_id INTEGER PRIMARY KEY,
  repository_id INTEGER NOT NULL REFERENCES git_repository(repository_id),
  evidence_source_id INTEGER REFERENCES registry_source(source_id),
  observed_utc TEXT NOT NULL,
  target_ref TEXT NOT NULL,
  source_ref TEXT NOT NULL,
  pre_target_sha TEXT,
  source_sha TEXT NOT NULL,
  resulting_sha TEXT,
  merge_result TEXT NOT NULL CHECK (
    merge_result IN ('FAST_FORWARD','MERGE_COMMIT','ALREADY_CONTAINS','CANDIDATE','BLOCKED')
  ),
  worktree_path TEXT,
  worktree_clean INTEGER CHECK (worktree_clean IN (0,1)),
  push_status TEXT NOT NULL DEFAULT 'UNKNOWN'
    CHECK (push_status IN ('NOT_PUSHED','CONFIRMED_PUSHED','UNKNOWN')),
  notes TEXT NOT NULL DEFAULT ''
);

CREATE INDEX idx_git_merge_observation_repo_time
  ON git_merge_observation(repository_id, observed_utc);

INSERT INTO schema_migration(version, migration_file, applied_utc, description)
VALUES (7, '007_git_merge_observations.sql',
        strftime('%Y-%m-%dT%H:%M:%fZ','now'),
        'Record local branch merge/fast-forward outcomes separately from pushes');
