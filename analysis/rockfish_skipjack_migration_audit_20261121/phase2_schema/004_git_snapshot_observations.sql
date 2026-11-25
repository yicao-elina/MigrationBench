-- Additive Git snapshot detail for cached local state and ref ancestry.
-- Apply only after 001_provenance_catalog.sql and the current importer complete.

CREATE TABLE IF NOT EXISTS git_repository_snapshot (
  repository_snapshot_id INTEGER PRIMARY KEY,
  repository_id INTEGER NOT NULL REFERENCES git_repository(repository_id),
  source_id INTEGER NOT NULL REFERENCES registry_source(source_id),
  observed_utc TEXT NOT NULL,
  head_ref TEXT,
  head_sha TEXT,
  status_exit_code INTEGER,
  tracked_dirty_count INTEGER,
  staged_dirty_count INTEGER,
  unstaged_dirty_count INTEGER,
  untracked_file_count INTEGER,
  ignored_matching_count INTEGER,
  worktree_dirty INTEGER CHECK (worktree_dirty IN (0,1)),
  status_error TEXT NOT NULL DEFAULT '',
  fetch_status TEXT NOT NULL CHECK (fetch_status IN ('NOT_RECORDED','NO_REMOTE_CONFIGURED')),
  remote_tip_status TEXT NOT NULL CHECK (remote_tip_status IN ('UNKNOWN_NOT_VERIFIED','NOT_APPLICABLE')),
  shallow_clone INTEGER CHECK (shallow_clone IN (0,1)),
  classification TEXT,
  classification_basis TEXT,
  parent_repo_key TEXT,
  parent_index_relation TEXT,
  notes TEXT NOT NULL DEFAULT '',
  UNIQUE(repository_id, source_id)
);

CREATE TABLE IF NOT EXISTS git_remote_observation (
  remote_observation_id INTEGER PRIMARY KEY,
  repository_snapshot_id INTEGER NOT NULL REFERENCES git_repository_snapshot(repository_snapshot_id),
  remote_name TEXT NOT NULL,
  fetch_url TEXT,
  push_url TEXT,
  fetch_status TEXT NOT NULL CHECK (fetch_status IN ('NOT_RECORDED','NOT_ATTEMPTED','SUCCEEDED','FAILED','UNKNOWN')),
  current_tip_status TEXT NOT NULL CHECK (current_tip_status IN ('UNKNOWN_NOT_VERIFIED','NOT_APPLICABLE')),
  current_tip_sha TEXT,
  notes TEXT NOT NULL DEFAULT '',
  UNIQUE(repository_snapshot_id, remote_name)
);

CREATE TABLE IF NOT EXISTS git_ref_context_snapshot (
  git_ref_observation_id INTEGER PRIMARY KEY REFERENCES git_ref_observation(git_ref_observation_id),
  repository_snapshot_id INTEGER NOT NULL REFERENCES git_repository_snapshot(repository_snapshot_id),
  source_id INTEGER NOT NULL REFERENCES registry_source(source_id),
  is_current_checkout INTEGER CHECK (is_current_checkout IN (0,1)),
  is_local_only_ref INTEGER CHECK (is_local_only_ref IN (0,1)),
  comparison_target TEXT,
  comparison_source TEXT,
  comparison_state TEXT,
  ahead_commits INTEGER,
  behind_commits INTEGER,
  local_mainline_ref TEXT,
  graph_relation_to_mainline TEXT,
  mainline_only_commits INTEGER,
  branch_only_commits INTEGER,
  remote_tip_status TEXT NOT NULL DEFAULT 'UNKNOWN_NOT_VERIFIED'
    CHECK (remote_tip_status IN ('UNKNOWN_NOT_VERIFIED','NOT_APPLICABLE')),
  notes TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS git_branch_pair_graph_snapshot (
  repository_snapshot_id INTEGER NOT NULL REFERENCES git_repository_snapshot(repository_snapshot_id),
  ref_a TEXT NOT NULL,
  ref_b TEXT NOT NULL,
  relation TEXT NOT NULL,
  a_only_commits INTEGER,
  b_only_commits INTEGER,
  common_base_ref_or_oid TEXT,
  notes TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(repository_snapshot_id, ref_a, ref_b)
);

CREATE INDEX IF NOT EXISTS idx_git_repo_snapshot_repo_time
  ON git_repository_snapshot(repository_id, observed_utc);
CREATE INDEX IF NOT EXISTS idx_git_ref_context_snapshot
  ON git_ref_context_snapshot(repository_snapshot_id, local_mainline_ref);
