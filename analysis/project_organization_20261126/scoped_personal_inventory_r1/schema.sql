PRAGMA foreign_keys = ON;

CREATE TABLE inventory_snapshot (
    snapshot_id TEXT PRIMARY KEY,
    started_utc TEXT NOT NULL,
    finished_utc TEXT,
    host_name TEXT NOT NULL,
    scanner_version TEXT NOT NULL,
    completion_state TEXT NOT NULL CHECK (completion_state IN ('IN_PROGRESS','COMPLETE','COMPLETE_WITH_ERRORS')),
    scope_note TEXT NOT NULL,
    excluded_paths_json TEXT NOT NULL
);

CREATE TABLE inventory_root (
    root_id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL,
    root_path TEXT NOT NULL UNIQUE,
    entry_count INTEGER NOT NULL DEFAULT 0,
    file_count INTEGER NOT NULL DEFAULT 0,
    directory_count INTEGER NOT NULL DEFAULT 0,
    symlink_count INTEGER NOT NULL DEFAULT 0,
    special_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    logical_bytes INTEGER NOT NULL DEFAULT 0,
    allocated_path_sum_bytes INTEGER NOT NULL DEFAULT 0,
    allocated_unique_inode_bytes INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (snapshot_id) REFERENCES inventory_snapshot(snapshot_id)
);

CREATE TABLE inventory_entry (
    root_id TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    parent_relative_path TEXT,
    top_level_name TEXT NOT NULL,
    depth INTEGER NOT NULL CHECK (depth >= 0),
    entry_type TEXT NOT NULL CHECK (entry_type IN ('ROOT','DIR','FILE','SYMLINK','SPECIAL')),
    logical_bytes INTEGER NOT NULL DEFAULT 0 CHECK (logical_bytes >= 0),
    allocated_bytes INTEGER CHECK (allocated_bytes >= 0),
    uid INTEGER,
    gid INTEGER,
    mode_octal TEXT,
    device INTEGER,
    inode INTEGER,
    link_count INTEGER,
    mtime_ns INTEGER,
    PRIMARY KEY (root_id, relative_path),
    FOREIGN KEY (root_id) REFERENCES inventory_root(root_id)
);

CREATE TABLE inventory_error (
    root_id TEXT NOT NULL,
    path TEXT NOT NULL,
    error_class TEXT NOT NULL,
    error_message TEXT NOT NULL,
    PRIMARY KEY (root_id, path),
    FOREIGN KEY (root_id) REFERENCES inventory_root(root_id)
);

CREATE INDEX inventory_entry_by_top_level ON inventory_entry(root_id, top_level_name, entry_type);
CREATE INDEX inventory_entry_by_size ON inventory_entry(root_id, logical_bytes DESC);
CREATE INDEX inventory_entry_by_inode ON inventory_entry(root_id, device, inode);
