PRAGMA foreign_keys = ON;

CREATE TABLE inventory_snapshot (
    snapshot_id TEXT PRIMARY KEY,
    captured_utc TEXT NOT NULL,
    source_root TEXT NOT NULL CHECK (source_root = '/scratch16/pclancy3/yi'),
    max_depth INTEGER NOT NULL CHECK (max_depth = 3),
    directory_rows INTEGER NOT NULL CHECK (directory_rows >= 1),
    source_tsv_sha256 TEXT NOT NULL CHECK (length(source_tsv_sha256) = 64),
    classification_tsv_sha256 TEXT NOT NULL CHECK (length(classification_tsv_sha256) = 64),
    scan_error_count INTEGER NOT NULL CHECK (scan_error_count >= 0),
    measurement_note TEXT NOT NULL
);

CREATE TABLE directory_entry (
    relative_path TEXT PRIMARY KEY,
    parent_path TEXT,
    depth INTEGER NOT NULL CHECK (depth BETWEEN 0 AND 3),
    direct_child_directories INTEGER NOT NULL CHECK (direct_child_directories >= 0),
    direct_regular_files INTEGER NOT NULL CHECK (direct_regular_files >= 0),
    direct_symlinks INTEGER NOT NULL CHECK (direct_symlinks >= 0),
    direct_other INTEGER NOT NULL CHECK (direct_other >= 0),
    direct_logical_bytes INTEGER NOT NULL CHECK (direct_logical_bytes >= 0),
    scan_error TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (parent_path) REFERENCES directory_entry(relative_path)
);

CREATE INDEX directory_entry_by_parent ON directory_entry(parent_path, relative_path);
CREATE INDEX directory_entry_by_depth ON directory_entry(depth, relative_path);

CREATE TABLE directory_project_classification (
    relative_path TEXT PRIMARY KEY,
    project_group TEXT NOT NULL,
    classification_basis TEXT NOT NULL,
    classification_note TEXT NOT NULL,
    FOREIGN KEY (relative_path) REFERENCES directory_entry(relative_path)
);

CREATE VIEW v_project_directory_summary AS
SELECT d.relative_path AS project_path,
       c.project_group,
       c.classification_basis,
       c.classification_note,
       d.direct_child_directories,
       d.direct_regular_files,
       d.direct_symlinks,
       d.direct_other,
       d.direct_logical_bytes,
       d.scan_error
FROM directory_entry AS d
JOIN directory_project_classification AS c
  ON c.relative_path = d.relative_path
WHERE d.depth = 1;

CREATE VIEW v_directory_tree AS
SELECT d.relative_path,
       d.parent_path,
       d.depth,
       c.project_group,
       c.classification_basis,
       c.classification_note,
       d.direct_child_directories,
       d.direct_regular_files,
       d.direct_symlinks,
       d.direct_other,
       d.direct_logical_bytes,
       d.scan_error
FROM directory_entry AS d
LEFT JOIN directory_project_classification AS c
  ON c.relative_path = CASE
       WHEN instr(d.relative_path, '/') = 0 THEN d.relative_path
       ELSE substr(d.relative_path, 1, instr(d.relative_path, '/') - 1)
     END;
