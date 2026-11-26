-- QPR01: snapshot coverage and physical/logical storage tallies by exact root.
SELECT s.snapshot_id, s.started_utc, s.finished_utc, s.completion_state,
       s.scope_note, r.root_id, r.root_path, r.entry_count, r.file_count,
       r.directory_count, r.symlink_count, r.special_count, r.error_count,
       r.logical_bytes, r.allocated_path_sum_bytes,
       r.allocated_unique_inode_bytes
FROM inventory_snapshot AS s
JOIN inventory_root AS r USING(snapshot_id)
ORDER BY r.root_path;

-- QPR02: all direct-child projects/folders, with descendant entry and byte sums.
SELECT r.root_id, e.top_level_name,
       SUM(CASE WHEN e.depth=1 THEN 1 ELSE 0 END) AS direct_entries,
       SUM(CASE WHEN e.entry_type='FILE' THEN 1 ELSE 0 END) AS descendant_files,
       SUM(CASE WHEN e.entry_type='DIR' THEN 1 ELSE 0 END) AS descendant_directories,
       SUM(e.logical_bytes) AS descendant_logical_bytes,
       SUM(CASE WHEN e.entry_type='FILE' THEN COALESCE(e.allocated_bytes,0) ELSE 0 END)
           AS descendant_allocated_path_sum_bytes
FROM inventory_root AS r
JOIN inventory_entry AS e USING(root_id)
WHERE e.relative_path <> ''
GROUP BY r.root_id, e.top_level_name
ORDER BY r.root_id, descendant_logical_bytes DESC, e.top_level_name;

-- QPR03: largest regular files per root. Metadata only; no file content read.
SELECT root_id, relative_path, logical_bytes, allocated_bytes,
       mode_octal, mtime_ns, device, inode, link_count
FROM (
    SELECT e.*, ROW_NUMBER() OVER (PARTITION BY root_id ORDER BY logical_bytes DESC,relative_path) AS rank
    FROM inventory_entry AS e WHERE entry_type='FILE'
)
WHERE rank <= 100
ORDER BY root_id, rank;

-- QPR04: in-root hard-link groups and path-level allocation duplication.
SELECT root_id, device, inode, COUNT(*) AS path_count,
       MIN(logical_bytes) AS min_logical_bytes,
       MAX(logical_bytes) AS max_logical_bytes,
       SUM(COALESCE(allocated_bytes,0)) AS allocated_path_sum_bytes,
       MAX(COALESCE(allocated_bytes,0)) AS allocated_unique_inode_bytes
FROM inventory_entry
WHERE entry_type='FILE' AND device IS NOT NULL AND inode IS NOT NULL
GROUP BY root_id, device, inode
HAVING COUNT(*) > 1
ORDER BY root_id, allocated_path_sum_bytes DESC;

-- QPR05: complete traversal errors. Zero rows means all enumerated directories opened.
SELECT root_id, path, error_class, error_message
FROM inventory_error
ORDER BY root_id, path;

-- QPR06: directory count and maximum observed depth by project/folder.
SELECT root_id, top_level_name, COUNT(*) AS descendant_directories,
       MAX(depth) AS maximum_depth
FROM inventory_entry
WHERE entry_type='DIR'
GROUP BY root_id, top_level_name
ORDER BY root_id, descendant_directories DESC, top_level_name;

-- QPR07: global unique-inode allocation over the three roots. This avoids
-- double-counting a hardlinked inode that happens to be reachable from two roots.
SELECT COUNT(*) AS unique_file_inodes,
       SUM(max_allocated_bytes) AS allocated_unique_inode_bytes,
       SUM(path_count-1) AS extra_hardlink_paths,
       COUNT(DISTINCT device) AS devices
FROM (
    SELECT device, inode, MAX(COALESCE(allocated_bytes,0)) AS max_allocated_bytes,
           COUNT(*) AS path_count
    FROM inventory_entry
    WHERE entry_type='FILE' AND device IS NOT NULL AND inode IS NOT NULL
    GROUP BY device,inode
);
