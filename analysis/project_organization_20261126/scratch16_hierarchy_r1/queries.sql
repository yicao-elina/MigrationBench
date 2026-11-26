-- QH01: scope/depth and root direct-child metadata.
SELECT snapshot_id, captured_utc, source_root, max_depth, directory_rows,
       source_tsv_sha256, classification_tsv_sha256,
       scan_error_count, measurement_note
FROM inventory_snapshot;

-- QH02: direct-child project/campaign directories under the scoped root.
SELECT project_path, project_group, classification_basis, classification_note,
       direct_child_directories, direct_regular_files,
       direct_symlinks, direct_other, direct_logical_bytes, scan_error
FROM v_project_directory_summary
ORDER BY project_path;

-- QH03: complete captured directory tree through depth 3.
SELECT relative_path, parent_path, depth, project_group, classification_basis,
       classification_note, direct_child_directories,
       direct_regular_files, direct_symlinks, direct_other,
       direct_logical_bytes, scan_error
FROM v_directory_tree
ORDER BY relative_path;

-- QH04: nested folders for one top-level project. Bind ? to its relative path.
SELECT child.relative_path, child.depth, child.direct_child_directories,
       child.direct_regular_files, child.direct_symlinks, child.direct_other,
       child.direct_logical_bytes, child.scan_error
FROM directory_entry AS project
JOIN directory_entry AS child
  ON child.relative_path = project.relative_path
  OR child.relative_path LIKE project.relative_path || '/%'
WHERE project.depth = 1
  AND project.relative_path = ?
  AND (child.relative_path = project.relative_path
       OR substr(child.relative_path, 1, length(project.relative_path) + 1)
          = project.relative_path || '/')
ORDER BY child.relative_path;

-- QH05: size of each first-level project/campaign family by directory count.
SELECT project_group, count(*) AS directory_count,
       sum(direct_child_directories) AS immediate_subdirectories,
       sum(direct_regular_files) AS immediate_regular_files
FROM v_project_directory_summary
GROUP BY project_group
ORDER BY directory_count DESC, project_group;
