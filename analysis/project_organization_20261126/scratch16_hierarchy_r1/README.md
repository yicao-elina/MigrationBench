# Scratch16 depth-3 hierarchy snapshot (R1)

This read-only snapshot indexes directory layout under `/scratch16/pclancy3/yi` only. It contains 5,288 directory rows: the root, 222 direct-child directories, 1,629 depth-2 directories and 3,436 depth-3 directories. The scan recorded zero permission or traversal errors within that depth. It does not inspect any other user's paths.

The snapshot combines the depth-limited `scandir` metadata with the R21 direct-child project-family labels. Those labels are navigation metadata, sometimes based on names or prior notes; they are not proof of scientific authority or a complete project-to-claim lineage.

## Files

- `scratch16_folder_hierarchy_r1.sqlite`: queryable snapshot.
- `scratch16_depth3_inventory_20261126.tsv`: exact directory metadata source used by the importer.
- `schema.sql`, `queries.sql`: versioned schema and five read-only query blocks.
- `import_hierarchy.py`: rebuilds the SQLite snapshot from the inventory TSV and the R21 directory-classification TSV.
- `query_hierarchy.py`: executes registered queries with SQLite read-only URI mode and `query_only=ON`.
- `snapshot_manifest.json`, `SHA256SUMS`: source/database identities and integrity checks.

## Queries

From the MigrationBench repository root:

```sh
python3 analysis/project_organization_20261126/scratch16_hierarchy_r1/query_hierarchy.py --query-id QH01
python3 analysis/project_organization_20261126/scratch16_hierarchy_r1/query_hierarchy.py --query-id QH02 --format csv
python3 analysis/project_organization_20261126/scratch16_hierarchy_r1/query_hierarchy.py --query-id QH03 --format csv
python3 analysis/project_organization_20261126/scratch16_hierarchy_r1/query_hierarchy.py --query-id QH04 --parameter MigrationBench_data_archive
python3 analysis/project_organization_20261126/scratch16_hierarchy_r1/query_hierarchy.py --query-id QH05
```

QH01 reports scope and row counts; QH02 groups first-level folders; QH03 emits the captured tree; QH04 filters to one first-level path and its descendants; QH05 counts folders by the recorded project-family labels. The SQL file may also be run directly against the SQLite database. QH04 takes one bound parameter (the exact relative name of a depth-1 directory).

## Measurement boundary

For each directory, child counts and `direct_logical_bytes` describe only its immediate entries. The byte field sums immediate regular-file and symlink `lstat` sizes; it is not recursive size, allocated space or quota usage. Symlinks were counted but not followed. No file payload hash or run/claim provenance is inferred. Direct-child folder-family classification is copied byte-for-byte from the R21 TSV; deeper children inherit the top-level label only in QH03 as a navigation aid.

An identical database, source TSV, manifest and checksum list is archived at:

`/scratch16/pclancy3/yi/MigrationBench_data_archive/project_hierarchy_snapshots/20261126/`
