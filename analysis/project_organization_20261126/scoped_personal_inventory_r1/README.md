# Scoped personal-root inventory R1

This metadata-only snapshot covers exactly these three authorized Rockfish roots:

- `/data/pclancy3/yi`
- `/scratch4/pclancy3/yi`
- `/scratch16/pclancy3/yi`

The scanner uses `lstat` metadata, does not read file contents, and never follows symlinks. It records every file, directory, symlink and special entry, path-level logical/allocated bytes, inode/device/link metadata, and traversal errors. `allocated_unique_inode_bytes` de-duplicates hardlinks within each root; `allocated_path_sum_bytes` intentionally counts path allocations. Neither value is a quota measurement. The sequential walk is not an atomic filesystem snapshot. The archive output directory itself is excluded from the Scratch16 walk to avoid self-inclusion; the excluded path is recorded in the SQLite snapshot metadata.

The SQLite sidecar is kept out of the scientific claim database and the 11-source bridge. Its schema and read-only query templates are versioned here. QPR01–QPR07 provide root summaries, project-folder rollups, largest files, hardlink groups, traversal errors, folder depth summaries, and a global unique-inode total. The inventory supports triage; it does not authorize file deletion or infer scientific irrelevance.

Rockfish capture command:

```sh
python3 scan_personal_roots.py \
  --output /scratch16/pclancy3/yi/MigrationBench_data_archive/storage_audits/scoped_personal_root_inventories/20261126_r1/scoped_personal_root_inventory_r1.sqlite
```

The snapshot is stored at that archive path with mode `0600`. Query it with Rockfish SQLite 3.26 or newer in read-only mode using the statements in `queries.sql`. The full per-path database, largest-file path rollup and broad filename-hit list remain on Rockfish; Git contains the schema, query code, integrity receipt and an aggregate project/folder map, not the personal path-level inventory.

The recursive scan completed with zero traversal errors. The scanner then raised a reporting error after completion because it checked the final output path before renaming the completed `.partial` database. The stored snapshot state and timestamps were independently verified, `quick_check` and foreign-key checks passed, and the database was promoted atomically to the final path. The original capture script and corrected reusable scanner are both preserved in this package; see `SNAPSHOT_MANIFEST_R1.json` and `ROCKFISH_ARCHIVE_RECEIPT_R1.json` for details.

`WFC_MIX_CANDIDATES_R1.json` records a broad case-insensitive filename query. Its 15 nonempty hits are LAMMPS example inputs/logs containing `.mix`, not QE WFC or mixing-restart payloads. They were retained. `EXACT_RESTART_RECHECK_R1.json` applies the exact `*.wfcN`, `wfc*.dat`, and `*.mixN` basename patterns against the full inventory and found zero candidates (zero nonempty bytes). Both results are read-only metadata queries.

The 2026-11-24 inventory separately recorded 1,592 exact `.mixN` names (9,236,905,920 logical bytes) in `/data/pclancy3/yi`; the 2026-11-26 R1 scan finds none. Their removal is supported by the 7,581-file 2026-11-23 guarded restart receipt (1,584 nonempty cohort files) and the 2026-11-24 receipt for eight zero-byte placeholders. Together with the earlier 384-file batch, these receipts reconcile the 1,976-entry difference from the older composite inventory. The receipts do not establish a quota change.
