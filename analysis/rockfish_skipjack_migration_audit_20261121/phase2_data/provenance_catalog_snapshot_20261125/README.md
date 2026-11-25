# Provenance catalog snapshot — 2026-11-25

The new Rockfish SQLite snapshot is
`/scratch16/pclancy3/yi/MigrationBench_data_archive/provenance_catalog_snapshots/20261125/provenance_catalog_20261125.sqlite`.
It was created from the immutable 2026-11-24 snapshot and records full traversal
totals for exactly `/data/pclancy3/yi` and `/scratch4/pclancy3/yi`.

The snapshot is 4,032,221,184 bytes, mode `0600`, SHA-256
`00750bbc76123777294e59338c0ac823309ec56bff483a482d3a87958274ba7f`.
The importer reported `PRAGMA quick_check=ok` and zero foreign-key violations.
The new `inventory_snapshot` rows hold complete traversal counts. Row-level
observations cover all in-tree hardlink members and the 200 largest files in
`/data/pclancy3/yi` (317,898 rows after overlap removal) and the 100 largest
files in `/scratch4/pclancy3/yi`. The full metadata inventories remain in the
Rockfish storage-audit archive and their checksum sidecars; the SQL database
does not claim a row for every path.

Use `phase2_schema/scoped_inventory_queries_20261125.sql` for snapshot totals,
large-file links to existing run/claim records, and hardlink path groups. The
database observations contain no newly computed file-content hashes.

The importer is `phase2_schema/register_scoped_inventory_20261125.py`. It
creates a new snapshot path and refuses to overwrite either an existing final
file or a partial output. The 2026-11-24 snapshot and its original manifest and
purge receipt remain preserved in `provenance_catalog_snapshot_20261124/`.
