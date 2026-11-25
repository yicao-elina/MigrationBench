# Rockfish catalog snapshot evidence — 2026-11-24

This folder preserves the byte-for-byte Rockfish manifest and purge receipt
for the verified snapshot at
`/scratch16/pclancy3/yi/MigrationBench_data_archive/provenance_catalog_snapshots/20261124/provenance_catalog_20261124.sqlite`.

The manifest reports source/destination size `3,970,719,744` bytes, SHA-256
`2a46a3380ea291db3fc02dc0f148931eba40297ada670ab773ddcab8fe75d053`,
`PRAGMA quick_check=ok`, and zero foreign-key violations at
`2026-11-24T10:07:30Z`. Personal quota-after was unavailable.

The purge receipt reports that `provenance_catalog_20261124.sqlite.transfer2.partial`
was removed at that time after an active-writer check, with exact content hash
`9b08146cac92151082dca72162bcc28ada19aba6e5b8c39d0a9161d8008f6d5f` and
logical size `1,966,440,448` bytes. The receipt does not provide allocated bytes
or an all-user Slurm queue snapshot; no quota change is claimed.

A separate `provenance_catalog_20261124.sqlite.partial` path from the earlier
capture changed metadata/content size and was later observed absent. It is not
listed in the purge receipt, so this audit does not attribute its disappearance
to that cleanup or count it as freed space. The normalized path-state record and
SQL importer are in `phase2_data/catalog_snapshot_recovery_20261124/` and
`phase2_schema/import_catalog_snapshot_partial_cleanup_20261124.py`.

The Rockfish manifest and receipt are immutable historical evidence. Later
snapshots use separate timestamped manifests rather than overwriting these
records; each manifest is stored outside the SQLite database it describes.
