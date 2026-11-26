# MigrationBench Revision 1 lineage overlay R4

R4 is an additive correction over the immutable R3 snapshot. It keeps the three observed MACE from-scratch terminal-instability runs as failure evidence and registers the hash-verified 37-payload archive under the user's personal Scratch16 archive.

## Identity correction

R3 recorded the original `common_initial.xyz` symlink's 145-byte `lstat` length alongside the SHA-256 of its 320,651,971-byte dereferenced target. R4 separates these into two artifacts: link metadata at the original Scratch16 path, and target content at the exact `/data/pclancy3/yi/...xyz` path with SHA-256 `bb42c493fb0c997af3957a89dfde03e3e5a67ebcfd37f5eced8f722dad546161`. The organized archive contains a separately verified copy of that target content.

## Scientific boundary

The three runs terminated before the requested 1,000,000 MD steps. Their result disposition remains `FAILURE_MODE / EXCLUDE_FROM_PUBLICATION`; R4 changes artifact identity and archive discoverability only. It does not promote these runs to stable transport results or alter any manuscript claim.

## Archive and SQL

The 37 copied payloads (1,819,048,604 bytes) are under:

`/scratch16/pclancy3/yi/MigrationBench_data_archive/campaigns/cr_sb2te3/mace_from_scratch_ablation_md_20261005/`

`ARCHIVE_COPY_MANIFEST.json`, `SOURCE_INPUT_SYMLINK.json`, `ARCHIVE_COPY_RECEIPT.json`, and `SHA256SUMS` capture the copy and identity evidence. The R4 database, importer, manifest, queries, and checksums are stored in the matching `revision1_lineage_overlays/20261125_r4/` snapshot directory. Query examples QL21–QL24 in `queries.sql` expose per-run archive paths, symlink/target identities, matching source/archive copies, and failure dispositions. QL25–QL26 in the same query file expose the Table S5 raw QE output and status rows.

## Rebuild

Use the exact R3 snapshot SHA-256 pinned by the importer and the frozen archive controls:

```sh
python3 import_mace_archive_lineage_r4.py \
  --r3-db /path/to/migrationbench_r1_lineage_overlay_20261125_r3.sqlite \
  --archive-copy-manifest /path/to/ARCHIVE_COPY_MANIFEST.json \
  --symlink-observation /path/to/SOURCE_INPUT_SYMLINK.json \
  --archive-receipt /path/to/ARCHIVE_COPY_RECEIPT.json \
  --archive-sha256sums /path/to/SHA256SUMS \
  --archive-root /scratch16/pclancy3/yi/MigrationBench_data_archive/campaigns/cr_sb2se3/mace_from_scratch_ablation_md_20261005 \
  --output-db /path/to/migrationbench_r1_lineage_overlay_20261125_r4.sqlite
```

The importer refuses to overwrite output, verifies the R3 base and frozen control hashes, checks each payload against the live archive when `--archive-root` is provided, and records its Python/SQLite build runtime, integrity, and row counts in the manifest. The Rockfish snapshot was built with Python 3.6.8 and SQLite 3.26.0; its byte-level database SHA is pinned for that snapshot. Rebuilds on a different SQLite runtime may serialize identical SQL content to different bytes, so cross-runtime verification should compare table contents, counts, queries, `quick_check`, and foreign keys. The recorded snapshot reports `quick_check=ok` and zero foreign-key violations.
