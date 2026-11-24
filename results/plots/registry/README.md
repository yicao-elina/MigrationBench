# Dated figure registry snapshots

The registry snapshot in this directory is a reproducible, reviewable
extension of the MigrationBench SQL-first asset registry.

## Files

- `migrationbench_asset_registry_20261101.sqlite`: materialized registry
  snapshot with 15 chart rows, including the user-certified R05 chart.
- `migrationbench_asset_registry_20261101.sqlite.bak`: byte-identical backup
  of the dated output snapshot.
- `migrationbench_asset_registry_20261101.sqlite.bak_pre_r05`: base snapshot
  copied before the R05 migration.
- `migrationbench_asset_registry_20261101_base_20261028.sqlite`: materialized
  local SQL-first base used for this update.
- `schema_base_20261025.sql`: base schema backup.
- `updates/20261101_cr_scratch600k_R05.sql`: idempotent SQL migration.
- `scripts/apply_figure_registry_update.py`: rebuild and integrity-check
  script.
- `registry_manifest_20261101.json`: base/output/migration hashes and row
  counts.

Run the update from the repository root with:

```bash
python results/plots/registry/scripts/apply_figure_registry_update.py --overwrite
```

The current `analysis/migrationbench_asset_registry_20261025/` SQLite path is
a File Provider `dataless` placeholder in this checkout. It was not
overwritten. The dated snapshot is therefore explicitly based on the
materialized 2026-10-28 local archive; refreshing the original registry later
should be a separate, source-verified migration rather than an implicit
replacement.

The registry stores chart-to-campaign, chart-to-panel, chart-to-data-file, and
chart-to-provenance links. It records where large remote inputs live without
pretending that pointer-only files are local reproducibility inputs.
