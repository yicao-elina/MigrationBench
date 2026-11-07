# MigrationBench Asset Registry — 2026-10-25

This directory is a **provenance-aware, additive inventory** of MigrationBench models, data assets, evaluations, failures, and open audit items. It was initialized from the three supplied Obsidian documents plus the linked current campaign locations. It is deliberately conservative: a campaign label is not promoted to a checkpoint identity unless its lineage is known, and scheduler completion is not promoted to scientific acceptance.

## Files

- `migrationbench_asset_registry.sqlite` — portable SQLite source of truth.
- `migrationbench_asset_registry.xlsx` — clean multi-tab spreadsheet for Google Sheets import (created by `build_workbook.mjs`).
- `registry_data.json` and `*.csv` — simple portable exports for review or import.
- `registry_app.py` — local management interface (safe read-only SQL plus targeted gap/status edits).
- `schema.sql` — normalized schema, compatible in spirit with SQLite now and MySQL/PostgreSQL after type/DDL adaptation.
- `data_quality_report.json` — structural validation from the build.
- `query_examples.sql` — read-only examples for model, task, coverage, chart, and Canva evidence queries.
- `migrations/001_canva_design_and_figure_relations.sql` — additive Canva design/page/source relation migration.

## Key design rule

The fact grain is **model identity × task/operator × material/system × campaign**. `coverage` is therefore the central matrix. It uses nullable `model_id` only for an explicitly aggregate campaign record whose source did not safely reveal the per-model mapping; those rows carry a gap that explains why.

## Model taxonomy

1. Foundation: MACE-MP, MACE-OMAT-0, MACE-MH, and MatPES PBE OMAT fine-tuned.
2. Train from Scratch: current Lmax1 Scratch-600K and Scratch-MultiT, each with seeds 123/234/345; historical/5%/Lmax2 variants remain separate.
3. Fine-tuning: Fine-tuning-600K and Fine-tuning-MultiT, each with seeds 123/234/345; grouped A2 and legacy variants remain lineage-pending.
4. Campaign-only labels are stored as `CAMPAIGN_LABEL`, never silently merged with canonical checkpoint IDs.

## Scientific boundaries encoded in the registry

- Fixed-path NEB and endpoint-only self-relaxed NEB are distinct task records and must not be pooled into a barrier ranking.
- Cr61 and Cr81 NEB are distinct systems.
- Interlayer DFT reference rows retain `COMPLETED_BUT_SCF_WARNING`; fixed-path results are diagnostics, while self-relaxed paths are endpoint-only MLIP proxies.
- Failed/unstable MD, optimizer nonconvergence, provenance gaps, and excluded legacy artifacts remain queryable rather than being deleted.

## Chart registry and data lineage

- `charts` gives every rendered or planned figure a stable `chart_id`; `chart_panels` separates composite panels by task/system; `chart_links` connects charts to model/task/system/dataset/campaign/asset entities; and `chart_data_files` records the CSV/TSV/JSON files used by the figure.
- `chart_sources` records figure exports, reproduction scripts, derived-data roots, manifests, hashes, repository/external scope, and whether each source was vendored at build time.
- `charts` gives every rendered or planned figure a stable `chart_id`; `chart_panels` separates composite panels by task/system; `chart_links` connects charts to model/task/system/dataset/campaign/asset entities; `chart_data_files` records the CSV/TSV/JSON files used by the figure; and `chart_provenance` stores manifest locators, output hashes, and input-file hashes where available.
- The local UI previews PNG/SVG/PDF assets when available. Opening a chart shows its linked panels, registry rows, and a bounded preview of each linked tabular data file. A missing or provisional figure is retained as a visible logical chart slot instead of being silently omitted.
- SQL results automatically surface charts when a query returns a registered `chart_id` or an entity identifier that is linked to a chart. This is an association aid; it does not promote warning-bearing or incomplete evidence to publication-ready status.

## Updating safely

1. Add immutable checkpoint details (`artifact_path`, SHA256, base checkpoint, training seed) before merging aliases.
2. Add one `coverage` row per model × task × system × campaign, preserving `run_status` and `scientific_status` separately.
3. Add a source and asset locator for every new data package.
4. Close a `gaps` row only with a linked audited source. Do not overwrite raw or historical rows.

## Local source of truth and Drive mirror

The preferred write location is the copy in
`/Users/alina/Project/migrationbench-paper-repro/analysis/migrationbench_asset_registry_20261025`.
This local copy is independent of the Google-Drive-mounted workspace. Rebuild
it with `python3 build_registry.py`, validate it with
`python3 sync_registry_mirror.py --check`, and stage a timestamped Drive
snapshot with `python3 sync_registry_mirror.py --stage`. See
`LOCAL_SOURCE_OF_TRUTH.md` for the one-way sync boundary and review policy.

## Local UI

```bash
python3 registry_app.py
```

Then open `http://127.0.0.1:8765` (or set `MIGRATIONBENCH_PORT` for a second
local instance). The interface has filtered views, a SQL SELECT console, and a
small gap/status editor. It is a local SQLite manager; it is not a replacement
for raw result provenance.

## Canva figure entity relations

The additive migration in `migrations/001_canva_design_and_figure_relations.sql`
records the current and historical Canva design IDs, stable page IDs
`CANVA-P01` through `CANVA-P10`, chart primary keys, source paths, reproduction
commands, data-file pointers, and provenance statuses. The current shortlink is
recorded as `REAUTH_REQUIRED`; the ten page mappings are explicitly historical
rows from the earlier catalog and are not silently asserted to match the new
design until Canva page-level access is restored.
