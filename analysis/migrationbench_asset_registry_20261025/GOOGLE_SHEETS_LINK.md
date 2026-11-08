# MigrationBench Asset Registry — Google Sheets mirror

Native Google Sheet: [MigrationBench Asset Registry — 2026-10-25](https://docs.google.com/spreadsheets/d/11USdbpky-S3LtiPInVlndrHwWNO9XDfd57_snm1TPr0/edit)

- Spreadsheet ID: `11USdbpky-S3LtiPInVlndrHwWNO9XDfd57_snm1TPr0`
- Created: 2026-10-25
- Tabs: Overview, Models, Task Matrix, Data Assets, Tasks, Systems, Campaigns, Assets, Gaps, Sources, Audit, Read Me, Model Name Map, Task Groups, Coverage Summary, Chart Index, Naming Aliases
- Local source of truth: `/Users/alina/Project/migrationbench-paper-repro/analysis/migrationbench_asset_registry_20261025/migrationbench_asset_registry.sqlite`
- Drive file mirror: `/Users/alina/Library/CloudStorage/GoogleDrive-yicao.alina@gmail.com/My Drive/Obsidian-google/Projects/MigrationBench/RegistrySync/current`

The Sheet is a collaborative review mirror of the local SQLite registry. It is deliberately not a two-way sync: update and validate the local SQLite registry first, stage a timestamped Drive snapshot with `sync_registry_mirror.py --stage`, then update the Sheet from that reviewed snapshot so that provenance and audit history stay coherent.
