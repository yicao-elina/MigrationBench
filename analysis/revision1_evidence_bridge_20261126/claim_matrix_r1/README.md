# Revision 1 claim-to-evidence SQL snapshot

This additive SQLite sidecar makes the committed 639-row `MIGRATIONBENCH_R1_CLAIM_EVIDENCE_MATRIX_20261125.tsv` queryable through the federated read-only launcher. It mirrors all 20 source columns and keeps source-native matrix IDs intact. It does not merge manuscript claim IDs with central-catalog IDs or validate the scientific truth of a statement.

The inventory was generated at source commit `412ba9c5276a07a13026184cf62e6ff36588e05f` from submitted Revision 1 sources pinned to baseline `334260cdda3869d55999081cd610b34bfba53257`. The TeX, reviewer ledger, and large registry input hashes still match the current MigrationBench main at snapshot preparation. Keep the historical generator commit as the origin pin; do not rewrite it to the later capture commit.

Rows include assertion segments, figure panels, table objects, numeric table cells, and reviewer questions. `TRACEABLE`, `PARTIAL`, and `MISSING` preserve the source inventory's lexical/artifact-lineage classifications. In particular, 13/639 `TRACEABLE` is not a scientific-validation rate; the main/SI prose segment subset has 0 `TRACEABLE`, 32 `PARTIAL`, 306 `MISSING`.

## Build an immutable snapshot

From the repository root, point `--registry-path` at the exact registry snapshot whose SHA appears in the source manifest. On Rockfish, use the archived snapshot path listed in `FEDERATED_SOURCES.json` for the `central` source.

```sh
python3 analysis/revision1_evidence_bridge_20261126/claim_matrix_r1/import_claim_matrix.py \
  --registry-path /scratch16/pclancy3/yi/MigrationBench_data_archive/provenance_catalog_snapshots/20261124/provenance_catalog_20261124.sqlite \
  --output /scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_claim_matrix_snapshots/20261126/claim_matrix_r1.sqlite
```

The importer validates the TSV, manuscript, reviewer-ledger, generator, schema, and registry SHA-256 pins; checks the registry read-only with `quick_check` and foreign-key validation; inserts the rows without changing source values; and checks SQLite integrity, row counts, status counts, and the output hash. It writes a snapshot manifest beside the database. The database is stored outside Git and attached as a separate read-only source.

## Queries

- `QCE01`: all indexed source rows.
- `QCE02`: row counts by type and lineage status.
- `QCE03`: partial/missing gaps by document and object.
- `QCE04`: snapshot pins and scope metadata.

Run from the repository root:

```sh
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QCE02
```

The table is a navigation and audit index. Review exact manuscript text, evidence artifacts, protocols, and acceptance gates before drawing scientific conclusions.
