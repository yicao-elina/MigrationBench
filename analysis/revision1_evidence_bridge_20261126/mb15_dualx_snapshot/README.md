# MB15 model-grain status and DUAL-X coverage snapshot

This is a **model-level coverage snapshot**, with one row for each of the 15 submitted Revision 1 MD models. It is not an attempt table and it is not a complete 15-model-by-task performance matrix. The status labels and numeric values are imported from the cited R1 status and figure-data sources without reinterpretation.

## Contents and pins

- `MD15_DUALX_COVERAGE.tsv`: curated 24-column identity/reuse crosswalk, SHA-256 `293d9de146ef7cf80d6922c519a48fc3e7674b86c718a40f753db9ccdce9f6d9`.
- `schema.sql`: version 1 SQLite schema, preserving source terms and enforcing model/input SHA syntax.
- `import_mb15_dualx_snapshot.py`: pinned-input importer; it refuses a TSV whose pinned SHA changes and refuses to overwrite an existing DB without `--replace`.
- `queries.sql`: QMB01–QMB07, registered in the shared read-only federated launcher.
- `snapshot_manifest.json`: input identities, upstream DB pins, scope, output size/hash, and SQLite check results.

The coverage TSV was built from MigrationBench main `fba92f7b1650893ef9aeeef160f76a516009118f` and DUAL-X v14. The R02 status source, Fig. 6 data, protocol manifest, exact model identity bridge, bridge snapshot manifest, DUAL-X v14 SQLite, and MigrationBench R4 SQLite hashes are stored in `snapshot_metadata` and repeated in `snapshot_manifest.json`. The submitted status TSV and Fig. 6 data are also committed under `paper/revision1/data_processed/current_release/` on current main.

The archived SQLite file is pinned by its exact SHA-256. It was built with local SQLite 3.51.2. Rebuilding the same TSV under Rockfish SQLite 3.26.0 produced the same table names, all row values, metadata, row counts, and integrity results, but a different binary SHA-256. Treat the archive hash as the identity of the frozen database file; compare row contents when rebuilding across SQLite versions.

## Queries

```sh
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QMB01
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QMB02
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QMB03
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QMB04
```

- QMB01 returns all 15 rows and labels their grain explicitly.
- QMB02 reports exact checkpoint identity and keeps candidate, NEB, inference, and Grad-CAM counts as distinct evidence types. Those counts are not a validation score.
- QMB03 summarizes the recorded MD statuses by model family.
- QMB04 lists models for which this pinned bridge has no result-level row; that does not assert absence from every archive.
- QMB05 returns the source-file, source-database, commit, capture-date, and grain pins.
- QMB06 recomputes exact-hash joins against the attached MB R4 and DUAL-X v14 sources at query time and reports link, NEB, and static-candidate counts separately.
- QMB07 expands each exact-hash-matched model over its DUAL-X v14 design tasks and lists the run IDs/statuses present in each design cell. It does not create or imply MigrationBench task results.

The Rockfish smoke returned **270 DUAL-X design cells** for the 15 exact-hash-matched models (15 × 18 tasks): 24 cells had run rows, containing 48 run rows in total; 246 cells had none in the pinned v14 database. These are DUAL-X's planned statuses and runs, not completed MigrationBench results.

## Scientific and provenance boundaries

The 15/15 exact checkpoint SHA matches establish checkpoint identity only. They do not establish the producing run/attempt, trajectory continuity, processed value, figure/table cell, held-out evaluation, or scientific acceptance. Six FT grouped-A2 models have narrower path-specific NEB links; two Foundation checkpoints have prepared static-candidate rows, not held-out DUAL-X outcomes. Historical DUAL-X MD statuses are time-stamped observations, not accepted current trajectories. Run/attempt and claim linkage remains partial.

Preserve the original categories: 3 `TARGET_REACHED_PROVISIONAL`, 1 `TARGET_REACHED`, and 11 `SCIENTIFIC_FAILURE`. Do not recast threshold exits as successes or infer missing attempts from model IDs.
