# Revision 1 evidence bridge index — 2026-11-26

This directory links the validated Revision 1 evidence interfaces that can be queried from MigrationBench main. Large models, trajectories, and source SQLite snapshots remain in the user's personal Scratch16 archive; Git carries schemas, importers, queries, compact hash indexes, and immutable snapshot manifests.

## Pinned database sources

| Snapshot | Rockfish path | SHA-256 | Role |
|---|---|---|---|
| MigrationBench lineage R4 | `/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_lineage_overlays/20261125_r4/migrationbench_r1_lineage_overlay_20261125_r4.sqlite` | `d033f47913eaf33d61a2ce7b0b0281b90b385d23190890a97f477e2f901896d7` | R1 campaign, Table S5, run/artifact/result links |
| Revision 1 manifest sidecar R1 | `/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_manifest_sidecars/revision1_manifest_sidecar_r1_20261126/revision1_manifest_sidecar_r1.sqlite` | `cf526601b4b5327578993dfeeb0ebaecba3723b5811df1cec227b2279a385c03` | 1,866 manifest observations; 9,825 literal path-reference occurrences; 2,322 stat/hash checks |
| DUAL-X results v14 | `/scratch16/pclancy3/yi/X-FORCE_data_archive/sqlite_sources/revision1_dualx_20261125_v14/revision1_dualx_results.sqlite` | `a7e46f4bdf5ae134ab501b308dbfa987348d4a15201503297edd540c954e6fa4` | 414 design cells, 80 runs, exact model/source links |
| MB15 status/DUAL-X coverage snapshot v1 | `/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_matrix_coverage_snapshots/20261126/mb15_dualx_coverage.sqlite` | `cb314a7e30560654e1d0576f764b24d505a7f75c37b7cd5c82258dd026e3e80e` | 15 model-grain R1 status rows and typed exact-identity/reuse counts; not an attempt/task matrix |
| Claim-to-evidence inventory R1 | `/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_claim_matrix_snapshots/20261126/claim_matrix_r1.sqlite` | `cdc723b93143b6778fa1b3d6915a798e0cfd6fc6f31a6dce9e969a580e766578` | 639 manuscript assertion, figure/table, numeric-cell, and reviewer-question index rows; not scientific validation |

The full MigrationBench central provenance catalog remains an immutable 2026-11-24 snapshot. The manifest sidecar is staged separately; it has not been merged into a new central catalog snapshot. The R4 and DUAL-X databases are attached read-only for cross-project queries so their source hashes and schemas remain independently verifiable.

## Interfaces

- `revision1_lineage_overlay_r4/queries.sql`: QL21–QL26 query R4 run archives, corrected symlink/input identities, Table S5 outputs, and their source statuses.
- `manifest_sidecar_r1/`: central-compatible schema, Python 3.6+ importer, Q1–Q7, staging snapshot manifest/receipt, and exact SQL row-grain definitions. The actual SQLite payload and frozen scan/revalidation inputs are in the pinned Scratch16 archive.
- `dualx_exact_bridge_r1/`: QX01–QX05 read-only attached-database queries, per-model exact SHA index, and source snapshot manifest.
- `mb15_dualx_snapshot/`: QMB01–QMB07 and an immutable model-grain SQLite snapshot for the submitted 15-model MD status plus exact-hash DUAL-X coverage. It preserves candidate, NEB, and historical run counts as distinct evidence types; QMB06 recomputes cross-source exact-hash joins and QMB07 expands the matching DUAL-X model/task design cells and run records.
- `md15_attempt_lineage_r1/`: QMA01–QMA07 and a separate per-model audit of attempt/run evidence, scheduler-ID candidates, trajectory segments, and missing links. Candidate jobs are not treated as confirmed attempts.
- `mb15_fig5_matrix_r1/`: QF501–QF504 and a separate, pinned snapshot of the submitted Figure 5 source cells (15 models × 8 tasks = 120 rows); it is not the complete MigrationBench model-by-task matrix.
- `learning_loop_review_r1/`: QLR01–QLR03 and a separate seven-row audit-summary index of failure-mode signals and missing next-round decision/gain evidence. Rows are summaries, not event labels.
- `claim_matrix_r1/`: QCE01–QCE04 and an immutable 639-row Revision 1 claim/evidence inventory. Source-native `matrix_id` values remain distinct from catalog IDs; lexical segmentation and `TRACEABLE` status do not establish scientific acceptance.
- `neb_dft/`: 24 Table S5 cell/output recheck rows and the issue/gap register. Its query source is QL25–QL26 in R4.

## One read-only SQL entry point

`query_federated.py` opens the immutable central catalog as SQLite `main`, attaches R4, the manifest sidecar, DUAL-X v14, the MB15 model-grain coverage snapshot, the MD15 attempt-lineage audit, the Figure 5 source-cell snapshot, the learning-loop audit index, and the claim matrix snapshot with URI `mode=ro`, checks exact pinned file sizes and SQLite `user_version`s, and exposes every table/view through source-prefixed temporary views (`central__*`, `r4__*`, `sc__*`, `dx__*`, `mb15__*`, `md15a__*`, `fig5__*`, `learning_review__*`, `claim_matrix__*`). The central snapshot stays at the 2026-11-24 release. Source-native IDs are intentionally kept separate.

Run from the repository root or pass the script path directly on Rockfish (Python 3.6+):

```sh
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --list-queries
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QL25
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id Q1 --format csv
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QX01
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QMB01
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QMB07
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QCE02
```

Registered IDs are QL21–QL26 (R4), Q1–Q7 (manifest sidecar), QX01–QX05 (DUAL-X), QMB01–QMB07 (MB15 status/reuse coverage and DUAL-X task-cell expansion), QMA01–QMA07 (MD15 attempt-lineage evidence), QF501–QF504 (Revision 1 Figure 5 source cells), QLR01–QLR03 (failure-mode review summaries), and QCE01–QCE04 (claim inventory and provenance gaps). The launcher uses the frozen SQL blocks in their original files. A custom cross-database query must be one `SELECT` or `WITH ... SELECT` statement and should refer to source-prefixed views, for example:

```sql
SELECT dx__models.model_id, dx__models.checkpoint_sha256
FROM dx__models
JOIN r4__models
  ON lower(dx__models.checkpoint_sha256)=lower(r4__models.checkpoint_sha256);
```

The `mb15__mb15_model_coverage` view is a 15-row model-summary/crosswalk source. Its exact checkpoint SHA joins can be used to find candidate project identity links; it does not supply attempt-level producer or scientific-acceptance joins. The separate `md15a__model_attempt_audit` source adds per-model evidence observations but does not provide definitive attempt UUIDs or raw trajectory hashes. Candidate Slurm IDs remain log-derived candidates.

The `fig5__fig5_matrix_cell` table stores the 120 Figure 5 records from the committed 15×8 source slice, including all primary, diagnostic-only, and supporting rows. It is not a full MACE task matrix; 24 source rows lack model SHA and remain unjoinable by checkpoint identity in that table.

The `learning_review__failure_mode_review` source is an audit synthesis. It has seven evidence-family rows, no event-level failure labels or decision/action records, and no measured learning benefit. Use it to navigate gaps and closure work, not as training labels or an active-learning classifier.

The `claim_matrix__claim_evidence_row` view holds one row per pinned source inventory item, including its exact quote/value, artifact pointer, source status, and recorded gap reason. It is an index, not an independent scientific validation. QCE01–QCE04 retain assertion segments, panels, table objects/cells, and reviewer questions at their original grains.

```sh
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py \
  --sql-file /path/to/read_only_query.sql
```

Each invocation enables `PRAGMA query_only=ON` and a SQLite authorizer that permits only `SELECT`, `READ`, `FUNCTION`, and recursive-CTE operations. Writes, DDL, transaction control, further `ATTACH`/`DETACH`, and non-SELECT statements are denied. Source databases are opened read-only. `--verify-hashes` performs full SHA-256 reads of all pinned snapshots (about 4.96 GB); by default, the launcher checks file sizes and `user_version` and emits the pinned SHA values without claiming to have rehashed them during that invocation.

The launcher is a federated query interface, not a merged database. It does not infer run/result/claim equivalence from coincident integer IDs or checkpoint hashes. Use `FEDERATED_SOURCES.json` for exact path, byte-size, SHA, and `user_version` pins.

## Full model-matrix boundary

`MATRIX_SCOPE_AUDIT.md` documents the measured limits: R4 coverage is sparse metadata; its 229 lineage-result rows have mixed grains; Figure 5 is a submitted 15×8 slice; MD15 is a model-status lane; static prediction rows are candidate-grain; and DUAL-X is its own design/run dataset. These sources must not be flattened into one complete MigrationBench performance matrix.

## Scientific and identity boundaries

- Hash identity is necessary but does not prove that a plotted value came from a particular run, attempt, job, processing recipe, or manuscript release.
- All 1,866 manifest files currently match their prior scan size/SHA. For referenced targets, 440 match size+prior SHA, 13 large targets match size only, and 3 paths resolve to directories. The 3,367 job/retry-like manifest field values remain `UNVERIFIED`; all 2,319 staged artifacts remain `PROVENANCE_PARTIAL`.
- Six legacy Table S5 QE outputs now have current exact identities and 12 result/artifact links, while their original expected hashes, producer jobs/configs, and convergence evidence remain partial. C1–C5 still have SCF warnings and force residuals above the stated gate.
- The 23 DUAL-X model hashes appear in the frozen MB R1 registry, but only 7 match the current R4 model table. Reusable row-level evidence is limited to 1,278 prepared static candidates, 18 path-specific NEB records, and six historical MD status rows. The 414-cell DUAL-X design is not a complete measured MB performance matrix.
- No manuscript scientific statement, raw experiment file, database source snapshot, or model identity was rewritten to close these gaps.
