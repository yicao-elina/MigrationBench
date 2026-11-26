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
| Cross-element checkpoint inventory R1 | `/scratch16/pclancy3/yi/MigrationBench_data_archive/training_checkpoint_inventory_snapshots/20261125/cross_element_checkpoint_inventory_r1.sqlite` | `a88efbf43ae12b58a690d48978ffa2c8780dd33e92b5580ede90ee2270834fe3` | 3,027 file metadata rows across three training seeds; 2,997 intermediate checkpoint payloads remain unhashed and held |
| Lmax1 G3 failure/retry evidence R1 | `/scratch16/pclancy3/yi/MigrationBench_data_archive/lmax1_g3_failure_retry_snapshots/20261126/lmax1_failure_retry_r1.sqlite` | `0e6de64489a8bd00fcd42a739042764dac4f5e6c3aa3eb09a0fbfe8cba98ded1` | 194 diagnostic collector rows; 50 exact source-record edges; 10 missing retry variants and 90 pending Tier-2 gates |

The full MigrationBench central provenance catalog remains an immutable 2026-11-24 snapshot. The manifest sidecar is staged separately; it has not been merged into a new central catalog snapshot. The R4 and DUAL-X databases are attached read-only for cross-project queries so their source hashes and schemas remain independently verifiable.

## Interfaces

- `revision1_lineage_overlay_r4/queries.sql`: QL21–QL26 query R4 run archives, corrected symlink/input identities, Table S5 outputs, and their source statuses.
- `manifest_sidecar_r1/`: central-compatible schema, Python 3.6+ importer, Q1–Q7, staging snapshot manifest/receipt, and exact SQL row-grain definitions. The actual SQLite payload and frozen scan/revalidation inputs are in the pinned Scratch16 archive.
- `dualx_exact_bridge_r1/`: QX01–QX05 read-only attached-database queries, per-model exact SHA index, and source snapshot manifest. The cross-source QX query reads the MigrationBench lineage tables from the existing `r4` attachment; it does not create a duplicate alias.
- `mb15_dualx_snapshot/`: QMB01–QMB07 and an immutable model-grain SQLite snapshot for the submitted 15-model MD status plus exact-hash DUAL-X coverage. It preserves candidate, NEB, and historical run counts as distinct evidence types; QMB06 recomputes cross-source exact-hash joins and QMB07 expands the matching DUAL-X model/task design cells and run records.
- `md15_attempt_lineage_r1/`: QMA01–QMA07 and a separate per-model audit of attempt/run evidence, scheduler-ID candidates, trajectory segments, and missing links. Candidate jobs are not treated as confirmed attempts.
- `mb15_fig5_matrix_r1/`: QF501–QF504 and a separate, pinned snapshot of the submitted Figure 5 source cells (15 models × 8 tasks = 120 rows); it is not the complete MigrationBench model-by-task matrix.
- `learning_loop_review_r1/`: QLR01–QLR03 and a separate seven-row audit-summary index of failure-mode signals and missing next-round decision/gain evidence. Rows are summaries, not event labels.
- `claim_matrix_r1/`: the immutable 639-row Revision 1 claim/evidence source and its R1 build instructions. The original R1 snapshot remains archived separately and addressable.
- `claim_matrix_r2/`: R2 preserves the R1 claim table and metadata unchanged while adding a 30-row summary-grain Fig. 4 → R05 crosswalk. QCE01–QCE04 query the unchanged tables; QCE05 returns 29 source-row matches and one explicit missing/excluded slot. No additional SQLite source is attached.
- `claim_matrix_r3/`: R3 preserves the R2 rows and adds exact submitted/PR #10 manuscript source identities, ledger deltas, and all 47 changed PR #10 files reconciled against current main. PR #10 is closed without a merge commit but its head is an ancestor of current main: 44 files match in place and three manuscript versions remain at candidate paths. The live Overleaf tip is not verified. QCE06–QCE09 query these records without adding an attached SQLite source.
- `cross_element_checkpoint_r1/`: QCK01–QCK06 and a separate no-payload index for three cross-element training runs. It preserves all 3,027 file paths and the captured scheduler state; 2,997 intermediate checkpoint content hashes were not computed and remain on hold.
- `lmax1_failure_retry_r1/`: QLM01–QLM07 and a source-pinned event-level sidecar for Tier-1, retry, perturbation, and Tier-2 diagnostic collector rows. It preserves the ten missing retry variants and all 90 `PENDING_G3_11` gates and does not claim active-learning benefit.
- `neb_dft/`: 24 Table S5 cell/output recheck rows and the issue/gap register. Its query source is QL25–QL26 in R4.

## One read-only SQL entry point

`query_federated.py` opens the immutable central catalog as SQLite `main`, attaches R4, the manifest sidecar, DUAL-X v14, the MB15 model-grain coverage snapshot, the MD15 attempt-lineage audit, the Figure 5 source-cell snapshot, the learning-loop audit index, the claim matrix snapshot, the cross-element inventory, and the Lmax1 event sidecar with URI `mode=ro`, checks exact pinned file sizes and SQLite `user_version`s, and exposes every table/view through source-prefixed temporary views (`central__*`, `r4__*`, `sc__*`, `dx__*`, `mb15__*`, `md15a__*`, `fig5__*`, `learning_review__*`, `claim_matrix__*`, `cross_element__*`, `lmax1__*`). The central snapshot stays at the 2026-11-24 release. Source-native IDs are intentionally kept separate.

Run from the repository root or pass the script path directly on Rockfish (Python 3.6+):

```sh
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --list-queries
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QL25
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id Q1 --format csv
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QX01
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QMB01
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QMB07
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QCE02
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QCE05
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QCE06
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QCE07
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QCE09
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QCK02
python3 analysis/revision1_evidence_bridge_20261126/query_federated.py --query-id QLM01
```

Registered IDs are QL21–QL26 (R4), Q1–Q7 (manifest sidecar), QX01–QX05 (DUAL-X), QMB01–QMB07 (MB15 status/reuse coverage and DUAL-X task-cell expansion), QMA01–QMA07 (MD15 attempt-lineage evidence), QF501–QF504 (Revision 1 Figure 5 source cells), QLR01–QLR03 (failure-mode review summaries), QCE01–QCE04 (submitted-release claim inventory and provenance gaps), QCE05 (Fig. 4 summary → R05 group-row mapping), QCE06–QCE08 (manuscript source versions and candidate-ledger deltas), QCE09 (full PR #10 changed-file reconciliation), QCK01–QCK06 (cross-element training artifact inventory and exact-hash bridge), and QLM01–QLM07 (Lmax1 diagnostic events, explicit retry/perturbation links, and gate states). QCE05 maps summary rows only; it does not imply a curve-to-run/attempt relation, validate output, or accept a scientific claim. QCE06–QCE09 index source identity, ledger-text deltas and PR file integration; they do not validate or promote candidate scientific claims. The launcher uses the frozen SQL blocks in their versioned files. A custom cross-database query must be one `SELECT` or `WITH ... SELECT` statement and should refer to source-prefixed views, for example:

```sql
SELECT dx__models.model_id, dx__models.checkpoint_sha256
FROM dx__models
JOIN r4__models
  ON lower(dx__models.checkpoint_sha256)=lower(r4__models.checkpoint_sha256);
```

The `mb15__mb15_model_coverage` view is a 15-row model-summary/crosswalk source. Its exact checkpoint SHA joins can be used to find candidate project identity links; it does not supply attempt-level producer or scientific-acceptance joins. The separate `md15a__model_attempt_audit` source adds per-model evidence observations but does not provide definitive attempt UUIDs or raw trajectory hashes. Candidate Slurm IDs remain log-derived candidates.

The `fig5__fig5_matrix_cell` table stores the 120 Figure 5 records from the committed 15×8 source slice, including all primary, diagnostic-only, and supporting rows. It is not a full MACE task matrix; 24 source rows lack model SHA and remain unjoinable by checkpoint identity in that table.

The `learning_review__failure_mode_review` source is an audit synthesis. It has seven evidence-family rows, no event-level failure labels or decision/action records, and no measured learning benefit. Use it to navigate gaps and closure work, not as training labels or an active-learning classifier.

The `claim_matrix__claim_evidence_row` view holds one row per pinned submitted-release source inventory item, including its exact quote/value, artifact pointer, source status, and recorded gap reason. It is an index, not an independent scientific validation. R3 retains 639 claim rows, 20 R1 metadata rows and 30 Fig. 4/R05 crosswalk rows, then adds exact source identities, 18 ledger deltas and the 47-file PR #10 reconciliation. See [claim-matrix R3](claim_matrix_r3/README.md) and its [source-authority audit](claim_matrix_r3/MANUSCRIPT_SOURCE_AUTHORITY_AUDIT_R3.md).

The `cross_element__artifact_inventory` view stores exact paths and stat/hash metadata for 3,027 output files across seeds 123/456/789. It does not contain checkpoint bytes; hashes for epochs 0–998 were not computed. `cross_element__training_run_snapshot` records the dated 2026-11-25 Slurm result capture, not live scheduler state. QCK04 joins only exact recorded SHA values to model checkpoint hashes and proves identity at that row grain, not shared task results.

Two independent SQLite snapshots are listed in [`AUXILIARY_SQL_SNAPSHOTS.json`](AUXILIARY_SQL_SNAPSHOTS.json): the traversal-complete three-root metadata inventory (excluding its own Rockfish archive subtree) and the normalized PR #10 screen R4 (source commit `19708d74`, 639 candidates, 1,917 static evaluations, 4 figure panels). They are intentionally queried through their own read-only SQL packages because the 11-source federation already uses SQLite's attachment limit. The R4 screen database links source row → candidate/frame/run → source MD checkpoint and separately evaluated checkpoint → figure panel and claim evidence; it does not claim DFT completion or DUAL-X gain.

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
