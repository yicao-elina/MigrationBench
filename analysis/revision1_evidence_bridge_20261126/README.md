# Revision 1 evidence bridge index — 2026-11-26

This directory links the validated Revision 1 evidence interfaces that can be queried from MigrationBench main. Large models, trajectories, and source SQLite snapshots remain in the user's personal Scratch16 archive; Git carries schemas, importers, queries, compact hash indexes, and immutable snapshot manifests.

## Pinned database sources

| Snapshot | Rockfish path | SHA-256 | Role |
|---|---|---|---|
| MigrationBench lineage R4 | `/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_lineage_overlays/20261125_r4/migrationbench_r1_lineage_overlay_20261125_r4.sqlite` | `d033f47913eaf33d61a2ce7b0b0281b90b385d23190890a97f477e2f901896d7` | R1 campaign, Table S5, run/artifact/result links |
| Revision 1 manifest sidecar R1 | `/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_manifest_sidecars/revision1_manifest_sidecar_r1_20261126/revision1_manifest_sidecar_r1.sqlite` | `cf526601b4b5327578993dfeeb0ebaecba3723b5811df1cec227b2279a385c03` | 1,866 manifest observations; 9,825 literal path-reference occurrences; 2,322 stat/hash checks |
| DUAL-X results v14 | `/scratch16/pclancy3/yi/X-FORCE_data_archive/sqlite_sources/revision1_dualx_20261125_v14/revision1_dualx_results.sqlite` | `a7e46f4bdf5ae134ab501b308dbfa987348d4a15201503297edd540c954e6fa4` | 414 design cells, 80 runs, exact model/source links |

The full MigrationBench central provenance catalog remains an immutable 2026-11-24 snapshot. The manifest sidecar is staged separately; it has not been merged into a new central catalog snapshot. The R4 and DUAL-X databases are attached read-only for cross-project queries so their source hashes and schemas remain independently verifiable.

## Interfaces

- `revision1_lineage_overlay_r4/queries.sql`: QL21–QL26 query R4 run archives, corrected symlink/input identities, Table S5 outputs, and their source statuses.
- `manifest_sidecar_r1/`: central-compatible schema, Python 3.6+ importer, Q1–Q7, staging snapshot manifest/receipt, and exact SQL row-grain definitions. The actual SQLite payload and frozen scan/revalidation inputs are in the pinned Scratch16 archive.
- `dualx_exact_bridge_r1/`: QX01–QX05 read-only attached-database queries, per-model exact SHA index, and source snapshot manifest.
- `neb_dft/`: 24 Table S5 cell/output recheck rows and the issue/gap register. Its query source is QL25–QL26 in R4.

## Scientific and identity boundaries

- Hash identity is necessary but does not prove that a plotted value came from a particular run, attempt, job, processing recipe, or manuscript release.
- All 1,866 manifest files currently match their prior scan size/SHA. For referenced targets, 440 match size+prior SHA, 13 large targets match size only, and 3 paths resolve to directories. The 3,367 job/retry-like manifest field values remain `UNVERIFIED`; all 2,319 staged artifacts remain `PROVENANCE_PARTIAL`.
- Six legacy Table S5 QE outputs now have current exact identities and 12 result/artifact links, while their original expected hashes, producer jobs/configs, and convergence evidence remain partial. C1–C5 still have SCF warnings and force residuals above the stated gate.
- The 23 DUAL-X model hashes appear in the frozen MB R1 registry, but only 7 match the current R4 model table. Reusable row-level evidence is limited to 1,278 prepared static candidates, 18 path-specific NEB records, and six historical MD status rows. The 414-cell DUAL-X design is not a complete measured MB performance matrix.
- No manuscript scientific statement, raw experiment file, database source snapshot, or model identity was rewritten to close these gaps.
