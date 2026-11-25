# Rockfish → Skipjack MigrationBench audit

This folder tracks the first cleanup and provenance pass. It is additive: the
legacy Rockfish trees remain the source of record until every retained input,
output, and scientific acceptance status is mapped.

The second-phase `/data` and `/scratch4` work is tracked in
[PHASE2_REPORT.md](PHASE2_REPORT.md). It includes completed guarded cleanup
batches, accessible-tree inventories for `/data/pclancy3` and
`/scratch4/pclancy3`, paper/figure provenance imports, and local Git
reconciliation. The two shared-filesystem mounts remain incomplete where
permissions or ownership prevent a safe traversal.

## Files

- [AUDIT_REPORT.md](AUDIT_REPORT.md) — what changed, what was verified, and what
  remains open.
- [DATA_LAYOUT.md](DATA_LAYOUT.md) — recommended SQL-first, artifact-linked
  layout for DFT, AIMD, NEB, MACE training, inference, and paper releases.
- [REGISTRY_RECONCILIATION.md](REGISTRY_RECONCILIATION.md) — snapshot counts,
  freshness, and provenance gaps across the three SQLite databases.
- [SOURCE_ROOTS.tsv](SOURCE_ROOTS.tsv) — observed Rockfish and Skipjack roots,
  quota context, and cleanup/migration status.
- [scoped_storage_inventory_20261122.tsv](scoped_storage_inventory_20261122.tsv) —
  counted Sb₂Te₃ subtrees and post-cleanup large-file scope.
- [BRANCH_CONSOLIDATION.tsv](BRANCH_CONSOLIDATION.tsv) — local merge record.
- [legacy_worktree_reconciliation_current_20261122.tsv](legacy_worktree_reconciliation_current_20261122.tsv) — refreshed path, size, and matching-path hash comparison after overlay and cleanup; the older `legacy_worktree_reconciliation.tsv` is the pre-overlay baseline.
- [legacy_639_overlay_manifest_20261122.tsv](legacy_639_overlay_manifest_20261122.tsv) — hashes for reconciled 639-model files and SVG outputs.
- [canva_metadata_overlay_manifest_20261122.tsv](canva_metadata_overlay_manifest_20261122.tsv) — hashes for the refreshed P03–P10 palette metadata.
- [legacy_worktree_campaign_inventory_20261122.tsv](legacy_worktree_campaign_inventory_20261122.tsv) — remaining local-only paths aggregated by campaign, with logical size, allocated bytes, dataless counts, and file types.
- [legacy_tracked_differences_20261122.tsv](legacy_tracked_differences_20261122.tsv) — hashes and review dispositions for same-path differences.
- [legacy_dataless_placeholders_20261122.tsv](legacy_dataless_placeholders_20261122.tsv) — macOS dataless flags, logical sizes, allocated blocks, and main-tree path matches.
- [cleanup/](cleanup/) — Rockfish wavefunction cleanup, held restart candidates,
  local Blender-backup, duplicate-package, and cache cleanups, retained evidence
  hashes, excluded LaTeX build products, and the SQLite cleanup ledger. The P16
  deletion receipt is [rockfish_p16_wfc_cleanup_20261122.tsv](cleanup/rockfish_p16_wfc_cleanup_20261122.tsv);
  its 111 retained-evidence hashes are in
  [retained_p16_protocol_sensitivity_evidence_sha256_20261122.tsv](cleanup/retained_p16_protocol_sensitivity_evidence_sha256_20261122.tsv).
  The Daily DFT cleanup receipt is
  [rockfish_daily_dft_wfc_cleanup_20261122.tsv](cleanup/rockfish_daily_dft_wfc_cleanup_20261122.tsv);
  its 181 retained-evidence hashes are in
  [retained_daily_dft_evidence_sha256_20261122.tsv](cleanup/retained_daily_dft_evidence_sha256_20261122.tsv).
  The database includes the local cleanup, dataless-placeholder, and four
  Rockfish WFC-cleanup batches; [cleanup_audit_queries.sql](cleanup/cleanup_audit_queries.sql)
  has ready-to-run summary queries.
- [provenance_catalog.sqlite](provenance_catalog.sqlite) and
  [phase2_schema/](phase2_schema/) — current cross-project path/provenance
  catalog, additive migrations 001–016, evidence importers, and cleanup guards.
- [PHASE2_REPORT.md](PHASE2_REPORT.md) — current cleanup totals, quota readings,
  inventory boundaries, paper-claim evidence status, and remaining owner/admin
  review.
- [migrationbench_registry_audit.ipynb](migrationbench_registry_audit.ipynb) —
  rerunnable read-only SQL and Python quality checks.
- [registry_audit_queries.sql](registry_audit_queries.sql) — compact standalone
  SQLite queries used by the notebook.

The paper package captured from the old workspace is archived at
[`archive/paper/revision1_camera_ready_as_found_20261121`](../../archive/paper/revision1_camera_ready_as_found_20261121).
Its file-level SHA-256 list is adjacent to that directory. The core asset
registry is an immutable copy at
[`registry/snapshots/migrationbench_asset_registry_core_20261101.sqlite`](../../registry/snapshots/migrationbench_asset_registry_core_20261101.sqlite),
with a machine-readable manifest alongside it.

## Current boundary

The guarded cleanup receipts record 228, 574, and 4,298 deleted QE restarts in
separate batches; the 574-file batch removed 1,536,705,954,176 bytes, and the
4,298-file batch removed 1,036,297,176,856 bytes by pre-delete logical size.
The 4,298 records comprise 3,817 WFC and 481 mixing restart files. A separate
coordinate-supported cleanup removed 7,581 restart files across 46 run roots,
including 175 entries / 155,250,907,476 bytes from seven 81-atom NEB roots
whose latest complete image paths were retained. These supersede the earlier
hold on NEB restart state. Fresh accessible-root checks find no nonempty exact
QE WFC/mixing restart payloads and no WFC/mix token basenames in
revision1_migrationbench_runs. Inputs, logs, XML, charge densities, path
coordinates, and deletion receipts remain. False-positive projwfc/PDOS and
software-name matches are retained. The Daily DFT tree has no per-run
manifest, so its scientific lineage remains partial.

The local provenance catalog records 14 cleanup batches (13 `COMPLETED`, one
historical Conda index-cache batch `PARTIAL`) and typed figure/claim records.
The accessible personal roots, including `/data/pclancy3/ycao73_home`,
`for-victor`, `for-yuqi`, `MHP_defect`, `/data/pclancy3/yi`, and
`/scratch4/pclancy3/yi`, received the same inventory, Git, owner, duplicate,
and reproducibility pipeline. Victor/Yuqi paths have no separate alumni
retention policy. The `/data` and `/scratch4` mount roots remain incomplete:
the latest root probes could not traverse 625/725 and 503/562 directories,
respectively; `/scratch4/pclancy3` also has 98 unreadable subdirectories.
Observed multi-terabyte archives and build trees belong to other owners and
were not modified. See `PHASE2_REPORT.md` for per-batch evidence, quota limits,
and the exact readable-scope boundary. At the time this inventory snapshot was captured, no GitHub push had been made.

The pre-overlay legacy-worktree inventory is a reconciliation aid, not a cleanup authorization. The refreshed snapshot records 14,708 accessible regular files, eight symlinks, and two tracked paths missing in the legacy checkout. The 12,386 local-only regular-file paths total 17.687 GB logical size and 12.627 GB allocated blocks. That includes 464 dataless placeholders (5.088 GB logical size, zero local blocks); 11,922 materialized local-only files account for the allocated bytes. All 12,386 local-only paths are untracked or ignored in the old Git checkout, so the branch merge did not migrate their contents. One more dataless iCloud item has a same-path file in main. Deleting the cloud placeholders would not free local data blocks and may delete uploaded iCloud data, so unique local files, cloud items, and symlinks still need provenance mapping before the old worktree can be retired.

The archive omits generated caches and LaTeX build products; their exact
paths, byte counts, and hashes are listed in
`cleanup/local_latex_build_products_excluded_20261121.tsv`.

## November 24 SQL catalog snapshot

`provenance_catalog.sqlite` is the working SQLite catalog for path inventory,
workflow provenance, paper claims, figures, and cleanup evidence. It is too large
for GitHub, so the durable snapshot is kept on Rockfish at
`/scratch16/pclancy3/yi/MigrationBench_data_archive/provenance_catalog_snapshots/20261124/`.
The adjacent `DB_SNAPSHOT_MANIFEST.json` records the exact file hash and size,
plus the SQLite integrity checks for that archived copy. New read-only queries
for the separate PR #10/#12/#13 review
candidates are in
[`phase2_schema/pr_candidate_release_catalog_queries_20261124.sql`](phase2_schema/pr_candidate_release_catalog_queries_20261124.sql).
That import preserves the submitted Revision 1 release and treats all three
PRs as separate review candidates.

The separate six-run Revision 1 grouped-A2 MD archive is indexed at
`/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_md_fig6_table_s8_20261124/`.
It contains 49 verified payloads (2,320,577,507 bytes), including all six
coordinate trajectory segments, final checkpoints, model weights, one shared
initial structure, and per-run provenance/results. It is an explicit subset of
the 15-model Fig. 6 / Table S8 / Table R7 cohort. The read-only retrieval query
is in `phase2_schema/revision1_catalog_queries.sql`; the importer is
`phase2_schema/import_revision1_md_fig6_archive_20261124.py`.

The final snapshot upload also exposed a stale 1,966,440,448-byte `provenance_catalog_20261124.sqlite.transfer2.partial` from an abandoned transfer. Its SHA-256 was `9b08146cac92151082dca72162bcc28ada19aba6e5b8c39d0a9161d8008f6d5f`; it had an epoch-era timestamp and no active snapshot/rsync writer remained. After verifying the complete current database (3,970,719,744 bytes; SHA-256 `2a46a3380ea291db3fc02dc0f148931eba40297ada670ab773ddcab8fe75d053`; SQLite `quick_check` `ok`; zero foreign-key violations), the partial was deleted and verified absent. Its receipt is `phase2_data/provenance_catalog_snapshot_20261124/ROCKFISH_PARTIAL_PURGE_RECEIPT.json`.
