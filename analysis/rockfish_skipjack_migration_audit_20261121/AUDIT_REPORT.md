# Phase 1 — cleanup, paper alignment, and mainline consolidation

**Snapshot date:** 2026-11-21 America/Los_Angeles (2026-11-22 UTC)

This report records the completed phase-1 snapshot. Phase-2 Rockfish `/data` and
`/scratch4` inventory and cleanup progress is tracked separately in
[`PHASE2_REPORT.md`](PHASE2_REPORT.md); later observations there supersede the
phase-1 storage boundary below.

## Work completed

- Reconciled the 17 differing 639-model release files from the legacy checkout
  into unified `main`, including its grouped figure, release notes, plotting
  code, and paper table fixes. The candidate-manifest and PDF hashes now match
  the release manifest exactly; both PNG hashes also match their figure
  manifests. The two SVGs listed by the release manifests were also recovered
  from the legacy checkout and copied with exact hashes; the unrelated
  unreferenced `figures/test.svg` was not treated as a release output. The
  copied file hashes are in `legacy_639_overlay_manifest_20261122.tsv`. The
  release handoff manifest anchors its evidence to code commit `c0aee66`; the
  live 639 branch tip is `476ca16` (a metadata-only pointer update), and public
  GitHub `main` is `817acf2` as of this check. Overleaf remains unsynchronized.
- Refreshed the Canva P03–P10 bundle README and manifest in main after all 24
  output hashes and four renderer hashes matched in both worktrees. The updated
  metadata files are listed in `canva_metadata_overlay_manifest_20261122.tsv`.
- Rechecked the strict Lmax2 Rockfish array in a login shell. Tasks
  `32055099_0` and `_1` failed before training with exit `1:0`; both stderr
  logs show `RF_CONDA_SITE: unbound variable`. Scheduler output and exact
  stderr bytes are retained under
  `analysis/strict_lmax2_control_20261015/rockfish_followup_20261122/`. The
  earlier pending note remains as history and is superseded by the new recheck.
- The original worktree reconciliation TSV is a pre-overlay snapshot. The
  refreshed filesystem scan records 14,708 accessible regular files, eight
  symlinks, and two tracked paths missing in the legacy checkout; 442 cleanup
  deletions remain in the same inventory for accounting. After cleanup, 12,386
  local-only regular-file paths have 17.687 GB of logical size. The filesystem
  reports 12.627 GB of allocated blocks for those paths. Of the local-only
  paths, 464 are macOS dataless placeholders (5,088,490,398 logical bytes,
  zero allocated blocks); the remaining 11,922 materialized files account for
  12,626,722,816 allocated bytes. One additional 70,406-byte dataless file has
  a same-path counterpart in main. A direct File Provider query on that PDF
  reports an uploaded iCloud item that is not downloaded; its local read is
  empty. It is therefore excluded from content-difference counts. The
  placeholder manifest is `legacy_dataless_placeholders_20261122.tsv`.
  All 12,386 local-only regular-file paths are untracked or ignored by the old
  checkout's Git and absent from main; merging Git histories did not migrate
  these data artifacts.
  Another 2,167 paths match main (400.955 MB); 135 tracked text files differ
  only by line endings, 13 text files and six locally readable binary files
  have content differences still under review. The remaining binary ledger
  row is the dataless iCloud placeholder, not a verified local content
  difference. By logical size, the largest unique groups are endpoint-only NEB
  (4.849 GB), Sb₂Te₃
  scene work (4.265 GB), NEB Blender refinement (2.338 GB), and NEB
  visualization assets (1.047 GB). Endpoint-only NEB contains 438 dataless
  paths (4.526 GB logical) and occupies 328.4 MB of local blocks; the campaign
  inventory reports logical and allocated sizes separately. See
  `legacy_worktree_reconciliation_current_20261122.tsv`,
  `legacy_tracked_differences_20261122.tsv`, and
  `legacy_worktree_campaign_inventory_20261122.tsv`. Dataless items consume no
  local data blocks, but deleting them may remove their uploaded iCloud items;
  the attempted single-file materialization did not download the sample. Keep
  the physical legacy worktree until its unique materialized files, cloud
  items, differing figures, and symlinks are mapped or archived.
- Removed 86 superseded Blender `.blend1` backups (3,867,296,788 bytes),
  each with a corresponding `.blend` scene at least as new; the ledger records
  both hashes and timestamps. Removed one redundant `.tar.xz` package
  (51,965,036 bytes) after all 150 member paths, sizes, and contents matched
  the retained ZIP. Removed 355 regenerable `.pyc`, `.DS_Store`, and `.ruff_cache`
  files (6,536,830 bytes). The ZIP remains in the legacy tree as an alternate
  paper package: its 150 members are not identical to the newer as-found
  camera-ready snapshot. Cleanup manifests are in `cleanup/`.

- Removed four evidence-checked batches of regenerable QE wavefunction
  restarts from Rockfish `scratch16`: 24 files from 12 completed endpoint
  runs (466,472,847,360 bytes); 48 files from 12 completed `nm_dft_unique`
  endpoint relaxations (233,371,779,840 bytes); 118 files from 18
  completed protocol-sensitivity SCF runs (101,450,575,448 bytes); and 38
  files from 18 completed `Daily/2026-09-02/dft_qe` SCF runs (68,513,191,336
  bytes). The P16 jobs were fresh-start calculations; all jobs in both newest
  batches were `COMPLETED 0:0` and absent from the live queue. For the older
  Daily runs, logs recorded randomized atomic starting wavefunctions and SCF
  convergence, and the QE XML files parsed. Across Rockfish, the four batches
  removed **228 files / 869,808,393,984 logical bytes (869.808 GB)**. A further
  3,925,798,654 bytes of local backup/archive/cache files were removed, for
  **873,734,192,638 logical bytes (873.734 GB) accounted across both
  locations**. Inputs, logs, manifests where present, XML, charge densities,
  runtime provenance, and deletion-time SHA-256 records were retained. The
  mode bits of the 118 P16 files were not captured before deletion and are
  marked `NOT_CAPTURED` in SQLite rather than inferred.
- Cleaned the 38 Daily WFC files only after checking 18 Slurm jobs, inputs,
  convergence markers, parsed XML, retained charge density, and an empty live
  queue. The directory has no per-run manifest; a text/source search within
  `/scratch16/pclancy3/yi/1-Cr-Sb2Te3` found no downstream `rattle_screening`
  reference. The input, QE log, scheduler logs, submission script, XML,
  charge density, and pseudopotentials remain with 181 SHA-256 rows. Treat the
  calculation lineage as partial until it is linked to paper/SQL records.
- Created a local unified `main` from the public GitHub `main`, then merged the
  639-model evidence branch, the unrelated local legacy history, the figure
  archive, the from-scratch MD ablation evidence, and the Cr-NEB evidence
  workflow. The legacy history had no shared Git ancestor with the GitHub
  reproduction repository, so its merge explicitly preserves both histories.
- Added the latest available core asset-registry SQLite snapshot (audit log
  ends 2026-11-01) and captured the old-workspace camera-ready package as an
  as-found archive. The archive has 265 files / 106,396,415 bytes. Generated
  `__pycache__`, `.pyc`, `.DS_Store`, and 35 regenerable LaTeX build products
  (859,058 bytes) were excluded; those LaTeX paths and hashes are listed in
  `cleanup/local_latex_build_products_excluded_20261121.tsv`. The package's
  own delivery manifest is retained, and a separate SHA-256 file covers every
  archived file.
- Recorded exact source paths, hashes, table counts, branch merge commits, and
  cleanup actions in this folder.
- Added the 442 completed local deletions and all 465 dataless placeholders to
  `cleanup/cleanup_audit.sqlite` as queryable rows; the corresponding review
  queries are in `cleanup/cleanup_audit_queries.sql`.

## Storage evidence

The Rockfish login banner reported group `pclancy3` at 13.62/10.00 TB on
`data`, 9.09/10.00 TB on `scratch4`, and 17.60/20.00 TB on `scratch16`.
After the revision1 cleanup and before the separate Daily WFC deletion,
`du -sh` reported `/scratch16/pclancy3/yi` at about 4.6 TB and
`revision1_migrationbench_runs` at about 2.4 TB. The previously observed quota
banner (17.60/20.00 TB on `scratch16`) predates this pass; these are rounded
`du` observations, not a refreshed group-quota report. The cleanup files in
`cleanup/` give exact removed file sizes.

The 2026-11-22 scan of `/scratch16/pclancy3/yi/revision1_migrationbench_runs`
found 574 files over 500 MB totaling 1,536,705,954,176 bytes: 535 QE WFC files
and 39 mixing restarts. They were removed in the guarded
`RF-SCRATCH16-YI-REV1-574-RESTART-CLEANUP-20261122-V1` batch; all 574 SQL
cleanup rows are `DELETE/VERIFIED`.

The later user-directed cleanup removed another 4,298 exact WFC/mixing
restarts from `revision1_migrationbench_runs`: 3,817 WFC and 481 mixing files,
with 1,036,297,176,856 pre-delete logical bytes. Batch
`RF-SCRATCH16-YI-REV1-QE-RESTART-CLEANUP-20261123` records all 4,298 as
`DELETE/VERIFIED`. A fresh 2026-11-24 scan now finds zero basenames
containing `wfc` or `mix` in that tree.

The earlier hold on restart files for 81-atom NEB continuation has also been
superseded by the user's instruction that the latest saved intermediate-image
coordinates/trajectory are sufficient for migration. A coordinate audit found
175 restart entries / 155,250,907,476 bytes across seven NEB run roots, with
complete latest `.path` trajectories and matching XML positions; batch
`RF-YI-QE-RESTART-CLEANUP-20261123` records each as `DELETE/VERIFIED`.
Inputs, logs, XML and path coordinates remain. No nonempty exact QE WFC or
mixing restart is left in the accessible personal roots.

Scoped scans of the additional Sb₂Te₃ trees are summarized in
`scoped_storage_inventory_20261122.tsv`. After the 38 WFC removals,
`/scratch16/pclancy3/yi/1-Cr-Sb2Te3` has 1,204 files and 11,803,978,268 logical
bytes, with no files over 500 MB. `/scratch4/pclancy3/yi/2.Cr-Sb2Te3` has
2,844 files and 425,753,425 bytes, with no files over 500 MB. The scoped
`/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3` tree has 48,484 files and
218,771,743,780 bytes, including 25 files over 500 MB (54,724,545,380 bytes).
Those large files include band-projection XMLs, collaborator perovskite data,
a NEB trajectory, and a Git pack, so they remain pending ownership and
paper-role review. A complete traversal of `/data/pclancy3/yi` was not
completed at the phase-1 cutoff; no `/data` or `/scratch4` files had yet been
deleted at that cutoff. Phase 2 now records 14 cleanup batches: 13 completed
and one historical Conda index-cache batch marked `PARTIAL`. They include the
`/scratch4` WSe₂ WFC cleanup, the 12-file Cr-in-gap mixing-history cleanup,
and a later eight-file zero-byte placeholder follow-up. The broader group
inventories remain incomplete. The November 22 `/scratch16` 574-file
threshold count was later reconciled to its per-file cleanup receipt. The
subsequent 4,298-file revision1 batch and 7,581-file coordinate-supported
batch are also registered in the provenance catalog.

## Scientific and provenance boundaries

- The latest camera-ready evidence package found in the old workspace contains
  22 claim rows. It records validation gaps, provisional references, exclusions,
  and descriptive diagnostics. The canonical 81-atom, 10-image DFT reference
  retains SCF warnings; its reported image-set values are not validated DFT
  kinetic barriers.
- The core SQLite has 26 Rockfish job-status rows, including rows labelled
  `RUNNING`; its newest audit event is from November 1. Those states are
  historical records, not live queue truth. A current Rockfish `squeue` query
  showed only pending jobs, and no job from the cleanup set was active.
- The six `spx` runs are *not* converged endpoints: their compact lineages show
  zero ionic steps, `bfgs_converged=false`, and time-limit termination. Their
  large restart state is retained because it could support continuation. The
  missing manifest conclusion from the initial scan was wrong; each has an
  input manifest, but none establishes endpoint acceptance.
- The collaborator DFT-NEB database contains 13 calculations, 12 endpoints,
  10 trajectories, 10 results, and 10 quality-gate records, but no artifact,
  structure, system, or publication-link rows. It is not yet a complete
  artifact-to-claim lineage database.
- At the phase-1 cutoff, Rockfish `/data` and `/scratch4` content and Skipjack
  scratch content with active jobs had not been deleted or relocated. Phase 2
  now records 14 cleanup batches, including evidence-cleared WSe₂ WFC and QE
  mixing-history cleanup. The remaining mount
  inventory, deduplication, job-history joins, and paper-role mapping are
  tracked in `PHASE2_REPORT.md`.

## Remaining work

1. Finish the file-level inventory across Rockfish `/scratch16`, `/scratch4`,
   and `/data`, then reconcile every large restart/cache against manifests,
   Slurm history, paper claims, and later superseding runs. The current exact
   post-cleanup large-file scan covers only
   `/scratch16/pclancy3/yi/revision1_migrationbench_runs`.
2. Refresh live Slurm status in a login shell before classifying pending or
   running data. Do not infer scientific acceptance from scheduler state.
3. Reconcile the core asset registry, chart/gallery registry, and DFT-NEB
   registry as separate scopes; close the missing hash and entity-link gaps.
4. Map each retained source directory to stable calculation, model, dataset,
   paper-claim, and artifact IDs before moving any raw trees.
5. Transfer to Skipjack using Globus after account/permission and target-storage
   checks. The supplied Skipjack `scratch_pclancy3/yi` path is scratch space;
   long-lived release data should have an approved Project Storage target.

At the cutoff of the earlier phase-1 snapshot, no push to GitHub or deletion
of other worktrees/source directories had occurred. The Git histories are consolidated locally, but physical worktree
retirement remains gated on resolving 11,922 materialized local-only files
(12.627 GB allocated), 464 local-only cloud placeholders, and seven unique
symlinks. The dataless paths do not account for local disk savings; removing
them is a cloud-data action that needs item-level identity checks.

## 2026-11-24 manuscript and asset consolidation

The three user-supplied PDFs and clean LaTeX ZIP were compared with the submitted
Revision 1 Git release and the latest available Overleaf-export branch snapshot.
All seven main-figure PDFs in the clean ZIP match the submitted release blobs;
the exact TeX-to-PDF build chain remains unavailable. The detailed report is
phase2_data/MANUSCRIPT_REVISION1_ALIGNMENT_AUDIT_20261123.md.

The manuscript/evidence review records these unresolved publication limits:
the canonical 81-atom QE NEB paths have maximum image forces of 0.38–3.2 eV/Å
against a 0.03 eV/Å threshold, while manuscript text still uses those paths as
canonical DFT references; SI Fig. S2's plotted data do not match its caption;
MD termination labels conflict across the main text, SI, response, and
reproducibility ledger; and submitted Table R2/S11 and MPA0 checkpoint labels
have recorded discrepancies. Stored Fig. 6 embeddings and the Fig. 7 SHAP
rerun are not fully regenerable from saved environments/outputs. These remain
review findings, not corrected manuscript claims or scientific acceptance.

PR #10, #12, and #13 are separately registered as review candidates in the
SQLite provenance catalog. The submitted release row was left unchanged.
PR #13's 1,336 archived files match source hashes and sizes, while their absence
from the two pinned Git trees remains recorded as a separate status. The SQL
query cookbook and import receipt are in the audit bundle.

The verified Rockfish asset archive
/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_git_large_assets_20261124
contains five payloads totaling 167,458,328 bytes: Foundation-MACE, FT-600K,
FT-Multi-T, Scratch-Multi-T, and the CANVA-P03 SVG. SHA-256 and Git blob
identities were checked where applicable. Four tracked model files and the
54,311,325-byte SVG are removed from the consolidated candidate's latest Git
tree. The original
/data model sources remain in place. Model loaders now resolve the archive
through MIGRATIONBENCH_MODEL_DIR or MIGRATIONBENCH_MODEL_ARCHIVE.

The 4,298-file revision1 batch removed the 3,817 WFC and 481 mixing restart
files previously inventoried under revision1_migrationbench_runs; the SQL
ledger records 1,036,297,176,856 pre-delete logical bytes and 4,298
DELETE/VERIFIED rows. Eight empty .mixN placeholders in the Cr-in-gap area
were also removed after a fresh all-user queue check. No nonempty exact QE
restart payload remains in the accessible personal roots. Broad projwfc/PDOS
and software-name matches were retained. The complete /data and /scratch4
mount inventories remain bounded by access limitations.

## Update — 2026-11-23 coordinate-supported WFC cleanup

At the user's direction, a second guarded cleanup removed 7,581 WFC and mixing
restart files from 46 Rockfish run roots beneath `/scratch16/pclancy3/yi` and
`/data/pclancy3/yi`. It freed 1,225,323,826,928 logical bytes by pre-delete
file size (1,225,535,520,768 allocated bytes by stat). The manifest and
per-file receipt agree on every path and stat identity, all 46 run roots had
fresh queue checks immediately before mutation, and each deleted path was
verified absent. Restart payload contents were not hashed. XML and path
trajectory evidence remains preserved and linked in SQLite.

A direct post-delete traversal of both user subtrees found no `.wfcN` or
`wfc*.dat` names. It found eight zero-byte `.mixN` placeholders, retained as
`HOLD` and matching the SQL ledger by exact path and inode. The 48 mixing
restarts whose scheduler WorkDir did not exactly resolve were deleted only
after matching a later parseable same-prefix XML; their lineage remains marked
uncertain. This is a cleanup decision, not scientific validation.

Personal `quota -s` readings were denied by Rockfish NFS permissions. The
captured `df` output is filesystem-wide and cannot establish a personal quota
change. SQLite quick-check returned `ok` and `foreign_key_check` returned zero
rows after the completed batch. Full `/data` and `/scratch4` large-file
inventory, repository branch reconciliation, GitHub PR completion, and
transfer to Skipjack remain separate work items; this exact-name scan does not
claim a complete file-level inventory of either whole filesystem.

## Update — 2026-11-24 residual restart cleanup and PR evidence catalog

A fresh metadata-only walk of the accessible `/scratch16/pclancy3/yi` and
`/data/pclancy3/yi` trees covered 114,907 directories with zero traversal
errors. `revision1_migrationbench_runs` now has no basename containing `wfc` or
`mix`; the previously reported 3,817 WFC and 481 mixing restart files are
absent. Across these accessible personal roots, a direct exact QE restart scan
found no nonempty WFC or mixing restart payloads. It found eight zero-byte
`.mix1`–`.mix4` placeholders in two Cr-in-gap output directories. After an
all-user queue check and exact owner/mode/inode/mtime/size checks, those eight
empty files were removed and verified absent. They freed zero logical bytes.
The delete receipt and path-level evidence are under
`phase2_data/wfc_mix_residual_20261124/`.

The broad WFC/mix token inventory contains 3,715 remaining names across the
two accessible personal roots. These include `projwfc`/PDOS outputs and
software/library names, not exact QE restart filenames; no payload was removed
based on those name matches. The full metadata-only inventory is retained as a
separate audit artifact. This does not complete the inaccessible portions of
`/data` or `/scratch4`, and a new personal quota reading remains unavailable.

The eight zero-byte placeholder deletions are recorded as the separate batch
`RF-YI-QE-ZERO-MIX-CLEANUP-20261124`. The SQL ledger retains the prior `HOLD`
decision rows, adds eight `DELETE/VERIFIED` rows, and links both the deletion
receipt and a later independent exact-path check (8 absent, 0 errors at
08:58:26Z). The receipt's all-user queue check is summary-only because no raw
queue snapshot was retained. Logical bytes freed are zero; allocated bytes and
quota delta are unmeasured and represented as unknown in SQL.

The PR #10, #12, and #13 source packages were imported into the separate
`provenance_catalog.sqlite` as three `REVIEW` candidates, preserving the
submitted release. The import adds 19 source-claim records (13 for PR #10, six
for PR #13), three unselected PR #12 figure candidates, and source-linked
package/evidence records. For PR #13, all 1,336 raw-evidence files were
re-hashed on Rockfish against the source manifest and registered as present
archive locations; this remains distinct from their absence in both pinned
Git trees. The one stale README manifest declaration is retained separately
from the verified current Git README. The import transaction completed with
zero foreign-key violations. The post-import SQLite `quick_check` returned
`ok`; the source-path reconciliation found all 1,336 expected Rockfish raw
archive locations with zero missing paths or hash/size mismatches.

## Selective Revision 1 MD evidence archive — 2026-11-24

The central Rockfish archive now holds a 49-file, 2,320,577,507-byte evidence
bundle for six grouped-A2 MD continuations referenced by Fig. 6 / Table S8 / R7.
It includes six coordinate trajectory segments, six source and six final
coordinate checkpoints, six per-seed model weights, one shared initial
structure, and manifests, status records, metrics, and continuation ledgers.
All 49 destination files passed size/SHA-256 verification. The checkpoint step
matches the `last_step` in each run status record. One run reached 1,000,000
steps; five stopped at earlier scientific thresholds. This six-run subset does
not replace the complete 15-model evidence cohort, and the archived outputs do
not establish scientific acceptance. Original source files remain in place.
The archive manifest, outcome table, and verification receipt are registered in
SQLite; the submitted Fig. 6, Table S8, and Table R7 rows link to this archive
subset through `paper_release_item_evidence`.

The eight zero-byte Cr-in-gap mixing placeholders have a separate SQL batch
`RF-YI-QE-ZERO-MIX-CLEANUP-20261124` with eight `DELETE/VERIFIED` paths and a
full empty-file SHA-256. The earlier `HOLD` batch remains as the prior decision
state. The user-directed 4,298-file batch remains separately recorded as
3,817 WFC plus 481 mixing restarts.

## SQLite catalog snapshot — 2026-11-24

The integrity-checked `provenance_catalog.sqlite` snapshot is stored under
`/scratch16/pclancy3/yi/MigrationBench_data_archive/provenance_catalog_snapshots/20261124/`.
The adjacent `DB_SNAPSHOT_MANIFEST.json` records the source and Rockfish
SHA-256, exact byte count, successful `PRAGMA quick_check`, zero foreign-key
violations, and verification time. The complete database remains outside Git;
the repository carries the schema, import scripts, query cookbook, and archive
pointer.
