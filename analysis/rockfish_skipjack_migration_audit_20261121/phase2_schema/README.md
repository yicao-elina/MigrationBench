# Phase 2 provenance catalog schema

## Purpose

`001_provenance_catalog.sql` defines a new, additive SQLite control-plane database for the Rockfish inventory and the evidence paths the cleanup depends on. It is designed to sit beside the existing project registries rather than replace them.

The catalog keeps separate records for:

- filesystem roots, group/user quota observations, inventory snapshots, and per-path observations;
- projects, campaigns, calculation/workflow runs, Slurm job-state observations, artifacts, and artifact locations;
- exact rerun commands, launch scripts, manifests, environment information, and required recipe inputs;
- derivation edges, scientific quality gates, claims and figure/table evidence;
- immutable references into existing registries, Git repository/ref snapshots, duplicate groups, and cleanup events.

Scheduler completion and scientific acceptance remain separate fields. A completed job does not imply a valid scientific result. An inventory scan's timestamps, byte counts, and hashes are observations, not acceptance judgements.

## Existing sources to retain and link

Use `registry_source` plus `registry_entity_ref` to link rows in the new catalog to existing sources. Keep each source database intact and record the exact file hash, capture time, and as-of date. Initial sources already present in the unified `main` worktree include:

| Existing source | Existing role | Integration approach |
|---|---|---|
| `registry/snapshots/migrationbench_asset_registry_core_20261101.sqlite` | MigrationBench model, task, system, campaign, asset, chart/panel, and chart-provenance snapshot | Treat as immutable, stale-by-date source; use typed external IDs for models, systems, campaigns, charts, and panels. |
| `data/collaborator_dft_neb/migrationbench_dft_neb.sqlite` | DFT-NEB calculations, artifacts, structures, endpoints, trajectories, quality gates, and publication links | Link calculation/endpoint/trajectory/artifact IDs. Its current schema has no artifact-to-publication completeness guarantee, so retain partial lineage flags. |
| `data/processed/cluster/training_provenance/zero_force_training.sqlite` | Frame-level training provenance and zero-force audit | Link dataset/frame keys; do not interpret this as a complete source-run provenance graph. |
| `analysis/rockfish_skipjack_migration_audit_20261121/cleanup/cleanup_audit.sqlite` | Prior local and Rockfish deletion ledger, evidence hashes, and local placeholders | Import or map its immutable row identifiers into `cleanup_item.legacy_cleanup_*`. Do not recast historical evidence as current inventory. |
| `results/plots/registry/migrationbench_asset_registry_20261101.sqlite` and `_base_20261028.sqlite` | Plot/gallery registry versions | Record separate source snapshots. Compare to core registry; do not collapse versions by filename. |

Rockfish Slurm job numbers can be reused. Treat a job instance as the
combination of cluster, job ID, array task ID, and `submitted_or_started_utc`;
store the submission timestamp when available, otherwise the evidenced start
timestamp. Do not join historical `sacct` rows to runs by numeric job ID alone.

The MigrationBench asset registry is the most mature project entity model, but it is not a general filesystem inventory. The DFT-NEB database has run/artifact/result entities, but its publication links do not fill the missing source artifact and claim mappings. The cleanup audit tracks deleted items but not the full retained file graph. The additive catalog supplies the common cross-project path and relationship layer while these records remain authoritative for their native tables.

The current additive schema has sixteen applied migrations. `001_provenance_catalog.sql`
creates the initial schema. `002_hardlink_counts.sql` adds observed link count
and an index on `(device, inode)`. `003_quota_file_counts.sql` adds GPFS file
counts beside block quotas. GPFS machine-readable block counts are 1 KiB units
and are multiplied by 1024 for byte-valued columns; a zero limit means no
configured limit and is stored as SQL `NULL`. Apply migrations in numeric order
and retain their rows in `schema_migration`.

`004_git_snapshot_observations.sql` adds Git worktree, remote-freshness, and
ref-graph snapshot detail, including a nullable `shallow_clone` observation.
`005_remove_redundant_storage_path_index.sql` drops only the redundant ordinary
`idx_path_root_rel`; the unique constraint on
`storage_path(root_id,relative_path)` remains. `006_cleanup_execution_receipts.sql`
normalizes per-batch deletion outcomes, before/after GPFS quota observations,
and queue-receipt references. `007_git_merge_observations.sql` records local
mainline reconciliation as an event distinct from pushing.
`008_duplicate_group_evidence.sql` links each duplicate group to immutable
hash, inode, review, or cleanup-manifest sources through a many-to-many bridge.
`009_paper_figure_archive.sql` adds typed paper-figure records, links figure
exports/data/manifests/rebuild pointers to hashed artifacts, and records source
claim status separately from the catalog's current scientific assessment.
`010_paper_claim_figure_links.sql` adds explicit source-backed links from a
paper claim to a figure entity; the Static-639 reconstruction row has one
figure-level link and no inferred panel map. `011_project_artifact_lookup_indexes.sql`
adds indexes on `(project_id, artifact_key)` and `(path_id, artifact_id)` for
project-scoped artifact search and reverse path lookup. An `EXPLAIN QUERY PLAN`
check confirmed both lookups use their new indexes. `012_revision1_release_and_scheduler_evidence.sql`
adds the submitted Revision 1 release index and per-run scheduler/cleanup
evidence. `013_duplicate_source_claim_rows.sql` preserves multiple distinct
reviewer questions that reuse a source claim label. `014_incomplete_recipes_allow_missing_command.sql`
lets incomplete or blocked recipes preserve an absent exact command without
invented placeholders. `015_cleanup_receipt_allocated_bytes_nullable.sql`
keeps unmeasured allocation totals as SQL `NULL`. `016_artifact_alias_redirects.sql`
records canonical artifact redirects while preserving alias identities and
historical path observations. The unique storage-path constraint remains.

The read-only Git reconciliation staging plan has been extended to 26 local
repositories, 31 captured local branch refs, and 297 cached remote-tracking
refs. Repo 26 is the shallow upstream LAMMPS `develop` base plus local FLARE
customization: one tracked edit and 406 untracked source paths are linked to
their saved hashes or symlink-target identities; live upstream tips remain
unknown. This snapshot importer checks that migration 004 is already recorded
and will not apply or replay it.

The 2026-11-22 isolated local merge candidates have explicit repository
snapshots and ref observations. Their eight refs are joined to eight
`git_ref_context_snapshot` rows; two `origin` observations preserve the
unverified-current-tip status, and five branch-pair rows record the captured
ancestry. MigrationBench `dev` is contained by candidate `main`; the `hpc`
branch is contained by the local `md_path_search` merge candidate. These are
unpushed local facts, and the original Rockfish refs/worktrees remain intact.

The LAMMPS build cleanup is complete in batch
`RF-LAMMPS-BUILD-DELETE-20261122-R01`: 1,876 generated files / 4,052,888,613
logical bytes (4,090,057,216 allocated bytes) were removed after the complete
candidate manifest, queue, and preserved dependency tree passed the guard.
The 5,689-file KokkosKernels subtree remains in its verified sibling location.
The Slurm rebuild recipe, customized source commit, dirty-source manifest,
dependency manifest, preflight, execution receipt, and quota observations link
to the `FLARE_LAMMPS` workflow record.

The unused `libstdcxx-14.2.0-h8f9b012_2` extracted Conda package cache was
removed in batch `RF-DATA-CONDA-PKG-LIBSTDCXX-UNUSED-20261122-R01`: 88 files,
two symlinks, and 15 directories. Its exact archive URL/SHA-256, live
pre-delete manifest guard, 20-prefix non-reference scan, queue check, per-leaf
receipt, post-delete check, and cache recreation script are linked to a
documented `HPC_ENVIRONMENTS` recipe. The cache location is marked `REMOVED`;
the recreation recipe is documented but was not executed.

## Cleanup invariants

1. Inventory each path and record the scan completeness and quota snapshot first. Preserve partial/unreadable status rather than treating an incomplete traversal as an empty directory.
2. Hash large files only for candidates or identity-critical artifacts; record `hash_scope`. A filename, size, directory label, `mtime`, or Slurm state alone is not a content identity.
3. Mark third-party, shared-owner, and unknown-owner artifacts explicitly. The DDL blocks a delete row unless the artifact is user-confirmed and owner clearance is recorded. The user explicitly authorized `/data/pclancy3/for-victor` and `/data/pclancy3/for-yuqi` to go through the same inventory, reproducibility, duplicate, Git, and cleanup pipeline as the rest of this project storage. Do not apply a separate retention rule to either directory; classify each path with the same evidence and owner checks used elsewhere.
4. For WFC and other regenerable intermediates, first link a concrete run, input, output, QE/other launch script, Slurm submission script and scheduler records; record the exact command/environment and manifest where available. The DDL blocks deletion staging for `REPRODUCIBLE_INTERMEDIATE` unless the recipe, script artifacts, manifest, input, and output are linked. Also verify the job is not live and that no downstream manifest or workflow requires the restart.
5. Preserve charge density, XML/result files, converged structures, compact tables, paper/figure inputs, failed-run evidence and the exact recipe unless separate evidence supports another retention decision. `FAILURE_MODE` or `EXCLUDE_FROM_PUBLICATION` does not by itself mean delete.
6. For duplicates, require SHA-256 or byte comparison before `VERIFIED_IDENTICAL`; metadata-only matches remain `LIKELY` and are not deletion grounds.
7. Record the pre-delete path, size, `mtime`, hash, reason, evidence bundle, and exact outcome. Re-scan and refresh GPFS group quota after each completed batch.
8. Preserve each Git repository's unique commits and dirty/untracked files before merging or pruning refs. Ref snapshots and per-run commit bindings belong in `git_ref_observation` and `run_git_binding`; a merge does not prove that the associated raw data were transferred.

## Current known inventory boundary

The `/data/pclancy3` R01 baseline has a complete UID 3901 metadata walk across
the user tree and the additional `ycao73_home`, `for-victor`, `for-yuqi`, and
`MHP_defect` roots; it is a pre-cleanup snapshot and later deletion receipts
supply the changes. The independent `/data` mount walk did **not** complete: 625 root
directories are inaccessible to this account, 187 additional permission
errors occurred before the scan was stopped, and only partial traversal had no
UID 3901 hits. A separate quota reconciliation estimates that about 6.279 TB
of the user's `/data` quota cannot be tied to the visible UID 3901 paths.
There is no evidence that inaccessible paths are stale or disposable.

The complete metadata-only `/scratch4/pclancy3` group walk records 1,540,972
observations, including 98 unreadable directories. It does not cover the entire
`/scratch4` mount. Its 1,380 files larger than 500 MB sum to 6.518 TB; ownership
is mixed, so this is a review set, not a user cleanup list. The post-cleanup
`/scratch4/pclancy3/yi` R02 is complete: 2,878 files, 597,580,272 logical
bytes, no WFC/mixing restart files, and no file larger than 100 MB.

The 2026-11-23 refreshes are additive `PARTIAL` snapshots. The `/data/pclancy3`
R02 emitted 598,316 observations before disconnecting inside `yi/.conda`; its
first four selected roots (`MHP_defect`, `for-victor`, `for-yuqi`, and
`ycao73_home`) were traversed according to control flow, without explicit
per-root completion markers. The same review and evidence thresholds apply
to `for-victor` and `for-yuqi`; neither was skipped or treated differently
because its collaborator graduated. The `/scratch4/pclancy3` R02 emitted
924,502 observations before a stalled third-party package-tree traversal was
terminated, including 98 unreadable paths. It is a lower bound, not a new
complete inventory. Its emitted large-file rows match R01; it found no
confirmed QE WFC restart candidate. Six duplicate indicators totaling 2.065
GB are metadata-only and unverified. Neither partial refresh authorizes
deletion by itself. Their raw streams, scanners, reports, summaries, and
selected review tables are separately hashed and registered.

The catalog also indexes the local DUAL-X Safe Core v6 bundle under the
separate root `LOCAL-DUALX-SAFECORE`; it is not a Rockfish storage inventory.
Its 119-file SHA-256 manifest was verified, then imported as 135 artifacts,
four workflow runs, seven figure IDs, 18 source claims, and explicit links
from figure exports to exact source tables and renderer scripts. The three
model checkpoints are external pointers without resolved paths. The original
inference recipe is incomplete; only the frozen-inference analysis, figure
render, and clean-checkout validation recipes carry the source-recorded
`REPRODUCED` state. Source claim statuses remain separate from catalog
assessment, and the older Rockfish X-FORCE tree has not been joined to this
local release by name alone.

At 2026-11-23 01:16:09Z, the latest stored quota rows were 14,941,272,342,528
bytes used by group `pclancy3` on `/data` against a 10 TiB hard limit (grace
expired; 336,855,040 bytes in doubt), 9,984,613,859,328 bytes for the group on
`/scratch4` against 10 TiB (982,532,096 bytes in doubt), and 6,563,841,441,792
bytes for user `ycao73` on `/data` with no configured limit. The exact Conda
package-cache cleanup receipt brackets a 14,745,600-byte and 105-entry quota
drop. These counters
cover shared filesets and do not measure a single inventoried subtree. The
schema's cleanup guards still require ownership and a linked reproduction
chain; no schema definition authorizes deletion by itself.

The 2026-11-23 access refresh found 625 of 725 immediate `/data` directories
and 503 of 562 immediate `/scratch4` directories non-traversable by the
current account. Aggregate boundary evidence and the fresh 1,295-job queue
capture are registered as immutable sources. The all-user queue had 116
WorkDirs under `/data`, 507 under `/scratch4`, none under `/data/pclancy3`, and
two under `/scratch4/pclancy3` (one running). Fresh inventories of the
readable project roots are being kept as separate snapshots.

An immediate-child owner probe found no UID 3901 entries directly in either
mount root. The only project-GID 3001 entries are `/data/pclancy3` and
`/scratch4/pclancy3`; other immediate entries belong to other UID/GID
identities and are outside this account's project-tree walk.

At 2026-11-23 02:39:59Z, the refreshed `/data` group row remained at
14,941,272,342,528 bytes against 10 TiB with grace expired; `/scratch4` group
usage was 9,984,613,859,328 bytes. That later raw capture is registered, but
it does not localize the shared `/data` overage to user-owned paths.

A narrower refresh at 2026-11-23 01:31:02Z showed one running job inside
`/scratch4/pclancy3`, in `/scratch4/pclancy3/madaline/gpaw_port`; no other
project-group WorkDir appeared in that capture. This later snapshot supersedes
the earlier queue rows for current cleanup guards, while per-batch live checks
remain required.

The later `pclancy3` account capture at 02:10:28Z again found running job 33224712 in
`/scratch4/pclancy3/madaline/gpaw_port`; pending jobs in that capture were
under `/scratch16`. Treat the `madaline` tree as active until a fresh queue
check clears it. The partial Git metadata pass did not run `git status`, so its
repositories have unknown worktree cleanliness.

The 02:39:59Z group quota capture shows `/data` at 14,591,086,272 KiB against
10,737,418,240 KiB, grace expired; `/scratch4` is at 9,750,599,472 KiB. The
over-limit `/data` count is shared across the group and has not been assigned
to this user's paths. Local Rockfish Git refs remain unchanged: candidate
objects are missing from their target repos and importing them would write to
the over-limit `/data` fileset. No write was attempted and no push occurred.

## Ongoing catalog maintenance

1. Register every new immutable manifest, report, queue capture, and quota capture by exact hash and timestamp.
2. Add refreshed inventories as new snapshots; retain earlier snapshots and deletion-time receipts.
3. Link runs, Slurm job instances, recipe scripts, inputs, outputs, and figure/claim evidence where supported; leave lineage partial where identifiers cannot be resolved.
4. Keep path-summed logical size separate from unique-inode allocation and quota counters.
5. Stage further cleanup only after ownership, active-job status, restart consumers, hash identity, and exact reproduction evidence are joined in SQL.

The read-only query cookbook in `catalog_queries.sql` provides starting points
for project-to-run-to-recipe/artifact tracing, claim-to-figure evidence,
cleanup receipts with quota brackets, latest inventory snapshots,
project-scoped artifact search, and query 10 for inventory-path-to-explicit-
artifact/project/run/recipe lookup. Queries 1/1b/1c return run+recipe versions,
run artifacts, and Slurm history as separate result sets to avoid
many-to-many row multiplication; query 6 lists exact recipe inputs. Query 7
shows explicitly linked claim/figure assets, and query 8 recursively follows
only exact artifact IDs across recorded derivation edges to upstream files,
scripts, runs, recipes, and Git commits. Query 9 lists incomplete run-to-recipe,
input/output, Slurm, and Git coverage; query 9b lists claims without an explicit
figure relation. Query 3 separates item receipts from
batch execution/quota/queue receipts. Missing links stay visible and no
lineage is inferred from filenames. Bind the example parameters rather than
building SQL from untrusted path text.

## Provenance coverage and review limits

The schema already represents project/campaign → workflow run → inputs,
outputs, logs and provenance artifacts → reproduction recipe and exact
recipe inputs → Slurm job instances and observed states → repository commit
bindings. Derived artifacts have `derivation_edge` links; paper claims connect
to source evidence and explicitly registered figures/assets; inventories and
cleanup batches retain dated observations, per-path decisions, execution
receipts, queue snapshots, and quota brackets. A schema migration is not
needed just to express these relationships.

Catalog coverage is still partial. After importing the three QE mixing cleanup
runs, the current catalog has 22 workflow runs, 22 recipes, 11 Slurm job
instances, and 5 run-to-Git bindings.
Three runs have no recipe, 12 have no Slurm record, and 17 have no Git binding.
Across 27 paper claims, 12 have an explicit figure relation. These gaps remain
visible and are not filled from directory names, job IDs alone, or temporal
proximity. The underlying databases and dated registries remain linked as
source records where their native mappings are more complete.

The catalog currently contains 102 inventory snapshots (11 partial) and
6,774,494 per-path observations. R03 added exact snapshots for four
`/data/pclancy3/yi/.conda` environment roots and selected `/scratch4`
subtrees. The Conda snapshots reconcile with the prior complete baseline,
partial R02 stream, and cleanup receipts to 689,599 current paths, with no
uncovered or unexpected paths.

R04 adds 68 Scratch4 snapshots / 877,292 observations: 63 COMPLETE recursive
subtree scans, four PARTIAL timed-out parent scans, and one PARTIAL 31-path
targeted lstat sample. The cganley user-tree child union exactly matches R01
at 379,462 paths; 16 other inactive roots match R01 at 158,447 paths. These
R04 scans were metadata-only. The cganley union belonged to UID 1310; the
other 16 roots had no UID 3901 entries. These scans are historical evidence,
not current cleanup candidates. The user's renewed whole-mount request puts
accessible `/data` and `/scratch4` paths back into inventory scope, but those
other-owner trees remain read-only without ownership clearance.

The current request covers both mounts end to end. The selected audited
subtrees include `/data/pclancy3/yi`, `/scratch4/pclancy3/yi`,
`/data/pclancy3/ycao73_home`, `/data/pclancy3/for-victor`, and
`/data/pclancy3/for-yuqi`; Victor and Yuqi receive the same pipeline as other
project-owned files. `/scratch4/pclancy3/yi` has a COMPLETE R03 snapshot with
2,875 files, 59 directories, zero unreadable entries, no files above 100 MiB,
and no QE WFC/mixing filename matches. `/data/pclancy3/yi` has a complete
777,719-path R04/R03 composite; `/data/pclancy3/MHP_defect` has a complete
193-path R04 snapshot reconciled to all 490 verified WFC cleanup rows. These
selected snapshots do not imply mount-wide coverage.

The latest quota capture requested only the `ycao73` user rows: 6,558,488,461,312
bytes / 658,801 files on `/data`, and 403,701,760 bytes / 2,934 files on
`/scratch4`. The queue capture was filtered remotely to WorkDirs under the two
`yi` roots and returned no rows. Quota counters are filesystem-level user
totals, not usage localized to either directory.

The cleanup-stage trigger is a guardrail, not a substitute for review. For a
`REPRODUCIBLE_INTERMEDIATE`, it checks that a recipe is at least
`DOCUMENTED`/`REVIEWED`/`REPRODUCED`, required script/manifest/input/output
relationships exist, and some retained input/output locations are available.
It does not by itself prove every `recipe_input.required` item is present, that
the historical Slurm job and Git commit are bound, or that a fresh recipe was
successfully run. Before each deletion, require the exact recipe and retained
input/output/script/Slurm evidence, explicit downstream restart-consumer
review, an all-user active-job guard, and full current path identity/hash
rechecks; a `REVIEWED` fresh-start recipe is not an exact continuation of the
deleted historical state.


## 2026-11-23 QE mix batch and whole-mount status

Batch `RF-DATA-YI-QE-MIX-R04-CLEANUP-20261123` is `COMPLETED`: 384 verified
`.mixN` files removed, 7,500,169,152 logical bytes and 7,518,289,920 allocated
bytes. Three workflow runs and their `REVIEWED` fresh-start recipes link the
inputs, outputs, QE and Slurm scripts, pseudopotentials, QE binary identities,
job instances, and manifest. The recipes were not run; historical scientific
status remains `PROVENANCE_PARTIAL`. Forty-eight paths remain held because
the candidate run directory does not match the historical `sacct` WorkDir.
The post-verifier confirmed all 432 staged paths: 384 absent, 48 present with
their original hashes, no mismatches. The cleanup receipt, per-file queue and
SHA guards, before/after quota captures, and post-verifier are linked in SQL.
`/data` group quota decreased by exactly the cleanup's allocated bytes;
`/scratch4` quota did not change. `/data` remains 3,933,284,794,368 bytes over
its shared 10 TiB limit.

The composite shows 1,592 exact `.mixN` paths remain after deletion. A focused
read-only review found 1,544 files / 8,313,215,328 logical bytes in 18
directories outside the 48 held paths; none are deletion-ready. Fourteen are
restart-mode NEB trees. Two `from_scratch` single-image jobs have image-4 SCF
failure evidence, so their mixing histories remain useful continuation state
until quality and consumer review is complete. Two more locations contain
only empty placeholders.

The refreshed Git review records 18 of 25 `yi` repositories and 3 of 4
`ycao73_home` repositories as dirty. Original Rockfish refs and dirty
worktrees remain unchanged; local MigrationBench and DUAL-X consolidations
remain unpushed. Isolated candidates align LLM4Chem and DynamicSciKG `main`
with live `main`, promote neuromorphic-filament's live `hpc` tip to local
`main`, and retain the verified `md_path_search` main+hpc merge. No GitHub
refs were pushed. A fresh full matrix covers all 29 repositories under `yi`
and `ycao73_home`: 27 have usable cached-canonical ancestry comparisons,
two lack a canonical remote; 13 cached default refs differ from live defaults
and all 13 target objects are absent locally. No further merge is safe from
available objects. The unique cached MACE commit remains held because its
current live `develop` target is missing from the local object store. See
`phase2_data/full_git_reconciliation_r04_20261123/`. The 2026-11-23 root
access probe found 625 of 725 immediate
`/data` directories and 503 of 562 immediate `/scratch4` directories
non-traversable, plus 98 unreadable directories in the broad
`/scratch4/pclancy3` walk. Whole-mount completion is therefore still open. Do
not infer stale files from inaccessible paths or delete another owner's
content without clearance.

The full Git reconciliation importer records 61 local/cached/live ref
observations across all 29 repositories in `git_ref_observation` and links the
receipts, matrix, and report to each `git_repository`. Live refs were observed
with `ls-remote`; the importer did not fetch objects or alter refs.

The R05 accessible `/data` outer-root scan attempted all 99 readable
immediate roots outside `/data/pclancy3`: 66 completed and 33 are partial.
It observed 106,336 paths / 9.592 TB logical as lower bounds. The partial
`RF-DATA-FS-R05-ACCESSIBLE-ROOT-CENSUS-20261123T085425Z-PARTIAL` SQL snapshot
contains 106,965 rows including 625 non-traversable root directories, the
`/data/pclancy3` boundary and two top-level files with unknown size. It does
not treat missing descendants as deletions. The bounded `/scratch4` outer-root
scan attempted all 58 accessible roots outside `/scratch4/pclancy3`: 18 were
empty and all 40 non-empty roots were attempted. Fifteen completed, 11 yielded
partial output, and 14 timed out. The 26 roots with emitted file totals
observed 309,515 files / 32.790 TB logical bytes as a lower bound; these are
other-owner paths. A separate UID 3901 check attempted all 58 roots, with 34
complete, 15 partial, and nine timeouts. It emitted no matches; partial and
timed-out roots remain unknown. The aggregate reports are linked to
`RF-SCRATCH4-FS` in the registry as mount-scope evidence, not a full per-file
`inventory_snapshot`.

A later migration can add geometry, model-training membership, environments as normalized package manifests, and file-ownership approvals without changing the initial storage/provenance relationships. The dated MigrationBench core and plot registries, the local figure archive, and the nine exact static-639 claim rows are now linked in the catalog. The claim matrix's source statuses and action strings live in `paper_claim_source_observation`; the current claim assessment remains provenance-partial when run/job lineage does not resolve. Claim-evidence support is `UNKNOWN` until reviewed; a matrix-listed file path is not automatically a scientific support decision. The MD FIG-004 A–G rows remain panel descriptions/gates, not paper claims. The DFT-MACE47 manifest is a separate review candidate whose C1–C6 matrix is not materialized in this archive; the Canva originals are missing; and the collaborator DFT-NEB source still has no publication links or result evidence-artifact bindings. Do not invent claim-to-run joins for any of these gaps.

## 2026-11-23 Revision 1 release and archive imports

`MB-REVISION1-SUBMITTED-20261110` records 52 submitted result/panel items, 36
reproducibility rows, and 22 reviewer claim observations (20 source claim
labels, with repeated IDs retained as separate rows). It has 319 item-to-
evidence links, including eight explicit links to Rockfish archive copies of
Fig. 6's reproducibility input. The canonical in-gap DFT barrier remains
source-qualified: 81 atoms, 10 images, 1.524299 eV, scheduler-completed but
SCF-warning and excluded from publication ground truth. Older 0.41 eV,
0.336050 eV, and 1.52 eV records remain separate where their structures,
protocols, or convergence evidence differ.

Restart cleanup batch `RF-SCRATCH16-YI-REV1-QE-RESTART-CLEANUP-20261123` is
`COMPLETED`: 4,298 WFC/mixing files were removed from
`/scratch16/pclancy3/yi/revision1_migrationbench_runs` (1,036,297,176,856
logical bytes). The manifest is metadata-only (`pre_delete_sha256` is NULL),
with coordinate/XML, path, scheduler, and queue evidence retained. This count
is scoped to that directory; wider Rockfish restart sets have their own audit.

Two complete archive inventories extend the same SQL path catalog. The initial
X-FORCE import had 1,094 paths / 5,540,622,192 logical bytes. After the
2026-11-23 model and E17 evidence closure, its updated manifest has 1,103 files
/ 5,653,787,919 logical bytes (1,072 snapshot assets plus 31 Git-index TIFF
exports), including four previously ignored model checkpoints, four E17 result
JSONs, and the staged comparison script. Its source tree was dirty, so these
are preservation records, not scientific validation. MigrationBench has 14 large
Git-index assets / 405,789,182 logical bytes, including Blender scenes,
high-resolution renders, two duplicate-content Fig. 6 input arrays, and one
video; these were SHA-256 verified in the Rockfish archive before leaving the
current Git tree. Allocated-byte fields for these archive snapshots are NULL
where immediate GPFS `st_blocks` readings were inconsistent. Aggregate `du`
observations are kept in each snapshot's notes. Use
`xforce_catalog_queries.sql` with `catalog_queries.sql` and
`revision1_catalog_queries.sql` for project, paper, cleanup, and archive
lookups. The import logic is in `import_rockfish_asset_archives.py`.

### 2026-11-23 historical Revision 1 restart cleanup receipt (prepared)

The old `remove_stale_wfc_mix_restarts_20261122_v1` action and its 574 path rows
have already been merged into the unified `cleanup/cleanup_audit.sqlite`. The
catalog importer is `import_revision1_historical_cleanup_574.py`; its source
connection is SQLite read-only. It checks the unified ledger against the
immutable 574-path TSV, the merge receipt, and Rockfish's post-delete
verification JSON before writing the provenance catalog. It does not repeat
the prior cleanup-ledger merge.

The prepared import records 574 historical file artifacts and their removed
locations, one completed cleanup batch, 574 verified delete items, and an
execution receipt for 1,536,705,954,176 logical bytes. The execution receipt
must keep `actual_allocated_bytes` **NULL**: the source did not record per-file
or aggregate allocated bytes. `pre_delete_sha256` and artifact content hashes
remain NULL for all 574 paths, and file mode remains unrecorded; the `not_recorded`
legacy sentinel is not treated as a mode value. The path-manifest SHA-256
identifies the TSV evidence file only.

Run migration `015_cleanup_receipt_allocated_bytes_nullable.sql` before the
importer so an unknown allocation total can be stored without converting it to
zero. The importer requires migration 015, checks source counts, statuses, byte
totals, manifest identity, post-delete absence, and foreign keys, and is
idempotent for the registered source hashes. It links a historical run only
when `scheduler_job_source_observation.job_id` has exactly one distinct
`workflow_run`; it does not infer a match from a legacy `run_id` string, path,
or directory name. Unmatched or ambiguous job IDs stay unlinked. The original
master-snapshot `cleanup_artifact.artifact_id` values differ from the re-keyed
IDs in the unified ledger, so `registry_entity_ref` stores an explicit
`REIDENTIFIED_AS` crosswalk from each original ID to the catalog artifact;
`cleanup_item.legacy_cleanup_row_key` continues to identify the unified-ledger
row. Corresponding artifact records retain partial provenance status.

Migration 015 was applied before the historical 574-file import. The batch
`RF-SCRATCH16-YI-REV1-574-RESTART-CLEANUP-20261122-V1` is recorded as completed
with 574 verified file rows and 1,536,705,954,176 logical bytes. Its aggregate
allocated bytes, per-file payload hashes, and file modes remain unknown; the
manifest hash identifies the TSV evidence, not the deleted payloads. Original
cleanup IDs have explicit crosswalks to unified-catalog IDs. An earlier
preflight failure is retained as historical evidence; it was followed by the
patched importer and post-import integrity checks.

### 2026-11-23 coordinate-supported WFC and mixing restart cleanup

Batch `RF-YI-QE-RESTART-CLEANUP-20261123` is `COMPLETED`. It removed 7,581
files from 46 non-active run roots: 5,204 `.wfcN` files (297,054,172,800 B),
201 `wfc*.dat` files (906,736,303,792 B), and 2,176 `.mixN` files
(21,533,350,336 B). The exact total is 1,225,323,826,928 logical bytes and
1,225,535,520,768 pre-delete allocated bytes by `stat`. Every deleted path was
matched against the reviewed manifest and pre-delete stat identity, then
verified absent. The guarded executor made 46 fresh all-user queue checks
immediately before each run-root mutation; it found no overlap. Restart payload
contents were intentionally not hashed. The retained XML and NEB path evidence,
plus the per-file deletion receipt, remain linked through SQL.

An independent direct Rockfish scan of both `/scratch16/pclancy3/yi` and
`/data/pclancy3/yi` at 12:23:47Z found zero `.wfcN`, zero `wfc*.dat`, and eight
zero-byte `.mixN` placeholders (0 logical bytes). Those exact paths and inode
values match the eight `HOLD` rows; the corrected scan, erratum, inventory, and
receipt are in the private archive
`/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_restart_cleanup_20261123/`.
The earlier hand-transcribed scan JSON with one path typo is retained there as
superseded evidence and is not used by the catalog.

The 48 WorkDir-mismatch mixing files were included only because a same-prefix,
parseable two-atom XML postdated them; their scheduler WorkDir relationship
remains explicitly uncertain, and SCF/error status remains attached. None of
these files is promoted to scientific ground truth. Personal `quota -s` reads
were denied by Rockfish NFS permissions. The captured `df` values are
filesystem-wide only, so this batch has no quota-after observation and no claim
that personal quota fell by the deleted file size. SQL quick-check returned
`ok`; `foreign_key_check` returned no violations.

Use query 3 for the per-file ledger and query 11 for receipt, post-delete scan,
quota command output, and their hashed Rockfish archive paths. The finalizer is
`finalize_restart_coordinate_cleanup.py`; the repeatable exact-name scanner is
`scan_postdelete_restart_names.py`.

Schema migration 016, `016_artifact_alias_redirects.sql`, adds an explicit
`artifact_alias` relation and records version 16 in `schema_migration`. It keeps
alias artifact rows and their historical locations while allowing cleanup and
run relationships to point to a canonical artifact. Apply migration 016 before
running `supplement_completed_restart_cleanup_r03.py`; its default mode is a
read-only preflight, and `--apply` is transaction guarded.

## PR #10/#12/#13 manuscript evidence candidates — 2026-11-24

The pinned source files for these three open pull requests were imported as
separate `REVIEW` releases by
`import_pr10_pr12_pr13_candidate_releases_20261124.py`. The importer validates
captured source hashes and PR-head metadata, and live-rehashes the 1,336 PR #13
raw-evidence files on Rockfish before creating their `PRESENT` archive
locations. Their absence from both pinned Git trees remains a separate status.
PR #10 has 13 source claim rows, with its malformed source digest retained and
a separately verified content SHA-256 recorded. PR #12's three figures remain
unselected and `PROVENANCE_PARTIAL`. PR #13 has six source claims; one stale
README manifest declaration is distinct from the current verified Git README.
None of these imports changes the submitted Revision 1 release or upgrades a
scientific status based on a hash alone. Use
`pr_candidate_release_catalog_queries_20261124.sql` to retrieve candidate
claims, figure assets, Git-tree reconciliation, and exact Rockfish archive
paths. The catalog is stored separately from GitHub because its SQLite file is
3.97 GB; a checksummed snapshot is archived in the Rockfish data archive.

The 2026-11-24 model/vector archive is imported by
import_revision1_git_large_asset_archive_20261124.py. Its five archived
payloads are queryable as project artifacts with live-verified Rockfish
locations. The audit deliverable registry also indexes the manuscript
alignment report, the archive controls, model-loader pointers, cleanup
receipts, and candidate release captures. Query 12 retrieves the five
model/vector payloads; query 13 retrieves the linked audit evidence packages.

`import_revision1_md_fig6_archive_20261124.py` registers the selectively
archived six-run grouped-A2 MD subset, including its coordinate segments,
source and final checkpoints, per-seed model weights, shared initial
structure, run metadata, source hashes, and Figure 6 / Table S8 / Table R7
evidence links. The bundle contains 49 payloads / 2,320,577,507 bytes; it is
not the full 15-model cohort. `revision1_catalog_queries.sql` includes a
read-only retrieval query for these paths and hashes.

The 2026-11-24 deletion receipt for eight exact zero-byte Cr-in-gap `.mix1`–
`.mix4` placeholders is imported separately by
`import_zero_byte_mix_cleanup_20261124.py`. Its default mode is a read-only
preflight; `--apply` performs one guarded transaction. The earlier 2026-11-23
`HOLD` rows remain as historical decisions; the follow-up batch records all
eight `DELETE/VERIFIED` items, the empty-file SHA-256, exact pre-delete stat
identity, receipt, and an independent 8/8 post-delete lstat source. The queue
check is recorded only in the receipt summary; no raw queue snapshot exists.
Logical bytes freed are zero; allocated bytes and quota delta are unknown and
stored as `NULL`. Queries 15–16 retrieve the path-level state and receipt.
