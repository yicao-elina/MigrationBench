# Phase 2 — Rockfish `/data` and `/scratch4` inventory and cleanup

**Status:** Accessible user-owned subtrees have been inventoried and evidence-cleared cleanup batches have run. The whole `/data` and `/scratch4` mounts are **not fully inventoried**: the latest root census could not traverse 625 of 725 `/data` directories and 503 of 562 `/scratch4` directories; the `/scratch4/pclancy3` tree also has 98 unreadable directories. Most observed multi-terabyte data belongs to other owners. The scope boundary is explicit below.

## Scope and cleanup policy

The control plane is the additive `provenance_catalog.sqlite` catalog. Each
candidate is tied to a path, owner, exact pre-delete identity where captured,
reproduction or duplicate evidence, queue check, and outcome. Inputs, launch
scripts, Slurm scripts/logs, scientific outputs, manifests, and failure
evidence remain available. Temporary QE WFC state is removable when it is not
required by a downstream restart and the retained input/script chain supports
a fresh run. A fresh rerun is not represented as an exact continuation of an
unfinished historical state.

The `/data/pclancy3/for-victor` and `for-yuqi` directories received the same
owner, path, duplicate, and reproducibility review as the rest of the
user-owned tree. The user explicitly confirmed these former-collaborator
directories should follow the same cleanup pipeline as the rest of the
project storage. The completed batches removed verified duplicates. Unique
scripts, symlinks, and data received the same path-level owner, provenance,
reproducibility, and cleanup review as every other project directory; there is
no separate alumni retention rule.

## Completed cleanup batches

| Batch | Files | Logical bytes | Allocated bytes | Result |
|---|---:|---:|---:|---|
| `RF-S4-WSE2-WFC-CLEANUP-20261122` | 8 | 5,045,101,744 | 5,045,174,272 | Guarded WFC removal; queue overlap 0; paths verified missing. |
| `RF-DATA-CONDA-INDEX-CACHE-20261122` | 42 | 1,123,596,223 | 1,124,405,248 | Post-hoc `PARTIAL` ledger. Conda reported dry-run but removed its regenerable index cache; pre-delete hashes and contemporaneous queue evidence were not captured. |
| `RF-DATA-MHP-NEB-WFC-CLEANUP-20261122` | 490 | 26,235,975,200 | 26,250,444,800 | Guarded WFC removal. This includes ten 1.312 GB `wfc1.dat` files from the unfinished neutral MHP NEB; the fresh-start input and Slurm recipe, QE outputs, XML, charge density, and incomplete-run log were retained. |
| `RF-DATA-YI-CR-IN-GAP-MIX-CLEANUP-20261123` | 12 | 5,352,688,896 | 5,352,980,480 | Guarded removal of QE mixing-history files after registering three partial-run chains and reviewed fresh-start recipes. All-user queue overlap 0; each path/hash verified missing; `/data` quota fell by the exact allocated-byte total and file count fell by 12. |
| `RF-DATA-FOR-YUQI-PSEUDO-DUPLICATES-20261122` | 65 | 39,657,991 | 41,811,968 | Exact UPF duplicates removed; canonical pseudopotentials retained under `/home/ycao73/pseudo`. |
| `RF-DATA-FOR-VICTOR-FE-DUPLICATES-20261122` | 9 | 5,814,720 | 5,965,824 | Exact formation-energy duplicates removed; six empty directories removed; unique `Ef_new.py` and the `cyqe` symlink retained. |
| `RF-DATA-YI-QE-MIX-R04-CLEANUP-20261123` | 384 | 7,500,169,152 | 7,518,289,920 | Guarded deletion of three reviewed QE fresh-start relaxation groups after exact input/output/Slurm/dependency/recipe links were registered; 48 paths with a WorkDir mismatch remain held. |
| `RF-S4-YI-CHARGEDENSITY-CLEANUP-20261122` | 4 | 214,623,824 | 214,646,784 | Guarded deletion of reproducible charge-density save state; three incomplete/failed charge densities remain held. |
| `RF-LAMMPS-BUILD-DELETE-20261122-R01` | 1,876 | 4,052,888,613 | 4,090,057,216 | Guarded deletion of generated LAMMPS build files; 219 empty directories including the old build root were removed. |
| `RF-DATA-CONDA-PKG-LIBSTDCXX-UNUSED-20261122-R01` | 90 | 14,223,129 | 14,785,024 | Guarded removal of 88 files and two symlinks from one unused package tree; 15 directories were also removed. Complete path-entry allocation estimate: 14,792,704 bytes. |
| `RF-SCRATCH16-YI-REV1-574-RESTART-CLEANUP-20261122-V1` | 574 | 1,536,705,954,176 | — | Historical exact-file cleanup imported with a TSV manifest hash; restart payload hashes and allocated-byte total were not recorded. |
| `RF-SCRATCH16-YI-REV1-QE-RESTART-CLEANUP-20261123` | 4,298 | 1,036,297,176,856 | — | Exact revision1 WFC/mix restart cleanup; paths were verified absent. Inputs, scripts, logs, XML, charge density, and fresh-start recipes remain; allocated-byte total is unavailable. |
| `RF-YI-QE-RESTART-CLEANUP-20261123` | 7,581 | 1,225,323,826,928 | 1,225,535,520,768 | Exact restart receipt with pre-delete `stat` allocation and post-delete verification; no personal quota-after read. |
| `RF-YI-QE-ZERO-MIX-CLEANUP-20261124` | 8 | 0 | — | Eight zero-byte `.mixN` placeholders removed; separate direct lstat check found all eight absent. Queue evidence is summary-only; allocated bytes and quota delta were not measured. |

The cleanup receipt table records all fourteen batches: thirteen are
`COMPLETED` and the Conda index-cache row remains `PARTIAL`. Evidence detail
differs by batch; historical restart rows with no payload hashes or allocation
measurements remain explicitly limited. The Conda index-cache row is marked
`PARTIAL` to preserve
its missing-hash and missing-queue limitations. The original ten `/data` and
`/scratch4` batches account for
2,980 leaf paths, 49,584,739,492 logical bytes, and 49,658,569,216 allocated
bytes across `/data` and `/scratch4`, including path-entry directory allocation;
this excludes the later `/scratch16` batches and the cross-root restart batch.
The eight-file zero-byte follow-up adds no logical bytes.
The removed Conda tree had 90 guarded file/symlink receipt rows and 15
verified-missing directories.

For LAMMPS, the 5,689-file KokkosKernels dependency was renamed to a sibling
directory and verified by full path/metadata/hash comparison. Its original
root inode was preserved; only the root directory `mtime` changed as expected
from rename. The remaining 1,876 build candidates passed a fresh full-hash,
identity, dependency-tree, and all-user queue preflight (zero overlaps), then
were individually verified missing. The `/data` group quota decreased by
4,089,774,080 bytes against 4,090,057,216 bytes of pre-delete allocated-byte
estimates; the 283,136-byte difference is retained as a quota accounting
variance. No rebuild job was submitted.

The Conda package-cache audit checked 20 visible prefixes and 2,190
`conda-meta` records. Its only unreferenced package tree was
`libstdcxx-14.2.0-h8f9b012_2`; every file and symlink had link count one. The
exact 105-entry manifest passed the live all-user Slurm workdir and Conda
process guard, and the pinned conda-forge archive URL/SHA-256 plus cache
recreation script are registered under `HPC_ENVIRONMENTS`. The guard verified
all 90 leaves and 15 directories missing. The `/data` user/group quota
decreased by 14,745,600 bytes and the filesystem entry count fell by 105. The
47,104-byte difference from path-entry allocation is retained as quota
accounting variance.

## Inventories and remaining large files

### `/data/pclancy3`

- The complete owner-filtered UID 3901 walk recorded 703,560 regular-file
  paths and 23,180 symlinks across `/data/pclancy3`; unique-inode allocation
  was 316,275,839,488 bytes at the R01 snapshot. Included roots are `yi`,
  `ycao73_home`, `for-victor`, `for-yuqi`, and `MHP_defect`. Later cleanup
  receipts—not an overwritten scan—record changes after that snapshot.
- The R04/R03 composite reconciles 777,719 current paths in
  `/data/pclancy3/yi`. It records 31 files over 500 MiB / 59.142 GB (19 outside
  `.conda`, 12 inside). The earlier 12-file QE mixing cleanup is verified;
  those removed paths are excluded from the R04 current set. `.conda` path
  coverage is complete against the R01 baseline, R02/R03 observations, and
  exact receipts, though owner/UID/GID freshness is partial because it was not
  fully restatted in R04. The remaining large paths include
  band-projection XML, trajectories, collaborator datasets, a training
  checkpoint, Git packs, and installed environment libraries. These are
  outputs or source-bearing data, not size-only temporary candidates; keep
  them until claim/reproduction/owner evidence supports a narrower action.
- The three same-hash Conda `libcublasLt.so.11` copies are recorded in an exact
  duplicate group. Two paths are hard links to one inode; removing the cache
  name alone cannot reclaim its blocks, and the installed environment copies
  remain in use. No file was removed from this group.
- The MHP neutral NEB is incomplete: its retained `neb.out` stops during image
  10 SCF and its last complete force is above threshold. The 490-file receipt
  includes its ten large WFC state files. Inputs, launch script, QE outputs,
  XML and charge densities remain. The recipe is a fresh-start rerun in a new
  directory, not a claim that the failed historical NEB converged.

### `/scratch4/pclancy3`

- The complete-path metadata walk for the accessible group tree recorded
  1,540,972 path observations and 98 unreadable directories; its completion
  status is **PARTIAL**, and it does not represent the whole `/scratch4`
  filesystem.
- The snapshot has 1,380 files larger than 500 MB / 6.518 TB. Of those,
  710 WFC or mixing-restart files sum to 2.326 TB. Most large paths are owned
  by other users or `root`. Review lists classify every row as HOLD; no
  deletion is staged based on file size or directory names.
- Priority review groups are `leyao/Bi2O3` (907 WFC/mixing paths / 1.049 TB),
  `leyao/Al2O3` (409 / 556.5 GB), archive/tree pairs, and 896 large `.tmp`
  paths. Those need owner/consumer/reproduction checks and a fresh live queue
  check before any action. The root-owned `XChen150.tar.gz` is outside this
  account's deletion scope.
- The latest batch-specific all-user queue guard at 2026-11-22 23:42:35Z
  parsed 1,353 jobs with no unknown workdirs and no overlap with the Conda
  cache; the only jobs under `/scratch4/pclancy3` were `jliu379` pending in
  `/scratch4/pclancy3/mia/Si_001` and `mmarlan1` running in
  `/scratch4/pclancy3/madaline/gpaw_port`. Those paths remain untouched.
  The earlier filtered query at 23:11:59Z also showed `jliu379`
  pending in `/scratch4/pclancy3/mia/Si_001` and `mmarlan1` running in
  `/scratch4/pclancy3/madaline/gpaw_port`.
- The latest `/scratch4/pclancy3/yi` R02 is complete for that subtree: 2,878
  files, 597,580,272 logical bytes, 617,763,328 allocated bytes, no WFC or
  mixing-restart files, and no files larger than 100 MB. Three incomplete or
  failed QE charge densities remain preserved.

### Mount-wide quota and permission gaps

At 2026-11-22 23:43:42Z, `/data` group `pclancy3` used 14,941,272,342,528
bytes against a 10 TiB hard limit, with grace expired and 351,600,640 bytes
in doubt. The `/scratch4` group used 9,984,613,859,328 bytes against 10 TiB,
with 982,532,096 bytes in doubt. The latest `/data` user quota was
6,563,841,441,792 bytes for `ycao73`, with no configured limit. The Conda
cache batch's before/after user and group readings both decreased by
14,745,600 bytes and 105 file entries.

At the 2026-11-23 01:16 quota checkpoint, the visible `/data/pclancy3` tree
accounted for about 284,763,154,432 bytes, leaving about 6.279 TB of the
user's `/data` quota unmatched to accessible UID 3901 paths. The 04:10 QE
mixing cleanup reduced the user quota by a further 5.353 GB, but the partial
tree scan was not rerun to recalculate that unmatched balance. The mount walk
has 625 inaccessible
root directories and 187 additional permission failures before cancellation;
the partial traversal found no extra UID 3901 paths but is not a full-mount
negative result. `.snapshots` was checked and no GPFS snapshots were found.
Closing this gap requires a storage-admin UID usage report or broader
traversal access; unknown quota usage is not a deletion target.

The `/scratch4` group quota is shared across multiple owners, while the
current account's `/scratch4` user quota is only about 604 MB. The large group
inventory is therefore not a cleanup list for the user's files. The current
whole-mount `/scratch4` inventory is also incomplete.

## SQL catalog and Git reconciliation

- `provenance_catalog.sqlite` has migrations 001–016 applied. It models
  project/run/artifact/recipe/derivation/claim links, storage paths and
  snapshots, hardlinks and duplicate groups, Slurm job observations, cleanup
  receipts and quota brackets, explicit Git merge events, and a normalized
  many-to-many duplicate-evidence bridge. Migration 016 adds canonical
  artifact-alias redirects; migration 009 adds typed paper-figure
  records, figure-to-file/pointer links, source claim-status observations, and
  explicit claim-to-figure relations.
- The final `PRAGMA integrity_check` returned `ok`; `foreign_key_check` returned
  zero violations after the figure/claim import, QE run and cleanup lineage,
  quota brackets, queue snapshot, and updated deliverable registration.
- The WSe₂ receipt was normalized to a per-batch `COMPLETED` row. The Conda
  index-cache event was normalized to `PARTIAL`, with evidence and gaps
  described above rather than upgraded to a fully guarded deletion.
- Isolated local Git candidates are clean and unpushed: MigrationBench `main`
  fast-forwarded from `715b2b0` to `817acf2`, with `dev` already contained;
  `md_path_search` `main` locally merged `hpc` at
  `79673bb4013b141bbe974cc2d74e9e2dd8612af1`. These three observations are
  stored in `git_merge_observation`. The existing dirty `md_path_search`
  worktree and its unique local files were not overwritten. No GitHub push
  occurred.
- The isolated Git candidate refs are now linked to their repository snapshots:
  eight ref-context rows, two origin remote observations, and five branch-pair
  ancestry rows record `dev` as contained by MigrationBench `main` and `hpc`
  as contained by the local `md_path_search` merge candidate. The observations
  are local snapshot facts; current remote tips were not rechecked or pushed.
- The 2026-11-23 local Git review found unified MigrationBench `main` clean at
  `269ab77`; the public-main snapshot, `e551a79`, old local `master`, and the
  reviewed local evidence branches are all ancestors, so no local mainline
  branch remains to merge. The old `master` worktree remains separate. Updates
  to Rockfish project refs remain blocked by missing objects, dirty source
  worktrees, or the `/data` quota overage; no ref or GitHub push was made.
- All 79 catalog duplicate groups link to their immutable hash, inode, review,
  or cleanup-manifest sources (88 evidence links total). The Conda package
  cache has a documented package pin and recreation method, and its 90
  file/symlink locations plus the synthetic tree location are marked removed.

### Figure, paper-claim, and project provenance

- The catalog now contains **18 figure IDs**, combining the 17 core-registry
  charts and one plot-registry-only chart, with **246 figure-to-file or explicit
  pointer links**. The 2026-11-01 core and plot SQLite snapshots remain
  separate source records. The local 2026-10-28 archive contributes 75
  hash-identified files across figure exports, derived tables, manifests, and
  indexes. Missing source paths and command-only render methods remain typed
  as missing or unmaterialized pointers.
- The static 639 evidence matrix contributes its **nine exact FS639 claims**.
  All 19 matrix-listed file paths were found under
  `analysis/multimodel_639_screen_20261014/` and linked to hashed artifacts;
  the DFT resource-gate wording in FS639-008 remains a text pointer with no
  file. The matrix's original statuses and job/run strings are preserved in
  `paper_claim_source_observation`. The catalog's current scientific status is
  `PROVENANCE_PARTIAL` for source-marked `VERIFIED` rows because historical job
  IDs 31981392, 31981441, and 31986653 do not map to catalog workflow/job
  instances. No numeric job-only joins were invented.
- The source's exact status/action strings are preserved separately from the
  catalog assessment. Claim evidence links use `UNKNOWN` support status until
  the source-to-claim relation is independently reviewed. Only FS639-009 has
  an explicit figure-level relation (`RECONSTRUCTION_OF`) to the static 639
  figure; no panel mapping is inferred.
- The static 639 handoff is a separate `REVIEW` candidate at commit
  `c0aee66e06ace1119ae04aff4daf7b236e69e45b`, which is an ancestor of local
  unified `main` `e551a799b48c7f77ded703b43e7fb9f35e693bfa`. Its source manifest
  says `HANDOFF_READY_NOT_SYNCHRONIZED`; no Overleaf version or publication is
  asserted. The renderer script is present and hashed in the source tree even
  though the archive rebuild index only says command pointer.
- The DFT-MACE47 figure is a separate `REVIEW` handoff candidate at
  `5353cdec3c5a286e68f5f0c9e3ee8dfdcee00f72`, not an ancestor of the captured
  unified `main`. Its archive contains FIG3 and derived data, but not the C1–C6
  claim matrix. The candidate manifest and commit are recorded; those claims
  are not imported or joined to collaborator DFT-NEB results. The manifest's
  `PENDING_USER_HANDOFF` text is a handoff state, not an Overleaf version; the
  catalog's `overleaf_version` is blank.
- MD FIG-004's A–G matrix rows describe panel contents and evidence gates; they
  are retained as figure assets, not fabricated as seven manuscript claims.
  Canva chart rows remain historical registry references with missing original
  worktree sources and no imported claims/release. The DFT-NEB source still has
  no publication links or result evidence-artifact bindings. The current
  catalog now includes a separate local DUAL-X Safe Core v6 evidence chain;
  it is not joined to the older Rockfish X-FORCE results without an exact
  source relationship.

## Remaining work

1. Review the imported static-639 claims against run-level scheduler/source
   records. The local DUAL-X Safe Core v6 package now has an exact project,
   run, recipe, figure, and claim map; its original inference invocation and
   external checkpoints remain unresolved. Recover the DFT-MACE47 C1–C6
   source matrix from its unmerged candidate bundle before importing claims.
2. Continue the 31-file `/data/pclancy3/yi` claim and reproduction review;
   do not remove result-bearing XML, trajectories, datasets, checkpoints,
   repository objects, or installed libraries without a precise evidence basis.
3. Obtain the missing `/data` UID accounting or traversal permission so the
   6.279 TB user-quota discrepancy can be resolved.
4. For other-owned `/scratch4/pclancy3` data, continue metadata/provenance
   review only; secure owner/admin authority and check the live queue before
   any deletion. The 98 unreadable paths and the rest of `/scratch4` remain
   outside the completed inventory boundary.
5. Finish normalizing changed audit scripts, reports, and sources in SQL;
   run final integrity/foreign-key checks and commit the local audit changes
   on unified `main`. No GitHub push is authorized or recorded.

## Fresh Rockfish checkpoint — 2026-11-23 01:16 UTC

The configured `rockfish` SSH alias is currently usable as `ycao73` (UID 3901,
primary GID 3001). The read-only immediate-child permission probe found 727
entries at `/data` (725 directories: 100 traversable and 625 denied) and 564
entries at `/scratch4` (562 directories: 59 traversable and 503 denied). The
`/data/pclancy3` and `/scratch4/pclancy3` roots themselves are readable. This
does not grant safe traversal of other owners' private directories; fresh
project-root metadata walks are stored as new snapshots rather than replacing
the 2026-11-22 inventories.
An owner-only immediate-child probe found no UID 3901 directories directly
under either mount; the sole project GID 3001 child under each mount is the
corresponding `pclancy3` root. The other 726 `/data` and 563 `/scratch4`
immediate entries belong to other UID/GID identities.

At 2026-11-23 01:16:09 UTC, GPFS reported user `ycao73` `/data` usage of
6,563,841,441,792 bytes across 658,813 files, with no configured user limit;
group `pclancy3` `/data` usage of 14,941,272,342,528 bytes across 1,100,237
files against a 10 TiB limit (grace expired); and group `/scratch4` usage of
9,984,613,859,328 bytes across 1,456,725 files against 10 TiB. These are
shared quota counters, not byte totals for one inventoried subtree. Their raw
capture and normalized rows are registered in SQLite.

At the 2026-11-23 01:42 UTC catalog snapshot, 498 WFC-named artifacts were
classified as REPRODUCIBLE_INTERMEDIATE and recorded REMOVED. Sixty other
catalog paths contained the token wfc but were UNCLASSIFIED; they were not
treated as QE restart candidates from a name match alone. Subsequent guarded
cleanup batches registered exact deletion outcomes for 574 large files and
4,298 additional revision1 restart files; all corresponding cleanup rows are
now DELETE/VERIFIED.

The 2026-11-23 01:42 UTC direct refresh is historical. It reported 3,817
basenames containing wfc and 481 containing mix in revision1 runs. The
2026-11-24 refresh and post-delete scan supersede it: the exact restart
inventory is empty in revision1_migrationbench_runs, and no nonempty exact QE
WFC/mixing restart remains in accessible personal roots. The 3,715 broad
WFC/mix token matches outside that revision1 tree are projwfc/PDOS and
software/library names, not exact restart payloads, and were retained.

The fresh all-user queue capture contained 1,295 jobs (319 running, 976
pending), all with absolute WorkDirs. It showed 116 WorkDirs under `/data` and
507 under `/scratch4`, with no jobs under `/data/pclancy3`; two jobs pointed
inside `/scratch4/pclancy3`, including one running job in
`/scratch4/pclancy3/madaline/gpaw_port`. There were 13 `ycao73` jobs, all
pending in `/home` or `/scratch16`; per-batch cleanup still requires a live
queue guard immediately before execution. Queue and permission captures are
registered by hash, and the quota refresh was imported into `quota_observation`.
At 2026-11-23 01:31:02 UTC, the narrower project-group queue refresh showed
only one live job inside `/scratch4/pclancy3`: running job 33224712 in
`/scratch4/pclancy3/madaline/gpaw_port`. Three ycao73/dsharm23 jobs were
pending under `/scratch16`; no other `/scratch4/pclancy3` workdir appeared.
That newer queue capture is also registered, and the running path remains out
of scope for deletion.

## Fresh partial metadata refreshes — 2026-11-23

The fresh `/data/pclancy3` stream is a separate, explicitly **PARTIAL** R02
snapshot, not a replacement for the earlier complete owner-filtered baseline.
It contains 521,626 regular files, 59,167 directories, and 17,523 symlinks
(598,316 observations total), with 265,480,498,203 logical bytes and
275,523,401,216 path-summed allocated bytes among emitted files. The stream
ended inside `/data/pclancy3/yi/.conda/envs/deepmd2/.../minicompat.py` after
SSH exited 255; it has no scan-finished marker. The first four selected roots
(`MHP_defect`, `for-victor`, `for-yuqi`, and `ycao73_home`) were traversed
before `yi` by scanner control flow, but their completion is inferred rather
than explicitly checkpointed. Both former-collaborator roots were included in
the same cleanup pipeline; neither emitted a `.git` marker. No new deletion
was made from this partial refresh. The raw stream, summary, and scanner are
preserved under `phase2_data/data_refresh_20261123/` and registered as
separate deliverables. Do not treat paths absent from this stream as absent
from `/data/pclancy3`.

The fresh `/scratch4/pclancy3` R02 is also **PARTIAL** and is only a lower
bound: 829,854 files, 86,036 directories, 8,612 symlinks, 98 permission
errors, and 3,044,686,008,334 logical bytes were emitted before the scanner
stalled in a third-party Python package tree and its SSH session disconnected.
The 98 unreadable paths match the earlier R01 snapshot. All 189 emitted files
larger than 100 MB match R01 metadata exactly; only 19 above 500 MB were
emitted, totaling 2.983 TB, principally archives owned by other users or
`root`. The partial refresh found no confirmed QE WFC or mixing-restart
cleanup candidate; 241 small filename matches were source/module false
positives. Six pairs totaling 2.065 GB are metadata-only duplicate indicators
and remain unverified. No deletion was made from R02. The earlier R01 review
queues for other owners' Bi₂O₃/Al₂O₃ WFC pools, archives, and `.tmp` files
remain review queues and are not authorized by size or name alone. The new
ledger, report, summaries, scripts, and candidate tables are under
`phase2_scratch4/scratch4_refresh_20261123/`.

The later `pclancy3` account queue capture at 2026-11-23
02:10:28 UTC shows one running job, 33224712, in
`/scratch4/pclancy3/madaline/gpaw_port`; that path remains held. Pending jobs
in this capture are under `/scratch16`. Git metadata was manually summarized
for 16 observed in-tree `.git` directories, but status could not run because
`git` is unavailable in the remote login PATH, and 20 `.git` files were not
followed. Do not label those repositories clean or stale based on this
refresh. The newer capture and partial scan artifacts have been imported or
registered by hash in the catalog.

At 2026-11-23 02:39:59 UTC, the latest `/data` group quota read was
14,591,086,272 KiB against a 10,737,418,240 KiB hard limit, with grace
expired (3,853,668,032 KiB over the hard limit). `/scratch4` group usage was
9,750,599,472 KiB against the same limit. The raw capture is preserved and
imported; the `/data` overage is a shared group counter and is not localized
to this user's files.

The Rockfish Git ref preflight was blocked without changing any refs. Three
candidate updates were examined: DynamicSciKG `main` is checked out and its
two live commits are absent from that repository's object store; MigrationBench
`main` can fast-forward to reviewed candidate `817acf2` but its objects are
absent from the target repo; `md_path_search` has reviewed merge candidate
`79673bb` but its objects are likewise absent. Since importing objects would
write to `/data` while the group hard limit is exceeded and grace expired, no
write was attempted. EDQUOT was not tested. Dirty and unique worktree files
were left untouched; no push occurred. The exact preflight/postcheck and
quota capture are registered. Retry these local ref updates only after a
fresh quota read shows usable headroom; DynamicSciKG also needs a safe
off-checkout ref-only path or separate worktree decision.

## QE mixing cleanup execution — 2026-11-23 04:10 UTC

The three Cr-in-gap QE runs, their Slurm jobs, input and submission scripts,
PP input/output, partial PW logs and XML, QE executable hashes, pseudopotential
hashes, and reviewed PW-only fresh-start recipes are registered in
`provenance_catalog.sqlite`. All three historical runs remain classified as
`FAILURE_MODE`: each PW relaxation hit `Maximum CPU time exceeded`, and each
PP step failed with `wrong nx`/`MPI_Abort`, despite `sacct COMPLETED 0:0`.
Fresh-start recipes are reviewed but have not been submitted or run; the
original 250,000-second limit remains, so a new run may again stop before
convergence. The deletion removes the exact continuation state of those
partial historical runs; only new fresh-start reruns are represented by the
recipe. The three runs are linked to campaign `MBREG-CR-IN-GAP-RELAX-202507`;
no specific paper figure or claim relation was established for these runs, so
the catalog does not invent one.

After the exact 12 paths, sizes, owner/mode/link/inode/mtime identities, hashes,
and retained artifact links were staged as `EVIDENCE_READY`, the guarded
executor queried all-user `squeue` on Rockfish, parsed 1,385 workdirs with zero
candidate overlap, and repeated the query and full SHA-256/identity checks
immediately before each unlink. The first attempt failed before queue parsing
because the login-node Python 3.6.8 lacks `subprocess.run(text=...)`; its
failure receipt is retained, and the guarded re-run used Python-3.6-compatible
arguments. The successful receipt records 12 paths verified missing,
5,352,688,896 logical bytes, and 5,352,980,480 allocated bytes. GPFS `/data`
group usage fell by exactly 5,352,980,480 bytes and file count fell from
1,100,237 to 1,100,225 between 04:01:47Z and 04:10:00Z. `/scratch4` usage did
not change. `/data` group usage is still 14,935,919,362,048 bytes, or
3,940,803,084,288 bytes above its 10 TiB hard limit; this batch did not resolve
the shared group overage or the mount-wide user-quota visibility gap. The
before/after captures, failed preflight receipt, successful all-user queue and
per-file deletion receipt, executor, and cleanup outcome are preserved and
linked to the cleanup batch in SQL. The 1,385-row preflight queue is a separate
`registry_source` and is linked through
`cleanup_execution_receipt.queue_snapshot_source_id`; the successful receipt
also retains each immediate pre-unlink queue recheck.

## DUAL-X project provenance crosswalk

The `DUALX` project row now has an evidence-backed local Safe Core v6 campaign
crosswalk; this checkout is outside Rockfish and is a separate storage root.
The local repository was read-only at branch `yicao/safe-core-m1-m6`, HEAD
`175039d3b92f42f1ab05c24b2a4ce59130371120`; the four untracked `.DS_Store`
paths are preserved in the Git snapshot. The `dualx-safe-core-m1-m6-v6` tag
peels to release commit `8448ed7eaa6e8e2ecce2d994477c98bccc150d71`. No refs,
repository files, or remote state were changed by this audit.

The catalog imports **135 DUAL-X artifacts**, four workflow runs, and four
recipe records. Analysis, figure rendering, and bundle validation recipes are
marked reproduced from the source clean-checkout record; the original
inference invocation is `INCOMPLETE`. All 119 files in the source hash
manifest were verified against SHA-256 and byte size. The crosswalk contains
seven unique figure IDs across 21 exported files, 54 figure-asset links, 18
source claims, 11 explicit claim-to-figure links, 38 claim-evidence links, and
127 derivation edges. The source claim status and allowed wording remain
preserved separately; catalog scientific status remains provenance-partial.

Three model checkpoints are represented by hashed external pointers because
their cache paths were not resolved. The clean-checkout reconstruction starts
from frozen inference outputs, so it is not represented as an exact rerun of
the original inference. This local Safe Core package is not joined to the
older `/data/pclancy3/yi/.../X-FORCE` result tree unless a specific run/data
relationship is established. The project record therefore supports retrieval
without collapsing distinct DUAL-X generations.

## Scoped R03 inventory refresh — 2026-11-23

### `/data/pclancy3/yi/.conda`

Four previously incomplete environment roots completed independent metadata-only
scans: `automatbench` (39,906 entries), `deepmd2` (39,534), `qe-dev`
(17,588), and `sklearn-env` (108,751); all emitted complete markers with zero
errors. Reconciliation joins the complete R01 baseline, the R02 partial
current scan, those four R03 roots, and the exact deletion receipts. The
current `.conda` path set is **689,599 paths**, with zero uncovered, unexpected,
or receipt-deleted paths still observed and no R02/R03 metadata mismatches.
All 148 paths removed by the two earlier cache cleanups are accounted for;
the R01 baseline predates those removals. This makes the `.conda` subtree
reconciled, but does not complete all of `/data/pclancy3`.

The reconciled `.conda` tree has 598,792 regular files totaling 45.717 GB
logical bytes; 75 files exceed 100 MB and 12 exceed 500 MiB. They are installed
environment libraries or package-cache payloads. All remain held: the audit
found no new unreferenced package tree, and the 574.6 MB `libcublasLt.so.11`
copies are hard-linked, so removing one name would not release the inode's
blocks. No files were removed in R03. Four complete environment snapshots,
their source ledgers/statuses, and the exact reconciliation report are
registered in SQL.

### Selected `/scratch4/pclancy3` subtrees

Eight isolated roots were rescanned, with one documented exception:

- `/scratch4/pclancy3/yi` is a complete 2,934-path snapshot with 2,875 UID
  3901 files totaling 382,958,312 bytes, no file above 100 MiB, and no exact
  QE WFC/mixing restart basename. Four paths missing since R01 are the exact
  charge-density files already in cleanup receipt
  `RF-S4-YI-CHARGEDENSITY-CLEANUP-20261122`. Four new paths form the
  `cluster_recipes/lammps-flare` tree; the Slurm rebuild script is already
  hash-bound in the LAMMPS build recipe and is retained.
- `leyao/Bi2O3`, `leyao/Al2O3`, `madaline/PNNL_Work`, `F/MD`,
  `maurice/connor_polymers_smd`, `scratch_archive_dsharm23/aaron`, and the
  non-venv portion of `scratch_archive_dsharm23/divya` completed independent
  scans. Their files are not UID 3901-owned. The original full Divya scan hit
  the 180-second limit; a bounded retry pruned only `divya/venv`, marked that
  exact excluded subtree in the ledger, and completed the rest. Nothing in the
  venv was deleted.

The selected-root candidate ledger contains **30,892** WFC/mixing-name
matches totaling **2.064 TB** logical size, all owned by other users; 2,358
exceed 100 MiB and 337 exceed 500 MiB. These are name-pattern review rows,
not confirmed stale restarts. No other-owner data was removed. Eight scoped
snapshots, exact queue evidence, root-by-root results and the partial Divya
attempt are recorded separately in SQL. The entire `/scratch4/pclancy3` and
`/scratch4` inventories remain incomplete: 17 of 24 top-level root path sets
still differ or lack full coverage, with 98 unreadable paths in R01. The
active job `33224712` in `/scratch4/pclancy3/madaline/gpaw_port` and pending
job `18063302` in `/scratch4/pclancy3/mia/Si_001` remain held.

### Latest quota and queue evidence

At **2026-11-23 05:02:48 UTC**, `/data` user `ycao73` used
6,558,488,461,312 bytes across 658,801 files. The shared `pclancy3` group
used 14,935,919,362,048 bytes against the 10 TiB limit—still
3,940,803,084,288 bytes over, with grace expired—and 1,100,225 files. The
`/scratch4` group used 9,984,613,859,328 bytes across 1,456,725 files.
These remain shared quota counters, not a per-path attribution. The latest
successful all-user queue snapshot is **04:36:22 UTC** with 1,407 parsed jobs,
including the two held `/scratch4/pclancy3` workdirs and none under
`/data/pclancy3`. A 05:01 UTC refresh attempt could not contact Slurm's DNS SRV
configuration; its failure is preserved, and no deletion followed that
attempt.

### Catalog structure and remaining coverage gaps

No core schema migration was needed: the existing model links project and
campaign, run, exact input/output/script artifacts, recipes, Slurm records,
Git bindings, derivations, paper claims/figures, inventory observations, and
cleanup/quota/queue receipts. The catalog now has 27 inventory snapshots (5
partial) and 4,978,990 path observations; the 12 R03 snapshots are kept at
their exact subtree roots. The read-only query cookbook now includes query 9
for missing run-to-recipe/input-output/Slurm/Git coverage and query 9b for
claims without explicit figure links.

The catalog currently contains 19 workflow runs and 19 recipes; 3 runs lack a
recipe, 12 lack Slurm records, and 14 lack run-to-Git bindings. Of 27 paper
claims, 12 have explicit claim-to-figure links. These remain visible coverage
gaps; filenames, nearby timestamps, and bare job IDs are not used to invent
relationships. The DDL cleanup guard does not guarantee a fresh-start recipe
was run or that every required recipe input and scheduler/Git binding is
present, so each deletion still requires an exact per-item evidence review.

### Former-collaborator cleanup status reconciliation

The as-found review in `phase2_data/former_collab_roots_review_20261122.md`
remains frozen as pre-delete evidence. Its “No files have been deleted” line
describes that earlier capture. The subsequent guarded batches removed 65 exact
Yuqi UPF copies and nine exact Victor formation-energy copies; all 74 SQL
cleanup items are `VERIFIED`, both batches are `COMPLETED`, and the canonical
copies remain present. The complete 2026-11-23 handoff scan finds only the
empty Yuqi `pseudo` directory and Victor's unique `Ef_new.py` plus `cyqe`
symlink. The dated execution details, current retained paths, raw receipts,
and SQL reconciliation method are in
`phase2_data/former_collab_cleanup_execution_addendum_20261123.md`.

## Git state after local alignment review

Unified local MigrationBench `main` was clean at `269ab77` when the local
alignment review ran; all reviewed local evidence branches and old `master`
were ancestors. The old `master` worktree remains preserved because it still
contains unique local-only files and cloud placeholders. Rockfish refs remain
unchanged: missing commit objects, dirty checkouts, and the over-limit `/data`
group prevent safe object imports/ref updates. No GitHub push was made.

## Current scope clarification — 2026-11-23

The current user request renews the objective to organize and clean both
Rockfish mounts, `/data` and `/scratch4`, end to end. Work therefore continues
across every accessible project-owned subtree, with read-only mount-boundary
inventory elsewhere; a file is not eligible for cleanup merely because it is
visible or large. The selected current review includes `/data/pclancy3/yi`,
`/scratch4/pclancy3/yi`, `/data/pclancy3/ycao73_home`,
`/data/pclancy3/for-victor`, and `/data/pclancy3/for-yuqi`. The two former-
collaborator roots follow the same ownership, duplicate, provenance, queue,
and guarded-cleanup pipeline as the project tree.

Before that scope was narrowed, R04 performed read-only metadata scans under
`/scratch4/pclancy3/scratch_archive_cganley2/users` and 16 other inactive
Scratch4 roots. The cganley child-scan union exactly matched R01 at 379,462
paths; the 16-root union matched R01 at 158,447 paths. These entries belonged
to other UIDs, not UID 3901. The scans read no file payloads, computed no file
hashes, and deleted nothing. Four timed-out parent attempts and one targeted
31-path lstat sample are retained as explicitly PARTIAL records. Those earlier
scans remain historical evidence; the renewed request makes accessible parts
of both mounts current scope, while entries owned by other users remain
read-only until their ownership and authorized cleanup basis are established.

The broad all-user Slurm capture at 2026-11-23T06:04:28Z is retained as
historical safety evidence. The 06:30:05Z queue query was filtered remotely to
the two `yi` WorkDir prefixes and returned no matching job rows. Each cleanup
batch, including the new R04 batch below, has its own fresh all-user guard.
Quota captures scoped to user `ycao73` remain filesystem totals rather than
per-directory attribution.

For `/scratch4/pclancy3/yi`, the latest COMPLETE R03 snapshot records 2,875
files, 59 directories, 382,958,312 logical bytes, 403,117,056 allocated bytes,
and zero unreadable entries. It has no file over 100 MiB and no QE WFC/mixing
basename matches. No new cleanup candidate is supported by that snapshot; the
three previously held incomplete/failed charge densities remain held.

For `/data/pclancy3/yi`, the R04 non-`.conda` scan and R03 `.conda`
reconciliation now form a complete 777,719-path composite. The path set has
zero unexplained additions or absences; `.conda` owner/UID/GID fields are not
all freshly restatted, and the composite is not an atomic full-tree scan.
Regular files total 274,325,891,723 logical bytes and 287,631,953,408
allocated bytes. Before the R04 batch, the R04/R03 composite contained 1,976
QE-style `.mixN` paths totaling 16,737,075,072 logical bytes. The focused
432-path review split into 384 deleted files and 48 held paths. The batch
removed 7,500,169,152 logical / 7,518,289,920 allocated bytes; its exact run,
recipe, dependency, Slurm, queue, pre-delete hash, post-delete, and quota
evidence is recorded in SQL. It targeted three completed non-NEB fresh-start
relaxation groups. The 48 held paths are in
`raw-discovery-relax-cr`; its current directory did not match the job's
`sacct` WorkDir. The retained input, output, and restart-policy evidence stay
cataloged as provenance-partial.

After that batch, 1,592 exact `.mixN` paths remain, totaling 9,236,905,920
logical bytes. A read-only review excluding the 48 held paths already in the
batch manifest found 1,544 files / 8,313,215,328 logical / 8,342,470,656
allocated bytes in 18 output directories. No further group is deletion-ready:
14 are active restart-mode NEB trees, two single-image NEB jobs say
`from_scratch` but image 4 failed SCF convergence and the mixers may preserve
valuable continuation state, and two Cr-in-gap locations contain only empty
restart placeholders. The 392 empty files in the reviewed residual set free no
space. The two image-4 groups remain held pending deeper consumer and quality-
boundary review. Filenames alone were not used to infer obsolescence.

The 2026-11-23 refresh of the other three selected roots recorded
`ycao73_home` at 25,231 files / 1,268,081,467 logical bytes / 1,863,645,184
allocated bytes; the difference from R01 exactly matches the verified 1,876-file
LAMMPS build cleanup. `for-yuqi` has only its empty `pseudo` directory.
`for-victor` retains the unique `Ef_new.py` and working `cyqe` symlink; its
original QE input/Slurm chain remains provenance-partial. The 74 duplicate
Yuqi/Victor copies remain verified removed. Four `ycao73_home` Git packs and
three temporary Git object files were reviewed separately; the Git packs and
temporary objects remain held pending object/reachability review.

The scoped Rockfish Git review covered 25 repositories under `yi` and four
large-pack repositories under `ycao73_home`. Eighteen of the 25 `yi`
repositories have staged, unstaged, or untracked work; three of four home-root
repositories are also dirty. No remote refs were
fetched or branches changed. Several local topic branches descend from their
main branches, while `/data/pclancy3/ycao73_home/flare` is 110 commits behind
its cached upstream and has untracked files; cached refs may be stale. The
local DUAL-X `main` contains its reviewed `yicao` history and latest Walsh
provenance merge at `5f22baf`, is 18 commits ahead of `origin/main`, and has
190 pre-existing untracked files. No GitHub push was made.

The separate DUAL-X local `main` was fast-forwarded through its 16-commit
`yicao` history, then received the latest verified Walsh Zenodo provenance
commit. The README conflict was resolved by retaining both the dataset's
verified source/license and the fine-tuning train/evaluation overlap caveat.
The resulting local `main` is `5f22baf`; the 190 pre-existing untracked paths
remain untouched. No push was made.

## QE mixing-history cleanup and current residual review — 2026-11-23

Three exact fresh-start relaxation groups were linked to three workflow runs,
three reviewed rerun recipes, their input/output artifacts, submission and QE
scripts, pseudopotentials, QE executable identities, Slurm job instances, and
the run manifest. The recipes are `REVIEWED`, not `REPRODUCED`; scientific
acceptance for the historical runs remains `PROVENANCE_PARTIAL`. The cleanup
removes regenerable SCF mixing history only; it does not assert that the old
run converged. Exact launch commands and dependency identities are in
`phase2_data/yi_mix_repro_review_20261123/qe_run_reproduction_inputs_manifest_20261123.json`.

The batch's selected-root all-user queue snapshot parsed 1,378 jobs with zero
matching WorkDirs and zero unknown WorkDirs. The executor repeated the queue
query and exact SHA-256 plus inode/device/owner/mode/link/size/allocation/mtime
checks immediately before each unlink. Independent postverification checked
all 432 paths: 384 authorized paths are missing, 48 held paths remain present
with their pre-delete hashes, and there are zero mismatches or unexpectedly
present delete paths. SQL marks the batch `COMPLETED`; its receipt records 384
files, 7,500,169,152 logical bytes, and 7,518,289,920 allocated bytes. SQL
integrity and foreign-key checks passed.

GPFS `/data` group use fell from 14,935,919,362,048 to
14,928,401,072,128 bytes and the group file count fell from 1,100,225 to
1,099,841 between 08:24:13Z and 08:26:47Z. This exact 7,518,289,920-byte
decrease matches the cleanup's allocated bytes. User `ycao73` `/data` file
count fell from 658,801 to 658,417; `/scratch4` group quota did not change.
`/data` remains 3,933,284,794,368 bytes above the shared 10 TiB group hard
limit, so this batch neither resolves the overage nor attributes it to one
owner or subtree. Raw quota captures, UTC-normalized copies, queue evidence,
execution receipt, postverification, and the normalizer are preserved.

The current `/data/pclancy3/yi` review found no additional large-file deletion
that met the evidence threshold. Its 31 files above 500 MiB total 59.142 GB
and remain held as scientific outputs, model or dataset artifacts, installed
environment libraries, or Git pack data. On `/scratch4/pclancy3/yi`, no file
exceeds 100 MiB and no WFC/mixing basename remains; three incomplete/failed
charge densities remain held. The 74 verified exact duplicate copies in
`for-yuqi` and `for-victor` remain removed. `for-yuqi` has only an empty
`pseudo` directory; `for-victor` retains its unique `Ef_new.py` and working
`cyqe` symlink.

The refreshed Git review corrected the `yi` dirty-worktree count from 16 to 18
of 25; three of four `ycao73_home` repositories are also dirty. Dirty Rockfish
worktrees and cached refs were left unchanged. Isolated local candidates now
track live `main` for LLM4Chem and DynamicSciKG; the neuromorphic-filament
candidate promotes its verified live `hpc` tip to local `main`. The existing
md_path_search candidate merges live `main` + `hpc`, and the MigrationBench
candidate `main` already contains live `dev`. These are local audit candidates;
no GitHub refs were pushed. See
`phase2_data/GIT_LIVE_MAINLINE_CANDIDATE_DECISIONS_20261123.md` and its
candidate receipts. A fresh full matrix covers all 29 Git repositories under
`yi` and `ycao73_home`: 27 have usable cached-canonical ancestry comparisons,
two lack a resolvable canonical remote; 13 cached defaults differ from live
defaults and all 13 live target objects are absent locally. No further local
merge is safe from available objects. The unique cached MACE commit remains
held because its current live `develop` tip is unavailable in the local object
store. See `phase2_data/full_git_reconciliation_r04_20261123/`. At the 2026-11-23 snapshot, unified MigrationBench `main` was unpushed;
DUAL-X `main` was `5f22baf`, 18 commits ahead of cached `origin/main`, and was
also unpushed.

After this batch's run import and the MHP_defect R04 scan import, the catalog contains 22 workflow runs, 22
recipes, 11 Slurm job instances, and 5 run-to-Git bindings. Three workflows
still lack recipes, 12 lack Slurm job records, and 17 lack Git bindings. It has
100 inventory snapshots (11 partial) and 6,773,386 path observations. These
remaining gaps are explicit catalog coverage limits, not inferred away from
directory names or job-number matches.

`/data/pclancy3/MHP_defect` now has a complete 193-path R04 snapshot
reconciled to R01 and all 490 verified WFC cleanup rows. It contains no
remaining WFC/mixing restart, common temporary-suffix, zero-byte, or repeated-
inode candidates. Its two largest results are the distinct 163,864,820-byte
charge-density `.cube` outputs; both remain. The 08:54:44Z all-user queue
capture parsed 1,544 jobs with no MHP or original `/scratch16` WorkDir and no
unknown rows.

## Whole-mount completion boundary

The selected project-owned subtrees and historical group scans do not amount
to a complete `/data` or `/scratch4` inventory. The 2026-11-23 access probe
found 625 of 725 immediate `/data` directories and 503 of 562 immediate
`/scratch4` directories not traversable by this account; the broad
`/scratch4/pclancy3` inventory also contains 98 unreadable directories. These
are access boundaries, not evidence of stale content. The accessible `/data`
outer-root census is complete as a bounded metadata scan: 99 roots were
attempted, 66 complete and 33 partial; it observed 106,336 paths / 9.592 TB
logical as lower bounds. The SQL catalog adds root-boundary rows for 625
non-traversable directories, the separate `/data/pclancy3` boundary, and two
top-level files; the resulting 106,965-row snapshot is explicitly `PARTIAL`.
Other-owner paths remain read-only for cleanup without owner clearance.
The bounded `/scratch4` outer-root scan attempted all 58 accessible roots
outside `/scratch4/pclancy3`: 18 were empty, and all 40 non-empty roots were
attempted. Fifteen completed, 11 yielded partial output, and 14 timed out.
The 26 roots with emitted file totals observed 309,515 files / 32.790 TB
logical bytes as a lower bound; all belong to other-owner scope. R05/R06
reports and per-root receipts are linked to `RF-SCRATCH4-FS` as aggregate
scope evidence, not a complete per-file snapshot. A separate
UID 3901 check attempted all 58 roots, with 34 complete, 15 partial, and nine
timeouts; no match was emitted, while incomplete roots remain unknown. The
direct mount boundary still has 503 non-traversable directories and the broad
`/scratch4/pclancy3` tree has 98 unreadable paths. Two active/pending jobs
under `/scratch4/pclancy3` remain held. A fresh 09:02:37Z boundary probe
reported zero non-permission errors; a previous probe's single non-permission
error did not recur. Both mounts remain incomplete until inaccessible paths
can be inspected or explicitly handed off.

## Update — 2026-11-24 restart residuals and paper-candidate SQL import

The 2026-11-24 metadata-only refresh traversed 114,907 directories across the
accessible `/scratch16/pclancy3/yi` and `/data/pclancy3/yi` roots without
traversal errors. `revision1_migrationbench_runs` has zero basenames containing
`wfc` or `mix`; no nonempty exact QE WFC/mixing restart remained in those
accessible personal roots. Eight empty Cr-in-gap `.mix1`–`.mix4` placeholders
were removed after a fresh all-user queue check and exact metadata checks;
post-delete verification found all eight absent (zero logical bytes freed).
Broad WFC/mix-token names were retained where they resolve to `projwfc`/PDOS or
software/library files. The whole `/data` and `/scratch4` mount scopes remain
incomplete and personal quota-after is unavailable.

The SQL cleanup ledger records the 4,298-file user-directed revision1 batch
(3,817 WFC and 481 mixing; 1,036,297,176,856 pre-delete logical bytes) as
`DELETE/VERIFIED`, alongside the earlier 574-file / 1,536,705,954,176-byte
batch. The coordinate-supported batch records 175 restart entries /
155,250,907,476 bytes from seven 81-atom NEB roots as `DELETE/VERIFIED`;
each eligible root retained a complete latest image path and parseable XML
coordinates. This supersedes the earlier hold on those NEB restart files.

PR #10/#12/#13 evidence is now in `provenance_catalog.sqlite` as separate
`REVIEW` candidates. PR #10 contributes 13 claims, PR #12 contributes three
unselected figures and no selected claims, and PR #13 contributes six claims
and four figure records. All 1,336 PR #13 raw-evidence files were independently
re-hashed on Rockfish and registered as present archive locations, distinct
from their absence in both pinned Git trees. The submitted release remains
unchanged; `quick_check` returned `ok` and the transaction completed with
zero foreign-key violations. See
`phase2_schema/pr_candidate_release_catalog_queries_20261124.sql` for
read-only queries.

## 2026-11-24 manuscript audit and large asset organization

The supplied main manuscript, response-to-reviewers, SI, and clean LaTeX ZIP
were compared with the submitted Git release and latest available
Overleaf-export snapshot. The seven main figures match the submitted release
blobs, but the exact clean-TeX-to-PDF build provenance is not available. The
alignment audit records unresolved NEB force-convergence, SI Fig. S2
plot/caption, MD stopping-label, Table R2/S11, MPA0 lineage, and Fig. 6/7
regeneration discrepancies. This is an evidence review, not a manuscript
correction or scientific acceptance.

Five large assets totaling 167,458,328 bytes are in the organized Rockfish
archive at /scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_git_large_assets_20261124:
four model checkpoints and the CANVA-P03 SVG. Their archive manifest records
sizes, hashes, source locations, and Git blob identities when tracked. The
model loaders and asset manifests point to this archive; original /data model
sources remain in place. The four Git-tracked model checkpoint files and large SVG are removed from
the consolidated candidate's latest Git tree.

The exact QE restart scan is now empty in revision1_migrationbench_runs.
Nonempty WFC/mixing restart payloads were absent from accessible personal
roots; eight zero-byte Cr-in-gap mixing placeholders were removed. These
deletions do not claim full /data or /scratch4 coverage, and the fresh personal
quota-after reading remains unavailable.

## Update — 2026-11-24 MD evidence archive and final mix receipt

The organized Rockfish archive now contains a selective six-run grouped-A2 MD
bundle at
`/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_md_fig6_table_s8_20261124/`.
Its 49 payloads total 2,320,577,507 bytes: six trajectory segments, six source
smoke checkpoints, six final coordinate checkpoints, six per-seed model
weights, one shared 2,050-atom initial structure, and run manifests/status,
metrics, and continuation ledgers. Every payload passed an independent
destination size/SHA-256 check; the six final checkpoint steps match their run
status records. This is a selected subset of the 15-model Fig. 6 / Table S8 /
Table R7 cohort, not a full reproduction package. One continuation reached
1,000,000 steps; the other five stopped at scientific thresholds earlier.
Original source trees remain intact, and the copied data do not establish
scientific acceptance.

The eight empty Cr-in-gap `.mix1`–`.mix4` placeholders were initially cataloged
as `HOLD`, then deleted under a separate exact-path receipt after a fresh
all-user queue check and pre-delete metadata/hash verification. The new SQL
batch `RF-YI-QE-ZERO-MIX-CLEANUP-20261124` records all eight as
`DELETE/VERIFIED`, with zero logical/allocated bytes freed. The earlier `HOLD`
rows remain as a historical decision state. The receipt hash is
`b2fc5e7552aa33bd0d086ae4b68bb22c95e09392cf69585439405dcc827cac79`; queries
15–16 retrieve its path-level state and execution receipt.

The eight-file follow-up is cataloged as
`RF-YI-QE-ZERO-MIX-CLEANUP-20261124`; it preserves the original batch's
historical `HOLD` decisions and records eight later `DELETE/VERIFIED` items.
The receipt and the independent post-delete exact-path check (8 absent, 0
errors at 08:58:26Z) are separate SQL sources. The queue check exists only as
the original receipt's summary; its raw `squeue` snapshot was not retained.
Logical bytes freed were zero. The SQL receipt leaves allocated bytes and the
quota change `NULL` because neither was measured.

## 2026-11-24 SQLite catalog snapshot

The full `provenance_catalog.sqlite` snapshot is stored at
`/scratch16/pclancy3/yi/MigrationBench_data_archive/provenance_catalog_snapshots/20261124/`.
Its adjacent `DB_SNAPSHOT_MANIFEST.json` records the exact size and SHA-256,
successful `PRAGMA quick_check`, zero foreign-key violations, and the verified
Rockfish destination. The database is an external Rockfish artifact; the
repository carries its schema, importers, read-only queries, and archive path,
not a multi-gigabyte database copy.

A stale `provenance_catalog_20261124.sqlite.transfer2.partial` (1,966,440,448 bytes; SHA-256 `9b08146cac92151082dca72162bcc28ada19aba6e5b8c39d0a9161d8008f6d5f`; epoch-era mtime) was removed after confirming no active snapshot/rsync writer and verifying the final database copy. The final copy is 3,970,719,744 bytes, SHA-256 `2a46a3380ea291db3fc02dc0f148931eba40297ada670ab773ddcab8fe75d053`, mode 0600, with `quick_check=ok` and zero foreign-key violations. The partial-removal receipt is `phase2_data/provenance_catalog_snapshot_20261124/ROCKFISH_PARTIAL_PURGE_RECEIPT.json`.
