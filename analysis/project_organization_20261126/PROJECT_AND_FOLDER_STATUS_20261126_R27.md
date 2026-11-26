# Project and folder status — R27

**Captured:** 2026-11-26 UTC  
**Rockfish scope:** only `/data/pclancy3/yi`, `/scratch4/pclancy3/yi`, and `/scratch16/pclancy3/yi`. No other users' directory trees were scanned.  
**Purpose:** navigation and progress snapshot; this is not a declaration that every payload is provenance-closed or safe to delete.

## Canonical layout

```text
Rockfish personal roots
├── /data/pclancy3/yi/                  source inputs and datasets
├── /scratch4/pclancy3/yi/              working outputs and intermediate campaigns
└── /scratch16/pclancy3/yi/             long-lived project organization
    ├── MigrationBench/                 sole canonical MigrationBench Git checkout
    ├── MigrationBench_data_archive/    MB source snapshots, inputs, runs, models, MD/NEB evidence
    ├── X-FORCE/                        sole canonical X-FORCE / DUAL-X Git checkout
    ├── X-FORCE_data_archive/           DUAL-X inputs, runs, models, receipts and evidence
    └── 218 other direct-child directories, grouped below
```

The Git repositories hold code, compact schemas/manifests, SQL query interfaces and paper-facing evidence indexes. Large model weights, trajectories, raw outputs and immutable database snapshots stay in the separate data archives. Each archived object should have a stable path, hash, source/run linkage, retention status and an SQL lookup when it is part of a registered evidence family.

## Direct-child inventory

The latest scoped first-level inventory records 17 directories and 3 files under `/data/pclancy3/yi`, 3 directories under `/scratch4/pclancy3/yi`, and 222 directories plus 106 files under `/scratch16/pclancy3/yi`. These are direct-child counts; they do not imply a complete recursive file-to-result map or a quota measurement.

The 222 Scratch16 directories comprise the four canonical repository/archive roots and 218 project/campaign directories. The current R21 TSV groups those 218 as follows:

| Project/folder family | Directories | Organization priority and current boundary |
|---|---:|---|
| MLIP training and MD campaigns | 86 | Shared model/training evidence used by several studies. Most campaigns still need run/checkpoint/claim-level mapping. |
| Lmax1 diagnostic campaigns | 36 | Failure/retry evidence is queryable; there are 194 indexed rows, 50 explicit source-to-retry/perturbation links, 10 missing retry variants and 90 pending Tier-2 gates. No measured active-learning gain is established. |
| MigrationBench Revision 1 model evaluations | 32 | Paper-critical model and matrix evidence; the registered model set is not a complete measured model-by-task matrix. |
| PolymerNetwork / generative projects | 25 | Separate study family; preserve its own authority and outcome labels. |
| MigrationBench NEB / DFT campaigns | 23 | Paper-critical paths and reference calculations. QE force residual/SCF-warning results remain provisional where recorded. |
| Shared tools and HPC environment | 6 | Infrastructure and utilities; not scientific evidence by folder name alone. |
| Project governance / archives / smoke tests | 4 | Operational records and smoke evidence. |
| Cross-element training and replicas | 4 | Shared model evidence; 2,997 intermediate checkpoint payloads remain on hold pending retention policy. |
| FLARE | 1 | Separate project area. |
| X-FORCE / DUAL-X experiments | 1 | Legacy experiment root; archive coverage is not complete enough to retire it. |

The family labels come from the R21 inventory and are navigation aids, sometimes based on names or prior folder notes. They do not prove that every child has a verified run, model, paper claim or archive destination. The [full 328-entry direct-child index](SCRATCH16_FOLDER_CLASSIFICATION_20261126_R21.tsv) and [directory-only index](SCRATCH16_DIRECTORIES_CLASSIFICATION_20261126_R21.tsv) preserve the individual paths and notes.

## Research flow and reuse boundary

```text
shared training inputs/checkpoints
    → MigrationBench model × task evaluation
    → run/attempt and raw output
    → derived datum and figure/table cell
    → submitted manuscript claim and release commit
    → explicitly selected DUAL-X training/diagnostic signal
    → held-out evaluation and measured gain
```

MigrationBench and DUAL-X can reuse model, failure and diagnostic evidence through pinned paths and SQL views. A diagnostic is not evidence of benefit by itself: reuse needs an explicit selection decision, source run and checkpoint identity, baseline, held-out evaluation, and a measured gain definition. The current database is a federated index over pinned SQLite snapshots, not one physically merged database and not a completed model-performance matrix.

## Repository and SQL state

- At the PR #40 baseline, MigrationBench GitHub `main`, the canonical local checkout, and Scratch16 `MigrationBench` were at `03084e55302e3a0f8e63b08217ad71c7ba7aea93`. The subsequent documentation-only PR #41 sync aligned all three refs at `4258db69c2422f454c63f418518a49c4deb39748`. PRs #35–#41 are merged. Scratch16 has zero other local branch heads. Because Rockfish has no Git executable, the updates used exact pack/index transfers, changed tracked files and checksum verification; the receipts do not call this a `git pull`.
- X-FORCE GitHub `main` and Scratch16 `X-FORCE` are at `f1e16d65e7ea5f72a2daf0c847226ba6f67f4b06`; no open X-FORCE PR or additional Scratch16 local head was found.
- The MigrationBench bridge pins 11 sources. QCE01–QCE05 ran on Rockfish SQLite 3.26 with `query_only=true`; QCE05 returns 30 summary mappings: 29 present and one explicitly missing/excluded. It does not establish run-attempt identity or scientific acceptance.
- The Scratch16 directory hierarchy is now queryable in a separate, immutable SQLite sidecar: 5,288 directory rows through depth 3, zero scan errors, and the R21 direct-child family labels. QH01–QH05 provide snapshot metadata, first-level family summaries, the captured tree, a project subtree, and family counts. It remains separate from the 11-source federation because Rockfish SQLite 3.26 is already at the default attachment limit. The database passed `quick_check` and has zero foreign-key violations; it is metadata, not recursive storage usage or run/claim provenance.
- The Revision 1 claim index retains 639 claim rows. The R2 snapshot adds the Fig. 4 to R05 summary crosswalk without replacing immutable R1.
- Cross-element checkpoint, Lmax1 and R2 claim databases are versioned snapshots in `MigrationBench_data_archive`; each has a separate manifest/hash and validation receipt. The federated source registry is the entry point for lookup.

The PR41 Scratch16 sync receipt and QCE outputs are archived at:

`/scratch16/pclancy3/yi/MigrationBench_data_archive/storage_audits/migrationbench_pr41_final_sync_20261126/`

The preceding PR39–40 receipt is archived at:

`/scratch16/pclancy3/yi/MigrationBench_data_archive/storage_audits/migrationbench_pr39_pr40_final_sync_20261126/`

The hierarchy database, source inventory, manifest and SQL package are archived at:

`/scratch16/pclancy3/yi/MigrationBench_data_archive/project_hierarchy_snapshots/20261126/scratch16_depth3_hierarchy_r1/`

The PR36–38 receipt remains in its earlier versioned directory. See the [Revision 1 bridge](../revision1_evidence_bridge_20261126/README.md) for source IDs, query IDs and source roles.

## Cleanup and branch reconciliation

- The two canonical Scratch16 project checkouts now track only their GitHub `main` refs. PR feature refs were removed from GitHub after merge; the MB and X-FORCE canonical Scratch16 directories have no additional local head refs.
- The old local MigrationBench workspace and several attached worktrees are **not** safe to delete. They contain divergent commits, uncommitted edits, unmapped evidence, conflicting snapshots and unique model payloads. Two already-merged ancestor refs with no branch-only paths were removed; other refs/worktrees remain held until per-file mapping, archive proof and pointer repair are complete.
- The local IDPP subtree has 825 files whose relative paths, sizes and SHA-256 values match Scratch16 exactly; Scratch16 has 60 more files. The local release manifests and three scripts still point into the local mirror. Update those pointers additively and issue a duplicate receipt before considering local deletion.
- The 2,997 intermediate cross-element checkpoints remain held, not deleted. Their metadata is indexed, but payload hashes and the retention decision are still outstanding.
- The scoped root inventory does not show quota recovery. No cleanup amount should be described as measured quota savings unless Rockfish returns a valid quota reading.

## Remaining decisions and work

1. **R05 PDF authority:** locate the manifest-pinned PDF or compare recovery/reconstruction candidates before choosing the accepted source.
2. **Checkpoint retention:** decide the retention rule after the verified checkpoint manifest is ready; keep current holds meanwhile.
3. **Cross-project learning:** define which failure/diagnostic signals qualify for DUAL-X, its baseline and held-out comparison, and the metric that counts as gain.
4. **Scientific evidence:** reconcile the SHAP source/version mismatch and manuscript references; keep QE residual/SCF cases provisional and resolve the Fig. S2 caption/data discrepancy against the submitted paper source.
5. **Folder closure:** map the 218 campaign directories to source, model/checkpoint, run/attempt, result/figure/claim and hold state. The R21 grouping is only the navigation layer.
6. **Local workspace retirement:** repair the IDPP path references, reconcile SHAP and NEB snapshots, map model weights and every remaining local-only item, then review exact duplicate candidates individually.

No scientific conclusion was changed in this status map. Storage cleanup and project organization are still in progress.
