# Rockfish project and folder map

This bundle is the current project-level and direct-folder-level index for the user's three Rockfish roots only:

- `/data/pclancy3/yi`
- `/scratch4/pclancy3/yi`
- `/scratch16/pclancy3/yi`

It does not include scans of other users' directories. Start with the [R27 project and folder status](PROJECT_AND_FOLDER_STATUS_20261126_R27.md) for the latest repository heads, folder hierarchy, evidence/database status, cleanup progress, and remaining decisions. The [R26 project/folder map](CURRENT_PROJECT_AND_FOLDER_MAP_20261126_R26.md) and [R25 map](CURRENT_PROJECT_AND_FOLDER_MAP_20261126_R25.md) remain dated snapshots. The [local workspace reconciliation](LOCAL_WORKSPACE_RECONCILIATION_20261126.md) records path-level mapping needed before retiring that separate local workspace. The [directory-only R21 table](SCRATCH16_DIRECTORIES_CLASSIFICATION_20261126_R21.tsv) contains 222 direct child directories and their family assignment. The full [R21 Scratch16 direct-child index](SCRATCH16_FOLDER_CLASSIFICATION_20261126_R21.tsv) preserves all 328 direct children: those 222 directories plus 106 files.

## Intended Scratch16 organization

The four canonical project/archive roots are:

```text
/scratch16/pclancy3/yi/
  MigrationBench/               canonical Git repository, main
  MigrationBench_data_archive/  large campaign evidence, model/run payloads, snapshots, audits
  X-FORCE/                      canonical Git repository, main
  X-FORCE_data_archive/         DUAL-X/X-FORCE run payloads and evidence
```

The Git repositories hold code, schemas, compact manifests, query interfaces and paper-facing evidence indexes. Large raw trajectories, checkpoints, model files and immutable SQLite source snapshots belong in the separate data archives, with paths, hashes, retention state and source lineage discoverable through the SQL federation where registered.

## Current state and limits

- MigrationBench main is at `296ef562a5bc28151a824d9201c383766cfa116f`; PRs #36–#38 are merged. The read-only SQL bridge has 11 pinned sources. PR #37 added the Lmax1 sidecar (194 diagnostic rows, 50 explicit links, 10 missing retries, 90 pending Tier-2 gates), and PR #38 fixed the duplicate SQLite attachment so the full bridge runs on Rockfish SQLite 3.26.0. QCK02/QCK04, QLM01–QLM07 and QX01–QX05 passed remote `query_only=true` smoke queries.
- All 222 Scratch16 direct-child directories are classified. Classification describes folder purpose; it does not close artifact hashes, attempt/run lineage, manuscript claims or safe-retirement decisions. Many of the 218 noncanonical campaign directories still need those links or a written hold disposition.
- A separate local legacy workspace has uncommitted and untracked project evidence. Its six priority groups have now been mapped at path/size/hash levels where feasible; see the reconciliation report. It is not an interchangeable clean main checkout, and retirement remains gated on finishing the listed metadata mapping and any author retention choices.
- The cross-element SQL index contains 3,027 artifact metadata rows. The 2,997 intermediate checkpoints remain held pending a documented retention decision; this bundle does not authorize their deletion.
- This is a dated navigation snapshot, not a statement that the full project organization, evidence closure, manuscript reconciliation or storage cleanup is complete.

## Evidence archive

The merged PR #36–#38 sync receipt and read-only remote query outputs are archived under:

`/scratch16/pclancy3/yi/MigrationBench_data_archive/storage_audits/migrationbench_pr36_pr38_final_sync_20261126/`

The receipt records the exact merge SHA, remote refs, pack/index/file hashes, 11-source count and QCK02/QCK04, QLM01–QLM07, and QX01–QX05 query details. The canonical GitHub source is [MigrationBench](https://github.com/yicao-elina/MigrationBench) at the pinned main commit above; this folder remains a map and index, while experimental payloads remain in the personal Scratch16 data archive.
