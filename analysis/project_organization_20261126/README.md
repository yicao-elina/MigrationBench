# Rockfish project and folder map

This bundle is the current project-level and direct-folder-level index for the user's three Rockfish roots only:

- `/data/pclancy3/yi`
- `/scratch4/pclancy3/yi`
- `/scratch16/pclancy3/yi`

It does not include scans of other users' directories. Start with [the current R25 project/folder map](CURRENT_PROJECT_AND_FOLDER_MAP_20261126_R25.md) for project relationships, root inventory, cleanup state, SQL/provenance boundaries, remaining work and consolidated author decisions. The [directory-only R21 table](SCRATCH16_DIRECTORIES_CLASSIFICATION_20261126_R21.tsv) contains the 222 direct child directories and their family assignment. The full [R21 Scratch16 direct-child index](SCRATCH16_FOLDER_CLASSIFICATION_20261126_R21.tsv) preserves all 328 direct children: those 222 directories plus 106 files.

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

- MigrationBench main is at `3c9f598ff96d3b52cffad4f1205efaae3f1a304d`; PR #35 added the tenth pinned read-only SQL source for cross-element training artifact metadata. Its CI passed, and the merged bridge plus QCK02/QCK04 remote query outputs were synced to the evidence archive.
- All 222 Scratch16 direct-child directories are classified. Classification describes folder purpose; it does not close artifact hashes, attempt/run lineage, manuscript claims or safe-retirement decisions. Many of the 218 noncanonical campaign directories still need those links or a written hold disposition.
- The existing local legacy workspace `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench` is dirty with uncommitted and untracked project evidence. Preserve it and map unique content to the canonical repository or verified archive before considering cleanup; it is not an interchangeable clean main checkout.
- The cross-element SQL index contains 3,027 artifact metadata rows. The 2,997 intermediate checkpoints remain held pending a documented retention decision; this bundle does not authorize their deletion.
- This is a dated navigation snapshot, not a statement that the full project organization, evidence closure, manuscript reconciliation or storage cleanup is complete.

## Evidence archive

The merged PR #35 sync receipt, R25 map copy, classification table, and read-only remote query outputs are archived under:

`/scratch16/pclancy3/yi/MigrationBench_data_archive/storage_audits/migrationbench_pr35_cross_element_20261126/`

The receipt records the exact merge SHA, remote refs, pack/index/file hashes, 10-source count and QCK02/QCK04 query details. The canonical GitHub source is [MigrationBench](https://github.com/yicao-elina/MigrationBench) at the pinned main commit above; this folder remains a map and index, while experimental payloads remain in the personal Scratch16 data archive.
