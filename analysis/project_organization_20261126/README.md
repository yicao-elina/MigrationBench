# Rockfish project and folder map

This bundle is the project-level and folder-level index for the user's three Rockfish roots only:

- `/data/pclancy3/yi`
- `/scratch4/pclancy3/yi`
- `/scratch16/pclancy3/yi`

It contains no scans of other users' directory trees. Start with the [R28 project and folder status](PROJECT_AND_FOLDER_STATUS_20261126_R28.md) for the complete recursive personal-root summary, canonical repository heads, project-family map, evidence/database status, cleanup progress, and grouped author decisions. The [scoped inventory package](scoped_personal_inventory_r1/README.md) provides QPR01–QPR07 over the full metadata inventory of the three authorized roots. The [depth-3 hierarchy SQLite package](scratch16_hierarchy_r1/README.md) provides QH01–QH05 over the 5,288 captured Scratch16 directories. The [R27 project/folder map](PROJECT_AND_FOLDER_STATUS_20261126_R27.md), [R26 project/folder map](CURRENT_PROJECT_AND_FOLDER_MAP_20261126_R26.md), and [R25 map](CURRENT_PROJECT_AND_FOLDER_MAP_20261126_R25.md) remain dated snapshots. The [local workspace reconciliation](LOCAL_WORKSPACE_RECONCILIATION_20261126.md) records path-level mapping needed before retiring that separate local workspace. The directory-only R21 table contains 222 direct Scratch16 child directories and their family assignment. The full R21 direct-child index preserves all 328 direct children: those 222 directories plus 106 files.

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

- MigrationBench `main` is at `19708d74d14103832e4f86b51b8b5d8875175cc3` on GitHub and Scratch16; X-FORCE `main` is at `f1e16d65e7ea5f72a2daf0c847226ba6f67f4b06`. Both GitHub repositories currently expose only `main` and have no open PR. The read-only SQL bridge has 11 pinned sources; QCE01–QCE09 verify the Revision 1 claim matrix, manuscript source versions and full PR #10 file reconciliation. PRs #5–#19 have heads in MigrationBench mainline ancestry, and PR #1–#8 are merged in X-FORCE.
- All 222 Scratch16 direct-child directories are classified. Classification describes folder purpose; it does not close artifact hashes, attempt/run lineage, manuscript claims or safe-retirement decisions. Many of the 218 noncanonical campaign directories still need those links or a written hold disposition.
- A separate local legacy workspace has uncommitted and untracked project evidence. Its six priority groups have now been mapped at path/size/hash levels where feasible; see the reconciliation report. It is not an interchangeable clean main checkout, and retirement remains gated on finishing the listed metadata mapping and any author retention choices.
- The cross-element SQL index contains 3,027 artifact metadata rows. The 2,997 intermediate checkpoints remain held pending a documented retention decision; this bundle does not authorize their deletion.
- This is a dated navigation snapshot, not a statement that the full project organization, evidence closure, manuscript reconciliation or storage cleanup is complete.

## Evidence archive

The merged PR #36–#38 sync receipt and read-only remote query outputs are archived under:

`/scratch16/pclancy3/yi/MigrationBench_data_archive/storage_audits/migrationbench_pr36_pr38_final_sync_20261126/`

The receipt records the exact merge SHA, remote refs, pack/index/file hashes, 11-source count and QCK02/QCK04, QLM01–QLM07, and QX01–QX05 query details. The canonical GitHub source is [MigrationBench](https://github.com/yicao-elina/MigrationBench) at the pinned main commit above; this folder remains a map and index, while experimental payloads remain in the personal Scratch16 data archive.
