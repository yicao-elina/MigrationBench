# Rockfish personal-root project and folder status — R28

**Captured:** 2026-11-26 UTC
**Scope:** only `/data/pclancy3/yi`, `/scratch4/pclancy3/yi`, and `/scratch16/pclancy3/yi`. No other users’ trees were scanned.
**Purpose:** traversal-complete recursive metadata index and organization progress report, except for the inventory archive subtree explicitly excluded from the Scratch16 walk to prevent self-inclusion. It is a map and evidence index, not proof every artifact is scientifically validated or safe to delete.

## Executive map

| Root | Files | Directories* | Symlinks | File logical bytes | Allocated path sum | Allocated unique inodes | Scan errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| `/data/pclancy3/yi` | 673,601 | 79,149 | 22,992 | 257,588,816,651 (257.59 GB) | 270,846,348,800 | 257,957,907,456 | 0 |
| `/scratch16/pclancy3/yi` | 209,915 | 37,108 | 1,079 | 1,190,433,730,511 (1190.43 GB) | 1,195,296,341,504 | 1,195,292,474,368 | 0 |
| `/scratch4/pclancy3/yi` | 2,875 | 58 | 0 | 382,958,312 (0.38 GB) | 403,117,056 | 403,117,056 | 0 |

*Directory count in the table is descendants only; the root itself is also recorded in SQLite. There are **1,026,780 path records including the three root rows**. The inventory archive subtree under Scratch16 is excluded. Allocated path sum counts hardlinked paths multiple times. Neither allocation metric is a quota measurement. Across the inventoried paths: **886,391 files**, **116,315 descendant directories**, **24,071 symlinks**, **1,448,405,505,474 logical bytes**, **1,466,545,807,360 allocated path-sum bytes**, and **1,453,653,498,880 unique-inode allocated bytes**. There are 167,694 extra hardlink paths over 718,697 unique file inodes; the 12,892,308,480-byte path-sum difference is shared inode accounting, not reclaimable space.

QPR04 finds 150,024 hardlink groups in `/data`, two in `/scratch16`, and none in `/scratch4`. For likely package-cache/environment hardlinks, deleting one pathname does not free the inode's blocks while another link remains.

## Research and storage organization

```text
/scratch16/pclancy3/yi/
├── MigrationBench/                  canonical MigrationBench Git checkout (main)
├── MigrationBench_data_archive/     raw evidence, model/run payloads, immutable SQL snapshots
│   ├── revision1_migrationbench_runs/
│   ├── revision1_md_fig6_table_s8_20261124/
│   ├── provenance_catalog_snapshots/
│   ├── revision1_claim_matrix_snapshots/
│   └── storage_audits/ and project_hierarchy_snapshots/
├── X-FORCE/                         canonical X-FORCE / DUAL-X Git checkout (main)
├── X-FORCE_data_archive/            DUAL-X inputs, model/results and receipts
└── 218 other project/campaign directories, currently indexed by family in R21
```

The intended information flow is **shared model/checkpoint lineage → MigrationBench model × task runs → raw outputs and derived results → figure/table cells and manuscript claims → explicitly gated DUAL-X diagnostic reuse → held-out test and measured gain**. The archives hold large payloads; Git holds code, compact manifests, SQL/query contracts, and paper-facing evidence indexes. The sidecar inventory database is separate from scientific provenance and remains on Rockfish.

### Three-root top-level folder picture

#### `/data/pclancy3/yi` — source data, QE/FLARE datasets, software environments, and other project material

| Top-level folder | Files | Descendant directories | File logical size |
|---|---:|---:|---:|
| `flare-data` | 48,025 | 7,504 | 202.55 GB |
| `.conda` | 598,792 | 69,103 | 45.72 GB |
| `cyqe` | 2,786 | 174 | 2.22 GB |
| `26ICLR` | 2,512 | 384 | 1.60 GB |
| `LLM` | 1,589 | 314 | 1.48 GB |
| `mylammps` | 16,230 | 1,227 | 1.47 GB |
| `other` | 465 | 7 | 1.33 GB |
| `ONCVPseudoPack` | 905 | 41 | 0.50 GB |
| `ycao-pip` | 606 | 95 | 0.45 GB |
| `25npj_CM` | 622 | 60 | 0.15 GB |

#### `/scratch4/pclancy3/yi` — small active/intermediate working area

| Top-level folder | Files | Descendant directories | File logical size |
|---|---:|---:|---:|
| `2.Cr-Sb2Te3` | 2,843 | 46 | 0.38 GB |
| `3.WSe2` | 31 | 9 | 0.00 GB |
| `cluster_recipes` | 1 | 3 | 0.00 GB |

#### `/scratch16/pclancy3/yi` — canonical repositories, long-lived archives, and campaigns

| Top-level folder | Files | Descendant directories | File logical size |
|---|---:|---:|---:|
| `cross_element_generalization_seeds_20261004` | 2,023 | 4 | 594.10 GB |
| `cross_element_generalization_final_20261004` | 1,013 | 3 | 297.05 GB |
| `MigrationBench_data_archive` | 2,977 | 1,399 | 54.06 GB |
| `FLARE` | 4,399 | 482 | 42.85 GB |
| `revision1_migrationbench_runs` | 8,597 | 1,940 | 24.99 GB |
| `mace_neb_cr_historical_reconstructed_four_model_20261016` | 2,570 | 434 | 11.85 GB |
| `defects_scripts` | 7,766 | 443 | 10.88 GB |
| `X-FORCE_data_archive` | 2,245 | 947 | 10.53 GB |
| `foundation_self_relaxed_4x3_20261024_r02` | 2,202 | 251 | 10.44 GB |
| `scratch_lmax1_g3_tier1_retry_20261025_r01` | 380 | 45 | 9.73 GB |

## Repository and source reconciliation

- MigrationBench canonical GitHub `main` and Scratch16 `/MigrationBench` are both at `19708d74d14103832e4f86b51b8b5d8875175cc3`; X-FORCE GitHub `main` and Scratch16 `/X-FORCE` are both at `f1e16d65e7ea5f72a2daf0c847226ba6f67f4b06`. Both repositories expose only `main`, have no open PR, and the Scratch16 checkouts have no local side branches. Merged/closed PR history has been mapped to main. Do not treat legacy local worktrees as duplicates until every unique file/commit/evidence pointer is reconciled.
- PR #10’s 47 paths: 44 byte-identical at their same paths; 3 manuscript/SI/ledger versions preserved as separate candidate files. PR #10 and #11 heads are ancestors of MigrationBench main. No code merge is needed for those PR heads; source-version and manuscript promotion questions remain separate.
- Revision 1 SQL claim snapshot R3 records 639 submitted-release claim rows, 20 metadata rows, 30 Fig. 4/R05 mappings, 18 ledger deltas, and all 47 PR #10 path reconciliations. Federated query checks QCE01–QCE09 passed on Rockfish SQLite 3.26 in read-only mode. This remains a federated set of pinned sources, not one physical mega-database or a complete observed model × task matrix.
- Scratch16 hierarchy R1 has 5,288 directory records through depth 3. R21 classifies all 222 direct-child directories into project families. The new R1 personal-root inventory adds 1,026,780 total path entries recursively across the 3 roots, zero traversal errors, and QPR01–QPR07. Individual family labels and file groups are navigation data; claim/run/checkpoint closure remains a separate task.
- The archive contains an 11-file SQLite catalog snapshot chain through R11 (~44 GB). R10/R11 receipts describe incremental row additions; the existing catalog index is stale at R9. No snapshots have been deleted or treated as duplicates.

## Current contents by project-family priority

1. **MigrationBench Revision 1, manuscript and matrix evidence:** submitted manuscript source is pinned; candidate paper files remain separate. The 23 model × 18 task design (414 units) has 80 measured run records, not 414 completed results. QE NEB residual/SCF warnings remain provisional; Fig. S2 caption/data mismatch and MD provenance gaps remain explicit.
2. **MigrationBench MD / failure-attribution evidence:** PR #10 campaign has 639 candidates and 1,917 successful static model evaluations across 3 checkpoints, 9 MD runs (8 failed, 1 reached 1 ns), and no completed DFT result. This is screening and observed failure data; it does not establish calibrated uncertainty, causal mechanism, DFT correctness, or active-learning benefit.
3. **Shared model training and cross-element campaigns:** checkpoints and many training/MD directories can inform both studies after lineage, protocol, system, and input hashes are linked. The 2,997 intermediate checkpoints remain held; their retention decision follows hash/lineage review.
4. **X-FORCE / DUAL-X:** its sole Scratch16 checkout is current with GitHub `main`; an exact-checkpoint-SHA subset can be exposed as read-only exploratory evidence while retaining source IDs and status. The DUAL-X external-evidence package pins its MigrationBench source revision to `412ba9c5` (PR #20), while the current MigrationBench `main` is `19708d74`; its 1,278/7,668 static, 42/120 NEB, and 90/171 Tier 1 exact-hash matches need a fresh source pin before claiming current-main coverage. The 7,668-row source is a separate 12-checkpoint prepared-candidate table, distinct from PR #10’s R4 snapshot of 1,917 rows over 3 checkpoints. These counts measure identity overlap only, not accuracy, completeness, equivalent tasks/protocols, or gain. Q06 still has no reviewed cross-project claim links. The MigrationBench dataset is not a completed DUAL-X matrix and must not be relabeled as a DUAL-X result.
5. **Other projects and infrastructure:** FLARE, PolymerNetwork, Cr/Sb₂Te₃, WSe₂, environments, scripts and other study trees remain separately classified. Names alone do not justify movement or deletion.

## Cleanup and storage progress

- Prior guarded cleanups removed 3,817 QE WFC restart and 481 mixing-restart files from the Revision 1 run tree: 4,298 files, 1,036,297,176,856 recorded pre-delete logical bytes. The receipt is historical evidence; its byte total is not a quota-after reading.
- A separate 2026-11-23 guarded batch removed 7,581 restart files totaling 1,225,323,826,928 recorded logical bytes; 1,584 nonempty `.mixN` members in `/data` account for the 9,236,905,920-byte cohort, followed by the eight zero-byte placeholders on 11/24. The receipts support pathname removal, but no personal quota delta was measured.
- R1 recursive metadata scan found zero traversal errors while excluding its own archive subtree under Scratch16. A broad case-insensitive `.wfc`/`.mix` filename query found 15 nonempty hits (468,957 logical bytes), all ordinary LAMMPS example inputs/logs under `/data/pclancy3/yi/mylammps/examples`. A second read-only query over the full inventory using exact `*.wfcN`, `wfc*.dat`, and `*.mixN` patterns found zero candidates and zero bytes. The path-level outputs are archived in the R1 Rockfish inventory directory; the report records only the counts.
- A historical 2026-11-24 snapshot reported 1,592 exact `.mixN` paths in `/data/pclancy3/yi` totaling 9,236,905,920 logical bytes after a separate 384-file cleanup. Their removal is supported by the 2026-11-23 guarded 7,581-file restart receipt, which includes the 1,584 nonempty entries in this cohort, and the 2026-11-24 receipt for the eight remaining zero-byte placeholders. R1 on 2026-11-26 found no such names. The 1,976-entry difference between the older composite inventory and R1 reconciles exactly to those 384 + 1,584 + 8 documented deletions. Zero-byte placeholders contribute no logical bytes; no quota delta was measured.
- No new deletion was made from the 15 benign `.mix` substring matches. The earlier 4,298-file Revision 1 restart cleanup remains independently receipt-backed; its byte total is pre-delete logical size, not quota-after savings.
- Biggest holders by logical size include `/data/.../flare-data` (~202.55 GB) and `.conda` (~45.72 GB); `/scratch16/.../cross_element_generalization_seeds_20261004` (~594.10 GB), `cross_element_generalization_final_20261004` (~297.05 GB), `MigrationBench_data_archive` (~54.06 GB), `FLARE` (~42.85 GB), and `revision1_migrationbench_runs` (~24.99 GB). These are research/model payloads or environments and need provenance/retention decisions before cleanup.
- The largest individual `/data` file is a 10.83 GB QE `atomic_proj.xml`; the next is a 9.45 GB X-FORCE Git pack stored inside `flare-data`; a 3.58 GB NEB trajectory and other multi-GB QE/ML dataset files follow. Their names, age, and size are insufficient deletion grounds.
- The R1 scan itself is metadata-only and sequential, not atomic. Rockfish has no fresh personal quota reading, so logical bytes and allocated bytes must not be reported as quota recovery.

## SQL and archive entry points

- Three-root metadata snapshot on Rockfish: `/scratch16/pclancy3/yi/MigrationBench_data_archive/storage_audits/scoped_personal_root_inventories/20261126_r1/scoped_personal_root_inventory_r1.sqlite` (495,513,600 bytes; mode 0600; SHA-256 `712125e415b6e646f4c27a4971a62e1ffe0afddd182b821cc9068c7749405971`; `quick_check=ok`; FK violations 0).
- Local query package: [`scoped_personal_inventory_r1/README.md`](scoped_personal_inventory_r1/README.md). QPR01 root rollup; QPR02 top-level folder totals; QPR03 largest files; QPR04 hardlink groups; QPR05 errors; QPR06 directory depth; QPR07 global unique inode allocation.
- The manifest includes exact scope, capture method, directory/file totals, hashes, mode, and recovery note for the scanner’s final reporting bug: [`SNAPSHOT_MANIFEST_R1.json`](scoped_personal_inventory_r1/SNAPSHOT_MANIFEST_R1.json). Rockfish receipt: [`ROCKFISH_ARCHIVE_RECEIPT_R1.json`](scoped_personal_inventory_r1/ROCKFISH_ARCHIVE_RECEIPT_R1.json).
- The normalized PR #10 source/evaluation database is archived at `/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_pr10_screen_snapshots/20261126_r4/snapshot.sqlite` (17,055,744 bytes, SHA-256 `35eccb46790ff1d576d8598bade50cff4e1051e1e4b858bb19f8d1a5fa8540ec`; source commit `19708d74`). It has 639 candidate/frame rows, 9 MD runs, 3 evaluated checkpoints, 1,917 successful static results, 4 figure panels, 13 claims, 23 evidence references, and `quick_check=ok` with 0 FK violations. Query with [`queries.sql`](../revision1_pr10_screen_sql_r4_20261126/queries.sql); the importer and build manifest are next to it. The database is separate from the 11-source federation because of Rockfish SQLite’s attachment limit.
- Scientific claim SQL and read-only federation: [`Revision 1 bridge`](../revision1_evidence_bridge_20261126/README.md). These are deliberately distinct databases: one locates storage paths; the other links paper evidence to pinned scientific sources.

## Author decisions to make together

1. **Manuscript authority:** keep submitted Revision 1 as fixed historical release and export/pin current Overleaf as a separate candidate (recommended), or promote a different version only after exact diff and review. Live Overleaf was not verified in this pass.
2. **DUAL-X reuse gate:** permit MigrationBench failure/diagnostic signals as exploratory inputs, or keep them diagnostic-only until a predeclared held-out comparison shows a gain. The current evidence supports diagnostic-only pending that experiment.
3. **Checkpoint retention:** keep all 2,997 intermediates until hashes and lineage are mapped, or define keep-best/keep-final after verification. Current policy remains hold.
4. **SQLite catalog versions:** retain all incremental 20261125 R1–R11 snapshots, or produce one verified compacted R11 canonical snapshot plus a durable delta/receipt archive. Do not delete any until manifest and lookup equivalence are demonstrated.
5. **Source workspaces and staging:** approve exact duplicate deletion only after every candidate is mapped and verified by hashes, repository ancestry and repaired path pointers. There is no safe blanket branch/worktree cleanup today.

The DUAL-X boundary should be specified as a prospective signal/threshold → replicated status → explicit next-round action → parent/child run and config hashes → acceptance gate → measured outcome against a named matched baseline. Current evidence has no replicated-gain result (`NOT_EVALUATED`). Identity-only matches do not establish input or task equivalence; 24 NEB sliding-cell hashes remain unresolved, and 72 sibling-cell hashes conflict with fixed-path hashes. Keep these rows `NOT_LINKED`, `CONFLICT`, or `PARTIAL` until reviewed; never combine fixed-path and self-relaxed NEB results under one score.

## Completion status and boundaries

The three authorized roots have traversal-complete metadata scans, with the Scratch16 inventory archive subtree deliberately excluded; the earlier 1,976-entry difference is matched to guarded cleanup receipts. The two Scratch16 Git checkouts align with GitHub main. Project organization, evidence closure, model-matrix population, DUAL-X transfer evaluation, catalog retention/compression, quota-confirmed cleanup, and local/worktree retirement remain in progress. Scientific conclusions have not been rewritten.
