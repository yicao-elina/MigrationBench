# Current project and folder map — R26

**Captured:** 2026-11-26 UTC, after MigrationBench PRs #36–#38 merged and the 11-source SQL bridge passed its Rockfish smoke queries  
**Rockfish storage scope:** `/data/pclancy3/yi`, `/scratch4/pclancy3/yi`, `/scratch16/pclancy3/yi` only. No other users' directory trees are in scope.

## Current project state

| Area | Current state | Remaining boundary |
|---|---|---|
| Canonical repositories | Scratch16 `MigrationBench` and `X-FORCE` remain the intended single Git checkouts, separate from their data archives. MigrationBench main is `296ef562a5bc28151a824d9201c383766cfa116f`; PRs #36–#38 are merged. | The Scratch16 sync used a verified pack/index and tracked-file transfer because Rockfish has no Git executable; it was not a remote `git pull`. Keep both projects' payload archives separate from Git. |
| Personal roots | Prior scoped inventories cover `/data/pclancy3/yi`, `/scratch4/pclancy3/yi`, and `/scratch16/pclancy3/yi`. | Recursive metadata inventories do not prove provenance, duplicate content, or quota recovery. No deletion follows from this map. |
| Scratch16 first-level organization | The 222 direct-child directories are classified; the four canonical roots are `MigrationBench/`, `MigrationBench_data_archive/`, `X-FORCE/`, and `X-FORCE_data_archive/`. | 218 noncanonical project/campaign directories still need deeper run, hash, claim mapping or a written hold disposition. |
| MigrationBench evidence SQL | The read-only federated bridge pins 11 SQLite sources. The R26 Lmax1 sidecar has 194 collector rows, 50 explicit failure-to-retry/perturbation edges, 10 planned retry variants missing, and 90 Tier-2 gates pending. | The bridge is a federated query interface, not a merged database or a complete model-by-task performance matrix. The Lmax1 evidence does not establish active-learning benefit. |
| Remote smoke | QCK02/QCK04, QLM01–QLM07, and QX01–QX05 ran on Rockfish with SQLite 3.26.0 and `query_only=true`; all succeeded. | These are query execution checks, not scientific validation. QCK04's 12 cross-element final artifacts had no exact SHA match in DUAL-X v14 or R4. |
| Raw data, checkpoints and trajectories | Large inputs, model files, trajectories, and source SQLite snapshots remain in user-owned Scratch16 campaign/archive paths. | Do not infer exact duplication from pathname and size alone. Archive identity and retention must be established per payload before cleanup. |
| Submitted paper evidence | Revision 1 claim and figure/table evidence is indexed in SQL. | The 639-row claim inventory is an evidence index, not scientific validation. QE force residuals and SCF-warning cases remain provisional; the Fig. S2 caption/data mismatch and some result/attempt links remain unresolved. The live Overleaf tip is not verified by this snapshot. |

## Research and data flow

```text
canonical Git metadata and queries
        │
        ├── MigrationBench_data_archive: MigrationBench inputs, runs, models, NEB/MD, source snapshots
        └── X-FORCE_data_archive: DUAL-X/X-FORCE inputs, runs, models, receipts
                    │
                    ▼
  exact input/model → run/attempt → raw output → processed datum
       → figure/table cell → manuscript claim/release commit
       → (if accepted) explicit cross-project selection/training decision
       → held-out evaluation and recorded gain
```

The last three links are an evidence requirement for future reuse. Current records do not establish a validated MigrationBench-to-DUAL-X active-learning gain. Model identity matches or diagnostic summaries alone are not sufficient to claim reuse.

## Repository and database milestones

- MigrationBench main is `296ef562a5bc28151a824d9201c383766cfa116f`. PR #36 added the project/folder map; PR #37 added the Lmax1 failure/retry SQL sidecar; PR #38 removed a duplicate SQLite attachment alias required for Rockfish's attachment limit. Each PR's `test-and-audit` passed.
- The SQL bridge exposes 11 pinned sources, including central catalog, R4 lineage, manifest sidecar, DUAL-X v14, MB15 coverage, MD15 attempt-lineage, Fig. 5, learning-loop review, claim matrix, cross-element inventory, and Lmax1 failure/retry.
- Remote read-only queries QLM01–QLM07 and QX01–QX05 succeeded. The Lmax1 SQLite snapshot is `/scratch16/pclancy3/yi/MigrationBench_data_archive/lmax1_g3_failure_retry_snapshots/20261126/lmax1_failure_retry_r1.sqlite`; it passed `quick_check`, has zero foreign-key violations, and is mode `0600`.
- The QLM evidence has 194 diagnostic rows, 50 exact source-record links, 10 missing retry variants, and 90 pending Tier-2 gates. It does not provide event-level accepted labels or a measured learning gain.
- The cross-element checkpoint index has 3,027 metadata rows; 2,997 intermediate checkpoint payloads remain unhashed and held pending retention review. No checkpoint payload was included in these documentation changes.
- Mainline and Scratch16 refs were synchronized to the same commit using the documented no-Git transfer process. This verifies repository state at the receipt's capture time; it does not establish the live scheduler state or replace a future Git pull.

The detailed bridge, source roles, query IDs, and scientific boundaries are in [the Revision 1 evidence bridge index](../revision1_evidence_bridge_20261126/README.md). The PR #36–#38 transfer receipt and remote query outputs are archived under `/scratch16/pclancy3/yi/MigrationBench_data_archive/storage_audits/migrationbench_pr36_pr38_20261126/`.

## Local workspace reconciliation and retirement status

A separate local legacy workspace was compared against canonical MigrationBench main and the authorized Scratch16 user tree. The exact group-level counts, bytes, path overlaps, compact-file hash outcomes, and suggested destinations are in [LOCAL_WORKSPACE_RECONCILIATION_20261126.md](LOCAL_WORKSPACE_RECONCILIATION_20261126.md). It is not a mergeable clean branch. Several high-value groups have unique evidence or conflicting snapshots, so it is not yet safe to retire. This R26 documentation does not move, hash large raw payloads, merge branches, or delete files.

## Scoped Rockfish inventory and classification

| Root | Recorded direct children | Current role |
|---|---:|---|
| `/data/pclancy3/yi` | 17 directories, 3 files | Source datasets, models, and inputs. Preserve as source evidence. |
| `/scratch4/pclancy3/yi` | 3 directories | User's working outputs. Previous refresh found no additional evidence-complete deletion target. |
| `/scratch16/pclancy3/yi` | 222 directories, 106 files | Long-lived project workspace, canonical repository roots, and campaign/archive destinations. |

The `/data` and `/scratch4` figures come from scoped metadata inventories, not quota measurements; hardlinks may make path byte totals exceed allocated storage. Directory classification does not close per-file provenance.

## Decisions versus routine closure work

### Author decisions that affect science or retention

1. **R05 source of record:** if the manifest-pinned PDF cannot be recovered, decide whether the historical expected artifact, the present same-name candidate, or a newly reconstructed export is authoritative. The present candidate does not match the recorded expected hash; see the reconciliation report.
2. **Checkpoint retention:** choose whether to retain all 2,997 intermediate payloads or review a verified, lineage-aware prune list when ready. They remain held in the meantime.
3. **Cross-project learning policy:** define which diagnosed failure signals may affect DUAL-X selection/training, the required baseline and held-out comparison, and what evidence counts as gain. Existing records do not answer this.
4. **External source-data retention:** only if re-archiving a collaborator-owned input becomes necessary, obtain the relevant owner's direction before changing its location or retention.

### Routine work that can proceed without choosing scientific authority

- Compare the SHAP CSV exports with newline-normalized and row-level checks, then add an explicit source/version mapping; retain the current SHA mismatch as a recorded integrity issue until resolved.
- Build file-to-campaign path maps and compact checksums for small manifests, scripts, and derived evidence; do not treat path/size agreement as content identity.
- Recover or locate the R05 PDF candidate, inspect its source metadata and production recipe, and prepare a side-by-side evidence packet for the author if hashes remain inconsistent.
- Map model-weight manifests to exact checkpoints and consuming campaigns, keeping payloads out of Git and preserving scientific status labels.
- Continue mapping the 218 noncanonical Scratch16 folders to campaign, source, claim, and hold/retention state.

## Completion limits

This is a dated navigation and reconciliation snapshot, not a declaration that all experiments or manuscript evidence are complete. The full measured MigrationBench model/task matrix is not unified. The SHAP source exports have a byte-level manifest mismatch. Several endpoint-only NEB inputs and outputs remain incompletely mapped; the preflight audit reports zero of ten DFT gate successes and does not establish formal MACE NEB results. DFT/SCF warnings remain provisional. No raw payload has been moved or deleted as part of this R26 update.
