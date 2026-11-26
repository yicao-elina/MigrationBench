# Current project and folder map — R25

**Captured:** 2026-11-26 UTC, after MigrationBench PR #35 merge and Scratch16 SQL sync
**Rockfish storage scope:** `/data/pclancy3/yi`, `/scratch4/pclancy3/yi`, `/scratch16/pclancy3/yi` only. No other users' directory trees are in scope. A separate local checkout comparison is included below because it affects safe Git/worktree retirement.

## Executive view

The existing hierarchy is understandable at the top level, but the evidence catalog and cleanup are not yet complete. The strongest current foundation is the pair of canonical Scratch16 Git checkouts, their separate data archive, and the read-only federated SQLite/query layer. The main remaining work is connecting campaign-level raw evidence to exact attempts, derived outputs, and paper claims, then deciding which historical payloads can be archived or pruned.

| Area | Current state | Completion boundary |
|---|---|---|
| Personal roots | All three requested roots have scoped inventories; `/data` and `/scratch4` recursive inventories are metadata-only. | Counts do not prove scientific provenance or quota release. |
| Scratch16 first-level organization | All 222 direct-child directories classified into project families. | 218 noncanonical campaign/project directories still need deeper run/hash/claim mapping or a documented hold disposition. |
| Canonical repositories | `MigrationBench` and `X-FORCE` each have a single `main` checkout on Scratch16, with matching `origin/main` refs at last verified sync. | Rockfish lacks Git; sync receipts verify tracked files and Git internals, but are not a remote `git pull`. |
| Local MigrationBench workspace | `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench` is a separate stale `master` tree with no remote, 20 tracked modifications and 12,719 untracked files (~17.17 GB logical). | It has no merge base with its stale GitHub ref, and many untracked files are unique. Preserve it until exact content is mapped and verified; no `git clean`, merge, or retirement. |
| MigrationBench SQL | Ten sources are merged on GitHub main. Cross-element checkpoint inventory is merged at `3c9f598ff96d3b52cffad4f1205efaae3f1a304d` and synced to Scratch16. | Query coverage is still a federated evidence index; exact identity does not establish a shared result or scientific reuse. |
| Raw scientific payloads | Stored under Scratch16 campaign trees and archive roots; large checkpoints/trajectories remain preserved. | No bulk relocation or deletion until each payload has exact mapping, retention decision, and verified archive copy. |
| Paper evidence | Submitted Revision 1 freeze is the supported manuscript baseline; claim matrix is SQL-queryable. | Live Overleaf tip is unverified; several claim/figure, QE and attempt-lineage gaps remain unresolved. |
| MigrationBench → DUAL-X reuse | Exact model identity and typed coverage can be queried. | There is no complete event-to-selection-to-training-to-held-out-gain chain; identity matches alone are not reuse evidence. |

## Project-level hierarchy

```text
/data/pclancy3/yi                 source datasets, inputs, models and tool data
          │                         (preserve as source evidence)
          ▼
/scratch4/pclancy3/yi             active/legacy working outputs
          │                         (scoped inventory; no new safe deletion batch)
          ▼
/scratch16/pclancy3/yi
  ├── MigrationBench/             canonical Git checkout, main
  ├── MigrationBench_data_archive/ raw campaigns, sidecars, snapshots, storage audits
  ├── X-FORCE/                    canonical Git checkout, main
  └── X-FORCE_data_archive/       DUAL-X / X-FORCE payload and evidence archive
          │
          ├── immutable raw/model/run artifacts + manifests/hashes
          ├── compact, versioned SQLite provenance sources
          ├── read-only federated SQL and stable query IDs
          └── claim → figure/table → release mapping
```

The desired research flow is:

```text
exact input/model → run/attempt → raw output → processed datum → figure/table cell
       → manuscript claim and release commit → (if accepted) cross-project reuse
       → explicit selection/training decision → new checkpoint → held-out evaluation
```

The last three links are a future evidence requirement. Present data do not yet show a validated active-learning gain.

## Separate local checkout and worktree retirement

`/Users/alina/Documents/ChatGPT/Wiley-MigrationBench` is not the canonical clone and is not inside the Rockfish inventory roots. It is at `master` commit `ffa168c76f133a7a4d817bcd851bc8a8e6cfeccd`, has no configured remote, and has no merge base with its stale `github/main` ref. Its committed tree contains 1,987 files versus 15,571 in canonical MigrationBench main. In addition to 20 modified tracked paths, it has 12,719 untracked regular files (17,170,974,430 logical bytes), seven untracked symlinks, and 15 ignored records. 9,857 untracked paths are absent from the canonical tracked tree; pathname collisions do not prove byte identity. It is therefore a **preserve-and-map** workspace, not a mergeable feature branch or safely disposable worktree. High-value unique areas include `analysis/canonical_model_matrix_20261016/`, `analysis/cr_interlayer_sliding_grouped_20261025/`, `analysis/task3_endpoint_only_neb/`, `analysis/shap_full_campaign_20261101/`, and `paper_evidence/`. The branch/worktree audit should map each coherent source to canonical Git or the matching Scratch16 campaign before cleanup.

## Root and folder inventory

| Root | Direct children | Recursive metadata inventory | Role and current disposition |
|---|---:|---:|---|
| `/data/pclancy3/yi` | 17 directories, 3 files | 79,150 dirs; 673,601 files; 22,992 symlinks; 257,589,198,471 logical bytes; 270,858,120,704 allocated-by-path bytes | Source datasets and inputs. Keep original evidence; do not infer quota savings from path sums. |
| `/scratch4/pclancy3/yi` | 3 directories, 0 files | 59 dirs; 2,875 files; 382,958,312 logical bytes; 403,117,056 allocated-by-path bytes | User's working outputs. No new evidence-complete deletion target in the last refresh. |
| `/scratch16/pclancy3/yi` | 222 directories, 106 files | First-level classification complete; full per-file provenance remains partial | Long-lived project/workspace and canonical repository host. |

`/data` and `/scratch4` sums are metadata-derived and not quota measurements; hardlinks may cause path sums to count the same storage more than once. Historical scan permission caveats that applied to other owners' directories are outside this personal-root scope.

### Scratch16 direct-child classification

| Project family | Directories | Current interpretation |
|---|---:|---|
| MLIP / MD | 86 | Training, inference, force-field and MD campaigns |
| Lmax1 diagnostics | 36 | Tiered failure diagnostics, retry and perturbation campaigns |
| MigrationBench Revision 1 / model evaluations | 32 | Revision evidence, model matrices and analysis packages |
| PolymerNetwork | 25 | Separate project campaigns and studies |
| MigrationBench NEB / DFT | 23 | NEB, relaxation and reference calculations |
| Shared tools | 6 | Reusable utilities and environment support |
| Governance / archive / smoke | 4 | Audits, receipts, validation and administrative evidence |
| Cross-element | 4 | Cross-element training and checkpoint evidence |
| Canonical / archive roots | 4 | Main repository and data archive roots |
| FLARE | 1 | Separate project directory |
| Historical X-FORCE / DUAL-X | 1 | Legacy evidence workspace, retained pending mapping |

Classification identifies what each folder is; it does not mean all files inside are hashed, tied to a run, or safe to remove. The older `SCRATCH16_PROJECT_CAMPAIGN_MAP_20261125.md` has superseded category counts and an obsolete “unclassified” count; use R24/R25 plus `folder_classification_update_R21/SCRATCH16_FOLDER_CLASSIFICATION_20261126_R21.tsv` as the current classification authority, and mark the older narrative historical.

## Database, evidence and cleanup progress

- MigrationBench `main` after PR #35: `3c9f598ff96d3b52cffad4f1205efaae3f1a304d`; X-FORCE `main`: `f1e16d65e7ea5f72a2daf0c847226ba6f67f4b06`. Scratch16 MigrationBench `main` and `origin/main` both match the merged SHA; X-FORCE remains at its previously verified matching SHA.
- The federated MigrationBench query layer now has ten merged read-only SQLite sources. PR #35 added 3,027 cross-element artifact rows and three historical Slurm run snapshots. It preserves 2,997 intermediate checkpoints as `HOLD_PENDING_USER_POLICY` and does not hash, move, or delete their payloads.
- PR #35 `test-and-audit` passed, including Snakemake, pytest, LaTeX, and clean-diff checks. The bridge tracked-file hashes, Git pack/index hashes, and Scratch16 refs were verified. Remote QCK02 returned 27 summary rows; remote QCK04 returned all 12 final artifacts with no exact SHA matches in DUAL-X v14 or R4. Both queries used SQLite 3.26.0 with `query_only=true`; full JSON outputs and the sync receipt are archived at `/scratch16/pclancy3/yi/MigrationBench_data_archive/storage_audits/migrationbench_pr35_cross_element_20261126/` and locally under `PR35_CROSS_ELEMENT_SQL_20261126/`.
- Previously completed WFC/mixing and stale-transfer deletions have receipts. Recorded bytes are logical/file bytes, not measured quota recovery.
- 2,997 intermediate checkpoints (about 828.46 GiB allocated by path) remain held. A precise prune list, complete lineage/archive verification, and the author's retention rule are still needed.
- Historical DUAL-X and MigrationBench worktrees remain held where embedded repositories, unique/untracked files, or old snapshots are not mapped to verified canonical evidence. Do not delete them to make the top-level directory count smaller.

## Scientific coverage boundaries

- MigrationBench's complete measured model-by-task attempt matrix is not yet unified. R4, Figure 5, MD15, static scoring, DUAL-X v14 and NEB sources have different grains and must stay distinct.
- The 639-row Revision 1 claim matrix is an evidence index (13 traceable, 313 partial, 313 missing), not a validation of those claims.
- QE force residuals and SCF-warning cases remain provisional; the Fig. 4/Table S3 pairing, Fig. S2 caption/data mismatch, and some restart/attempt links remain unresolved.
- The Lmax1 evidence supports an additive source-linked failure → retry record, including 10 missing retry variants and 90 pending Tier2 gates; it does not yet support an active-learning decision or measured gain.
- The current X-FORCE/Dual-X crosswalk can identify exact model-hash overlaps. Reusing a MigrationBench signal still requires a task/protocol match, exact attempt and input/output artifacts, accepted failure/convergence semantics, and an explicit DUAL-X selection/training/held-out record.

## Decisions to batch for the author

These are the choices that change scientific meaning or irreversible retention; routine indexing, checksums, and report refreshes do not need a decision.

1. **Checkpoint retention:** keep all 2,997 intermediate checkpoints or review a hash- and lineage-verified prune list once prepared.
2. **Manuscript authority:** continue using the submitted Revision 1 freeze unless a newer dated Overleaf export/version is supplied.
3. **Figure and DFT interpretation:** choose the authoritative Fig. 4/Table S3 attempt pairing; resolve Fig. S2 caption versus plotted source; decide whether residual/SCF-warning QE rows should remain provisional, be reconverged, or be excluded from paper claims.
4. **Cross-project learning policy:** define which diagnostic/failure signals can change DUAL-X selection or training, the required held-out baseline, and what evidence constitutes gain. Current records do not imply an answer.

## Next completion sequence

1. Add the Lmax1 failure/retry sidecar as a separate reviewable PR; keep event, planned, missing, diagnostic, and pending-gate rows explicit, without active-learning claims.
2. Add the summary-grain Fig. 4 → R05 row crosswalk separately, with the unresolved cell explicit and no invented attempt identity.
3. Continue campaign-by-campaign file/run/claim mapping for the 218 noncanonical directories; archive or retire only after exact content is mapped and verified.
4. Prepare the 2,997-checkpoint retention manifest and author review packet; do not prune until the decision and archive gate are satisfied.
