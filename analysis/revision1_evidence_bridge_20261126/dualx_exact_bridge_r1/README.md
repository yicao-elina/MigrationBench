# MigrationBench → DUAL-X exact evidence bridge (read-only)

Captured 2026-11-26. Scope is the user's personal Scratch16 roots and the two canonical local repositories. This package exports existing rows; it does not modify either source database, repository, model, trajectory, or scientific claim.

## Pinned sources

| Source | Identity |
|---|---|
| MigrationBench canonical main | `cf6c2b1649ba993d079d0eadf8478775a81736d2` |
| MigrationBench R4 lineage snapshot | `/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_lineage_overlays/20261125_r4/migrationbench_r1_lineage_overlay_20261125_r4.sqlite`; SHA-256 `d033f47913eaf33d61a2ce7b0b0281b90b385d23190890a97f477e2f901896d7` |
| X-FORCE canonical main | `f1e16d65e7ea5f72a2daf0c847226ba6f67f4b06` |
| DUAL-X v14 result DB | `/scratch16/pclancy3/yi/X-FORCE_data_archive/sqlite_sources/revision1_dualx_20261125_v14/revision1_dualx_results.sqlite`; SHA-256 `a7e46f4bdf5ae134ab501b308dbfa987348d4a15201503297edd540c954e6fa4` |

The two source DBs remain separate and immutable. `EXTERNAL_EVIDENCE_BRIDGE_DRAFT.sql` is for read-only attached-database queries; use exact checkpoint SHA, then task/system/protocol/input/result identity. Never promote a SHA-only match to a result or manuscript claim.

## What the exact join establishes

- DUAL-X v14 registers **23 models × 18 tasks = 414 design cells** and **80 run rows**. Runs comprise 40 inference and 40 GradCAM executions; 58 are primary and 22 exploratory. There are 29 primary model/task contexts, 79 complete runs, and one partial run. A design cell is not a measured result; empty or blocked cells remain visible in `DUALX_MATRIX_CELLS_V14.tsv`.
- The exact SHA join finds all **23/23 DUAL-X checkpoint hashes** in the frozen MB `rev1_model_registry` field `sha256_recorded` (25 matching rows because two Foundation checkpoints have two recorded path aliases). These are registered hash matches, not a fresh rehash of all checkpoint files on the capture date.
- Only **7/23** have a match in MB R4's current canonical `models.checkpoint_sha256` table. The v14 explicit `mb_external_model_links` relation contains 29 `PASS` rows but only **9 distinct DUAL-X model IDs** and 21 MB model IDs. The MB canonical model acceptance snapshot contains 21 models: 8 SHA-linked and 13 `NOT_LINKED`. These layers have different grains and should not be flattened into one count.
- `mb_external_model_coverage` contains **115** source/model rows: 29 `LINKED`, 86 `NOT_LINKED`. The MB catalog also has 110 declared model/task/system/campaign coverage rows; this is coverage metadata, not 110 performance results.

## Result evidence that can be reused, with limits

| Evidence | Exact rows | Reuse boundary |
|---|---:|---|
| MB static candidate predictions | 1,278 = 2 checkpoint SHAs × 639 candidates | Exact per candidate/geometry diagnostics for `foundation_mp0b3` and `foundation_mpa0`. One static task and prepared candidate set. Not DUAL-X common or MultiT held-out measurements. |
| MB NEB summaries | 18 = 6 checkpoint SHAs × 3 path rows | Exact model/task/path and manifest rows. 15 are `CONVERGED / PRIMARY_EVIDENCE`; 3 are `FAILURE_MODE / DIAGNOSTIC_ONLY`. Keep fixed-path, self-relaxed, and other path protocols separate; no model-wide barrier score. |
| Historical grouped-A2 MD status | 6 status rows | Snapshot states only (four scientific-threshold failures and two running at the historical capture). No accepted 1 ns trajectory or diffusivity follows from these status rows. |
| DUAL-X v1/v2 source coverage | 115 coverage rows | Preserve the 86 unmatched rows. `LINKED` is not a task result or claim verification. |

The extracted source rows are in `DUALX_STATIC_EXACT_ROWS_V14.tsv`, `DUALX_NEB_EXACT_ROWS_V14.tsv`, `MB_MD_STATUS_V14.tsv`, and `DUALX_MODEL_LINKS_V14.tsv`. The per-model summary is `MODEL_IDENTITY_BRIDGE.tsv`. `DUALX_RUNS_V14.tsv` and the full design list preserve DUAL-X execution and planned states separately.

## Main unresolved bridge gap

This package does **not** create a complete 41×16 MigrationBench result matrix in DUAL-X. The existing MB canonical 21-model acceptance snapshot and the 110-row catalog are not full performance matrices. Exact hash identity is broader than exact source-result/task evidence: only two checkpoints currently have 639-row exact static extracts, six have the 18 row-level NEB summaries, and the MD snapshot is status-only. Claims, publication panels, run attempts, references, geometry/protocol equivalence, and acceptance must still be joined for each requested use.

The 29 explicit external model-link rows should be used as the current reviewed bridge. The 23/23 raw registry hash presence is an additional identity audit layer; it must not be substituted for task/run/source-result joins. The earlier 10/23 crosswalk, 23/23 broad raw-registry check, and 7/23 current canonical model table answer different questions; this package pins those grains separately instead of expanding the 10-row crosswalk automatically.

## Query use

`EXTERNAL_EVIDENCE_BRIDGE_DRAFT.sql` provides QX01–QX05 for: model SHA identity layers, static candidate rows, per-path NEB rows, all 414 cells with actual runs left-joined, and source coverage status. Open v14 read-only, attach the MB R4 database read-only, and run QX01 first. QX02/QX03 return evidence by model, task, and geometry/path without claiming equivalence to DUAL-X R1 held-out observations.

## Package files

- `MODEL_IDENTITY_BRIDGE.tsv` — one row per DUAL-X checkpoint with separate raw-registry, current-MB-model, explicit-link, static, NEB, and observed-run counts.
- `DUALX_MATRIX_CELLS_V14.tsv` — all 414 design cells and planned states.
- `DUALX_RUNS_V14.tsv` — all 80 run rows with model/task/scope/type/status/checkpoint/input/output-manifest identities.
- `DUALX_STATIC_EXACT_ROWS_V14.tsv` — 1,278 exact model/candidate rows with geometry hashes.
- `DUALX_NEB_EXACT_ROWS_V14.tsv` — 18 exact model/path rows with attempt, manifest, convergence, and publication state.
- `DUALX_MODEL_LINKS_V14.tsv`, `DUALX_SOURCES_V14.tsv`, `MB_MD_STATUS_V14.tsv` — existing explicit link rows, source declarations, and the historical MD status-only rows.
- `MB_CANONICAL_MATRIX_MODELS_V14.tsv` — the 21-model acceptance snapshot, including 8 exact SHA links and 13 unmatched models.

`SHA256SUMS.txt` and `BRIDGE_SNAPSHOT_MANIFEST.json` pin the exports, source DB hashes, commits, and counts. No output here changes the DUAL-X release or MB publication status.
