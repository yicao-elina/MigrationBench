# MigrationBench full model-matrix source audit

**Scope:** Read-only audit of the named personal project roots and pinned MigrationBench / DUAL-X sources. No scientific data or SQLite source was modified.

## Finding

The current catalogs do not contain one complete, attempt-linked MigrationBench model-by-task matrix. The evidence is queryable in several sources, but those sources carry different grains and protocols. They must remain distinct until exact attempt and artifact links are established.

- **R4 overlay:** 44 model identities, 16 task definitions, 110 sparse coverage rows, 229 lineage-result summaries, and 16 lineage-run rows. Coverage spans 32 non-null model IDs and 12 task IDs. Coverage is expected/completed/status metadata, not metric data. The 229 summaries consist of 120 Figure 5 processed cells, 76 submitted-table cells, 30 Figure 4 profile summaries, and 3 terminal-state summaries.
- **Figure 5 release slice:** 15 × 8 = 120 rows (40 primary, 5 diagnostic-only, 75 supporting); 24 rows lack a model SHA. Input TSV SHA-256: `ee9d43cbb4f6141fb50730a073b25d0a71f6963379ab54614ebe4b17a5019ead`; SQLite SHA-256: `2a9e78f0a01ec1ae809055eaa711752c417f447022bf1748b2aa4c10428500dd`. This is the submitted Figure 5 slice, not the full matrix.
- **MD15 lane:** 15 model-level statuses (3 provisional target, 1 target reached, 11 scientific failures), plus separate attempt-lineage audit. It is not an attempt-complete table; row-to-trajectory provenance still has gaps.
- **Static candidate scoring:** 12 checkpoint hashes × 639 candidate geometries = 7,668 predictions. This is one static task over candidates, not a model-by-many-task performance grid. The present DUAL-X bridge has exact rows for two hashes only: 1,278. Export SHA-256: `557d8ba52c801ccb5a7a78b07d4f8abfcf46961ff17b8d9b4a7b45e32030e760`.
- **DUAL-X v14:** 414 design cells (23 × 18) and 80 run rows belong to DUAL-X. They are not MigrationBench result rows. Twenty-three hashes match the frozen MB R1 registry, but only seven are present in the current R4 canonical model table; hash identity does not prove result reuse.
- **NEB bridge:** 18 model-by-path summaries over six hashes (15 primary, 3 diagnostic). These remain path/protocol-specific.
- **Manuscript mapping:** The submitted Figure 5 crosswalk points to the 15 × 8 processed slice. The 7,668 static predictions lack a reviewed row-level manuscript claim/panel crosswalk. A 639-row claim inventory is also mixed-grain and must not be read as a performance matrix.

## SQL modeling direction

Maintain separate source-native tables for `model_checkpoint` (exact hash plus identity evidence), `task_protocol` (project, task, protocol version), `design_cell` (planned/status only), `evaluation_attempt` (one run/retry with attempt ID, model and input hashes, seed/job/status), `metric_observation` (value/unit, processing version, result artifact hash), and `publication_cell_link` (release, figure/table cell, processed-row hash, source result IDs). Keep coverage rows as coverage metadata. Preserve failed, diagnostic, planned, missing, and unlinked states explicitly; do not infer attempts from checkpoint identity or fill blank measurements by implication.

## Pinned source identities

- R4 overlay `/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_lineage_overlays/20261125_r4/migrationbench_r1_lineage_overlay_20261125_r4.sqlite` — SHA-256 `d033f47913eaf33d61a2ce7b0b0281b90b385d23190890a97f477e2f901896d7`.
- DUAL-X v14 SQLite — SHA-256 `a7e46f4bdf5ae134ab501b308dbfa987348d4a15201503297edd540c954e6fa4`.
- Figure 5 source TSV and SQLite hashes are listed above.

## Completion gate

A measured matrix cell is complete only when exact model hash, task/protocol version, evaluation attempt, input and output/log hashes, processing version, acceptance/failure state, and manuscript cell (where applicable) are linked. The current dataset does not meet this gate for every model/task cell.
