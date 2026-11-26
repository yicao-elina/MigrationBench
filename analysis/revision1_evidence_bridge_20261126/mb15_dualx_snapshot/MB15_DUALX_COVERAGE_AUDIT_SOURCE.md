# MigrationBench 15-model Revision 1 coverage audit

Captured 2026-11-26 from canonical MigrationBench main `fba92f7b1650893ef9aeeef160f76a516009118f`, using the submitted R1 release tables and the read-only DUAL-X v14 exact-hash bridge. This is a read-only audit of the user's personal project scope. It does not read other users' paths or alter scientific data.

## What the 15-model table actually covers

`MD15_STATUS_AUDIT_R02.tsv` contains 15 model-level status rows: 3 Foundation, 3 FT-600K, 3 FT-MultiT, 3 Scratch-600K, and 3 Scratch-MultiT. Every row also has a checkpoint SHA-256 and common-input SHA-256. `MD15_FIGURE_DATA_R06G.tsv` contains 15 figure-data rows keyed to the same model IDs.

The status counts are **3 `TARGET_REACHED_PROVISIONAL`, 1 `TARGET_REACHED`, and 11 `SCIENTIFIC_FAILURE`**. All 15 rows have finite-state audit fields; the table reports 17,725 eligible frames and 0 invalid frames in aggregate. Status category must remain as recorded: the eleven threshold exits are scientific failure modes, while provisional target reach is not the same as full acceptance.

The table is a 15-model MD continuation/status lane, not a complete 15-model × all-task performance matrix. Each row is a model summary, not an enumerated independent-attempt table. It carries no per-model attempt ID, Slurm job ID, raw trajectory SHA-256, or trajectory continuity receipt. The accompanying `md_protocol_manifest.json` describes a six-model FT grouped-A2 campaign (`32468152_[0-5]`) and a common input SHA, but does not close all 15 rows to attempt-level producer manifests. Automatic walltime continuation is part of that protocol, so continuation segments cannot be treated as independent seeds.

## Exact MigrationBench → DUAL-X bridge result

- **15/15** status-table checkpoint SHAs exactly match DUAL-X model checkpoint SHAs and the frozen MB R1 registry. This establishes weight identity at the bridge-export grain; it does not prove each checkpoint file was freshly rehashed on capture day or that two projects' task/results are equivalent.
- **6/15** (the six FT grouped-A2 models) also match rows in MB R4's current canonical model table and have explicit DUAL-X external model link rows. These same six have **18 path-specific NEB rows** (3 paths/model: 15 `CONVERGED / PRIMARY_EVIDENCE`, 3 `FAILURE_MODE / DIAGNOSTIC_ONLY`). Keep path, protocol, attempt, and status separate.
- **2/15** Foundation checkpoints (`MPA0`, `MP0b3`) have **639 prepared static-candidate rows each** (1,278 total). This is a candidate/geometry diagnostic extract, not DUAL-X held-out inference or a complete cross-task matrix.
- **6/15** FT models have historical MD status snapshots in DUAL-X v14, captured 2026-10-25 20:18 UTC. Those six records are timestamped scheduler/scientific observations; `current_status=NOT_ASSERTED`. They are not accepted 1 ns trajectory outputs and do not provide DUAL-X diffusion results. The R02 status table is a separate release artifact and is not linked row-for-row to these job/attempt records.
- The remaining **7/15** have exact checkpoint identity but no result-level MB reuse established in the bridge. `OMAT0` has identity only; six Scratch models also have identity-only status in the current bridge.
- DUAL-X v14 itself has 414 design cells (23 × 18) and 80 run rows; this is not the MB 15-model measured matrix and cannot be substituted for it.

## Provenance and table/figure closure

The R1 release includes the two 15-row model-keyed TSVs. The R1 claim-evidence matrix contains the Table S8/MD claims and records table/object links at mixed grains; current extracted rows mark 15-model values as partial where per-cell source-row hashes and producer attempt identities are absent. The matrix and plot data therefore show the **reported values and their model key**, while a stronger end-to-end chain still needs each exact model row → chosen run/attempt or continuation chain → exact input/checkpoint/raw trajectory → processed row/generator → figure/table cell → submitted manuscript version.

The manifest sidecar staging database does not currently add workflow-run/Slurm/run-result/paper-claim foreign keys; it remains `PROVENANCE_PARTIAL`/`UNVERIFIED`. R4 plus QL25–QL26 addresses campaign archives and QE outputs, not all MD15 trajectory chains. QX01–QX05 expose identity and narrow external evidence without pretending to close those joins.

## Immediate closure tasks

1. For every one of the 15 model rows, identify the canonical run/attempt manifest, continuation segments, Slurm job IDs, trajectory files and exact SHA-256; reconcile duplicates and superseded attempts without deleting evidence.
2. Confirm that each row's chosen run artifact is the one consumed by `MD15_FIGURE_DATA_R06G.tsv`; hash the processed data and figure generator inputs; register exact Table S8/figure-cell links in one additive SQL snapshot.
3. Preserve the dated historical six-row v14 MD snapshot as a historical observation, and separately link the later 15-row R02 status artifact to the actual current run outcomes. Never overwrite historical rows.
4. For DUAL-X reuse, consume the six model-specific NEB rows and two foundation 639-row candidate extracts as typed external evidence; keep MD as status-only until exact trajectories/results are registered.

## Source pins

- Canonical MB main: `fba92f7b1650893ef9aeeef160f76a516009118f`.
- DUAL-X v14 result snapshot: `/scratch16/pclancy3/yi/X-FORCE_data_archive/sqlite_sources/revision1_dualx_20261125_v14/revision1_dualx_results.sqlite`, SHA-256 `a7e46f4bdf5ae134ab501b308dbfa987348d4a15201503297edd540c954e6fa4`.
- MB R4 overlay: `/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_lineage_overlays/20261125_r4/migrationbench_r1_lineage_overlay_20261125_r4.sqlite`, SHA-256 `d033f47913eaf33d61a2ce7b0b0281b90b385d23190890a97f477e2f901896d7`.
- Query interfaces: MigrationBench `analysis/revision1_evidence_bridge_20261126/dualx_exact_bridge_r1/EXTERNAL_EVIDENCE_BRIDGE_DRAFT.sql` QX01–QX05; MB R4 `analysis/revision1_lineage_overlay_r4/queries.sql` QL21–QL26.

### Input artifact hashes

- `MD15_STATUS_AUDIT_R02.tsv`: `89f32a3e5f6587e8ca3652e3b65ca6249d4891601e46fe74fa08ac336c0dd9c3`
- `MD15_FIGURE_DATA_R06G.tsv`: `929c15b6b46d80b4c69afb9a6b6d6e34acf229c684dfcec2957f65e7a571a62e`
- `md_protocol_manifest.json`: `316a8f31cd20a0026c2b7312b894d8ad1c2503d62b4a37472371c285b95eda88`
- `MODEL_IDENTITY_BRIDGE.tsv`: `cca5aaa8774cc4436e9ca4e89c55fce22a25c34482a59a399154fc81b8ec8c89`
- `DUALX_NEB_EXACT_ROWS_V14.tsv`: `599ac2cf3097e9482ed1f41f655045789d3fe78856770bf206cd34913add6bfd`
- `DUALX_STATIC_EXACT_ROWS_V14.tsv`: `557d8ba52c801ccb5a7a78b07d4f8abfcf46961ff17b8d9b4a7b45e32030e760`
- `MB_MD_STATUS_V14.tsv`: `4bda1bd2f8c126f2ee3b3bf5e09e15949480685aa4591c824838175002a508b0`
- `BRIDGE_SNAPSHOT_MANIFEST.json`: `e3ccbb801d26804318cc4399d10cc766688f81183e9734d5704b433253cdd214`

`MD15_DUALX_COVERAGE.tsv` is keyed one row per submitted model and pins the exact SHA joins and reuse class. Its SHA-256 is in `SHA256SUMS`.
