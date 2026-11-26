# MD15 attempt lineage scan R01

Read-only, targeted follow-up for all 15 rows in the R02 MD status table. Inspected only paths named by that table and its registries/status/run manifests, plus immediate children of their explicitly named Scratch16 campaign, run, smoke, and logs directories. No `/data` files were read or scanned; no `/scratch4` or other-owner roots were inspected. No source status or remote file was modified.

## Findings

- All 15 R02 rows are preserved verbatim in the table’s original columns.
- Fourteen checkpoint files had a direct Scratch16 path and an expected SHA-256 in a registry or run manifest; all 14 matched. The Foundation MPA-0 checkpoint is named at `/data/...` in its run manifest and was not checked.
- Six FT registered full-run directories exist. Each has a run manifest, continuation ledger, status, metrics, and named segment. Per-model smoke and continuation job IDs are mapped from the scoped campaign logs/preflight records; no attempt UUID appears in manifests or ledgers.
- All three Foundation terminal status/metrics/segment records exist. Their ledgers preserve multi-segment chains (MPA-0: 220000→605000→990000; MP-0b3: 220000→615000; OMAT: 605000→610400→610800). Scoped logs supply candidate job IDs; ledgers do not record those IDs.
- All six Scratch smoke directories contain a model-bound run manifest, status, metrics, and smoke trajectory, with exact job IDs recoverable from scoped logs. Their registered full-run directories are absent, with no full-run status, continuation ledger, or 1 ns trajectory at those paths.
- Raw trajectory SHA-256 values are absent from the direct manifests; trajectory sizes are reported from direct `stat`. Checkpoint hashes were compared only where a supporting registry/run manifest provides the expected digest.

## Inputs and output

- Source table: `/Users/alina/.codex/worktrees/migrationbench-rockfish-canonical-main-20261125/paper/revision1/data_processed/current_release/MD15_STATUS_AUDIT_R02.tsv`
- Source table SHA-256: `89f32a3e5f6587e8ca3652e3b65ca6249d4891601e46fe74fa08ac336c0dd9c3`
- Direct remote stat/hash receipt: `REMOTE_STAT_HASH_RECEIPT.json`
- Row-level evidence: `MD15_ATTEMPT_LINEAGE_R01.tsv`

The added job IDs are lineage candidates recovered from per-model preflight/Slurm logs; they do not alter or reinterpret R02 status labels. Missing links are explicit in the TSV.
