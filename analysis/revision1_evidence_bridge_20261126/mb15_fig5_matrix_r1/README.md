# Revision 1 Figure 5 matrix snapshot R1

This read-only SQLite snapshot contains the 120 source records in the committed Figure 5 corrected input: **15 model rows × 8 submitted Figure 5 task cells**. This is a measured Figure 5 data slice, not the complete MigrationBench MACE model-by-task matrix.

## Provenance

- Source: `paper/revision1/data_processed/current_release/fig5_corrected_input.tsv` at MigrationBench commit `8e50d5adb56087ce388a1aa593150d14fd70c66f`.
- Source TSV SHA-256: `ee9d43cbb4f6141fb50730a073b25d0a71f6963379ab54614ebe4b17a5019ead`.
- Snapshot: `/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_matrix_snapshots/20261126/mb15_fig5_matrix_r1.sqlite`, pinned in `snapshot_manifest.json`.

## Status counts and evidence limits

- 40 `PRIMARY_EVIDENCE` rows.
- 5 `DIAGNOSTIC_ONLY` rows (including nonconverged evidence).
- 75 `SUPPORTING_EVIDENCE` rows.
- All source rows and their original status labels remain present. 24 rows do not carry a model SHA in this input; these stay NULL and cannot be joined by checkpoint identity from this table alone.

Queries `QF501`–`QF504` return source cells, status counts, per-model task coverage, and provenance pins. Model SHA joins establish checkpoint identity only. These Figure 5 aggregate rows do not by themselves provide the producing attempt, raw input/output files, trajectory hashes, or claim/figure audit chain.
