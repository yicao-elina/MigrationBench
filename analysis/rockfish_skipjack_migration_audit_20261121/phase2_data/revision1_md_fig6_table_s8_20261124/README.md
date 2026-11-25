# Revision 1 MD coordinate evidence bundle

Captured 2026-11-24 under `/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_md_fig6_table_s8_20261124/`. This selection is tied to the manuscript `RESULTS_INDEX.tsv` entries for Fig. 6 (current 15-model MD feature data) and Table S8 / R7 MD continuation outcomes. It contains only the six completed grouped-A2 FT-600K / FT-MultiT continuation jobs with run manifests, status records, metrics, continuation ledgers, continuation-segment coordinates, the source smoke checkpoint coordinate file, and the final latest coordinate checkpoint for each continuation.

- 6 run directories; 49 payload files; 2320577507 bytes copied and verified by size + SHA-256.
- 6 coordinate trajectory segments; 1917017196 bytes.
- 6 final coordinate checkpoints (`latest_restart.extxyz`); 1922133 bytes. Each checkpoint comment step was checked against the run status `last_step`.
- The source smoke checkpoint is also retained because the continuation starts at step 220,000; it is separately named `source_smoke_latest_restart.extxyz`.
- Status outcomes: FT600K_A2_seed234 reached step 1,000,000/1,000,000. The other five stopped at `scientific_threshold` before target (exact last steps are in `RUN_SUMMARY.tsv`). These outcomes remain provisional and must not be recast as six 1 ns completions.
- `run_manifest.json` records model/input/checkpoint identities and source paths. The bundle verified model, shared input, and source-checkpoint hashes against those manifests. Six unique per-seed model payloads and the one shared initial structure are included once under `shared/`; SHA-based deduplication was checked against the existing four-checkpoint archive and the original 42 payload records.
- Source run directories remain intact. No restart/WFC files were copied. No MLIP performance or scientific validity is implied by copy verification.
- Exact file source/destination paths, sizes and hashes: `ARCHIVE_MANIFEST.tsv`; per-run outcome and model/input/source/final-checkpoint lineage: `RUN_SUMMARY.tsv`.

## Model weights and shared initial structure

The six run manifests reference six per-seed model checkpoint weights, and all six runs share one 2050-atom initial structure. Their source SHA-256 values were checked against the run manifests. Content was deduplicated against the four-checkpoint archive and the existing MD bundle; only new unique payloads were copied. New copied payloads: 7 files / 399227774 bytes. See `RUN_SUMMARY.tsv` for each run’s resolved archive path and dedup status; each copied file is recorded in `ARCHIVE_MANIFEST.tsv`.

Current payload count after model/input organization: 49 files / 2320577507 bytes.
