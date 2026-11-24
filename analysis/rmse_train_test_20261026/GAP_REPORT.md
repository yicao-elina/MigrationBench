# Data and interpretation report

## What was found

The earlier package contained dataset-level RMSE summaries and frame-level error values, but not paired `DFT/reference` and `MLIP/prediction` arrays. The Rockfish training JSONL logs likewise contain epoch/checkpoint metrics rather than the paired pointwise values required for a parity plot. A search of the relevant model-training output locations did not recover a complete paired prediction matrix.

Therefore the parity data in this additive package were regenerated directly on Rockfish from the hash-checked model files and the same grouped-A2 train/test XYZ files. This is a direct re-evaluation of the released model/data lineage, not a claim that the historical training process saved these pairs.

## Figure definitions

- `fig_parity_test_2x4`: the publication-facing parity layout shows only the test split, with energy in the first row and force in the second row. All three seeds are overlaid for each model. Every test energy frame and force component is retained in the source NPZ files; R², RMSE, MAE, and bias are computed from the full arrays. The raster/vector display is deterministically limited to at most 12000 points per model/observable only for legibility.
- `fig_parity_train_test_common_2x4`: the updated comparison view restores train points as low-saturation gray context, overlays colored test points, orders 600K before MultiT, shares energy x/y limits within the two 600K columns and within the two MultiT columns, and uses one common force x/y range across all four models. Energy insets retain a visibly gray training context while keeping the local test-range view.
- `fig_parity_r2_4x2`: the earlier train/test layout is retained as audit history; it is superseded for the publication-facing parity panel because the full train energy range compressed the test cloud.
- `fig_rmse_frame_distributions`: every evaluated frame is plotted. Energy is the absolute per-atom error; force is the frame-level force-component RMSE. Box summaries and open diamonds show the distribution and the global dataset RMSE, respectively. The four energy panels use one shared global log-y range, and the four force panels use one shared global log-y range for direct cross-model comparison.
- The older seed-level figure is correctly interpreted as a seed comparison: each marker was one aggregate RMSE per model/seed/split, not one AIMD point.

## Scientific caveats retained in the release

The 600K test energy ranges are narrow relative to their absolute energy baseline. Consequently, a nonzero energy RMSE can produce a negative test R²; this is a variance-sensitive diagnostic and should not be read as a universal model ranking. No frames, force components, seeds, or models were removed to improve appearance. The large Scratch-MultiT seed-123 test-force tail remains visible and is recorded in `FAILURE_EXCLUSION.tsv` as a retained observation.

Two failed Slurm wrapper attempts are preserved under `raw/failed_wrapper_lc_all/` and `raw/failed_wrapper_mace_polar/`. They were shell/environment failures before valid scientific evaluation, and are excluded from the scientific result set. The final job 32511152 produced 24/24 `PARITY_EVAL_OK` evaluations with model and data SHA256 checks passing.

## Release status

`LINEAGE_SUPPORTED`, `DIRECT_EVAL_OK`, and `PUBLICATION_PROVISIONAL`. The figure sources, raw paired arrays, summary CSV, audit tables, and release hashes are included so that an independent review can reproduce the plotted values.
