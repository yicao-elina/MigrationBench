# Full-point AIMD parity and frame-level RMSE package

This is an additive follow-up to `analysis/rmse_train_test_20261025`. The previous package contained dataset-level RMSE summaries and per-frame errors only. This package regenerates the paired DFT/reference and MACE/MLIP predictions for all four model families, three training seeds, and train/test splits.

Outputs:

- `raw/rockfish_parity/*__parity.npz`: full energy-per-atom and force-component reference/prediction arrays.
- `raw/rockfish_parity/*__per_frame.csv`: frame-level energy absolute error and force RMSE, with reference and prediction energy values.
- `figures/fig_parity_test_2x4.*`: publication-facing test-only parity panels, arranged as energy × 4 models over force × 4 models, with R².
- `figures/fig_parity_train_test_common_2x4.*`: train/test parity with low-saturation gray training context, colored test overlays, common 600K/MultiT energy axes, one common force axis across all four models, and energy insets that retain the gray training context.
- `figures/fig_parity_r2_4x2.*`: superseded train/test parity layout retained for audit history.
- `figures/fig_rmse_frame_distributions.*`: per-frame distributions with all points, box summaries, and global-RMSE diamonds; all four energy panels share one global log-y range, and all four force panels share one global log-y range.

The reference data are the same hash-checked grouped-A2 train/test XYZ files used in the previous evaluation. The MACE evaluator runs on Rockfish and writes model/data hash evidence before prediction. Training JSONL logs were inspected separately and contain epoch-level metrics, not paired per-point predictions.

Quality/provenance records are in `QUALITY_AUDIT.tsv`, `MODEL_REGISTRY.tsv`, `CLAIM_EVIDENCE_MATRIX.tsv`, `FAILURE_EXCLUSION.tsv`, `GAP_REPORT.md`, and `RELEASE_MANIFEST.json`. The package status is `DIRECT_EVAL_OK; LINEAGE_SUPPORTED; PUBLICATION_PROVISIONAL` pending independent scientific/visual review.
