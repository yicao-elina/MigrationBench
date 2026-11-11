# Full-campaign SHAP analysis — 2026-11-01

Status: `COMPLETE_WITH_COMMON_TEST_SCOPE`; figures are rendered for review and remain publication-provisional pending independent scientific/visual review.

This is an additive update of the original Fig. 6 SHAP reconstruction. It uses the current local asset registry and the complete four-family evaluation package, with three training seeds per family:

| Family | Seeds | Test rows used |
|---|---:|---:|
| Fine-tuning · 600K | 123, 234, 345 | 267 each |
| Fine-tuning · Multi-T | 123, 234, 345 | 267 each |
| Scratch · 600K | 123, 234, 345 | 267 each |
| Scratch · Multi-T | 123, 234, 345 | 267 each |

## Scope boundary

The main comparison uses the common 267-frame test trajectory `2L_o3_t3_600K_aimd_1`, with 86 atoms per frame. The Multi-T tables contain 767 rows, but their additional 500 `2L_tetra_Cr4_600K_aimd_1` rows have no paired 600K predictions. Those rows are retained in the copied source tables and explicitly excluded from the cross-family SHAP comparison; they are not silently merged or dropped from provenance.

The common XYZ source is the Rockfish grouped-A2 test file:

`/scratch16/pclancy3/yi/revision1_migrationbench_runs/ft600k_grouped_A2/data_grouped/test.xyz`

Local SHA256: `d49cf4b2ab49fc1e521d314c5828cf5dcaea6853e70cecb2727ffd0678bb27ab`.

## Method

The target is the absolute total-energy error per atom, in `meV atom^-1`, from the per-frame evaluation CSVs. Each model-seed run uses the same 407-feature representation: 390 periodic DScribe SOAP features (`Cr/Sb/Te`, `r_cut=5.0`, `n_max=4`, `l_max=4`, inner average) plus 17 structural features. A final `GradientBoostingRegressor` is fitted per model-seed run and explained with `shap.TreeExplainer`. The plotted values are mean absolute SHAP values; error bars in panel (a) are sample SD across the three training seeds.

For a lightweight generalization smoke check, a fixed 5-fold shuffled K-fold split (`random_state=42`) is evaluated with a smaller gradient-boosting surrogate. This quality smoke is not substituted for the final TreeSHAP estimator and does not change the attribution values.

## Outputs

- `figures/fig_shap_full_campaign.pdf/png/svg`: combined two-panel figure.
- `figures/fig_shap_full_campaign_panel_a.*`: top-20 grouped bar panel, mean +/- SD across seeds.
- `figures/fig_shap_full_campaign_panel_b.*`: top-50 shared-scale heatmap across four model families.
- `results/shap_seed_metrics.csv`: one row per model family and seed, including target range and CV smoke metrics.
- `results/shap_seed_feature_importance.csv`: per-seed feature-level attribution table.
- `results/shap_family_aggregate.csv`: family-level mean and SD across the three seeds.
- `results/shap_values_full_campaign.npz`: frame-aligned SHAP arrays for all 12 runs.
- `results/analysis_manifest.json`: scope, method, source hashes, output contract, and quality gates.
- `figures/figure_manifest.json`: figure hashes, colors, and selected feature lists.

## Rebuild

From `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench`:

```bash
/Users/alina/Project/migrationbench-paper-repro/.venv/bin/python \
  analysis/shap_full_campaign_20261101/run_shap_full_campaign.py \
  --xyz analysis/shap_full_campaign_20261101/source_data/common_test_267.xyz \
  --source-tables analysis/shap_full_campaign_20261101/source_tables \
  --output analysis/shap_full_campaign_20261101/results

/Users/alina/Project/migrationbench-paper-repro/.venv/bin/python \
  analysis/shap_full_campaign_20261101/plot_shap_full_campaign.py \
  --analysis analysis/shap_full_campaign_20261101/results \
  --output analysis/shap_full_campaign_20261101/figures
```

## Scientific note

This is a surrogate attribution of the error target, not a causal attribution of the underlying MACE model. All 12 runs are finite and complete on the common scope, and the surrogate has in-sample `R^2=1` by construction. The 5-fold smoke `R^2` spans approximately `0.927–0.984`; Scratch-MultiT seed 123 is the lowest smoke score and has the largest observed target error range, so seed-level stability should be reviewed before using the figure for a strong mechanistic claim.
