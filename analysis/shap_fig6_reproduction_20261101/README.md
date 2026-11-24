# Fig. 6 SHAP reproduction audit

Date: 2026-11-01

## Located source chain

- Analysis script: `/Users/alina/Project/migrationbench-paper-repro/scripts/shap_analysis.py`
- SOAP label map: `/Users/alina/Project/migrationbench-paper-repro/scripts/shap_feature_labels.py`
- Surviving plotting wrapper: `/Users/alina/Project/migrationbench-paper-repro/scripts/plot_shap_fig6.py`
- Compact XYZ: `source_data/sampled_trajectory_Cr2_temp.xyz`
- Three-model prediction CSV: `source_data/sampled_trajectory_Cr2_temp_predictions_with_binding1.csv`
- Locked revised ranking table: `locked_source_tables/shap_top_features_revised.csv`
- Locked revised CV table: `locked_source_tables/shap_surrogate_r2_revised.csv`

The attached PDF and the paper-reproduction copy are byte-identical:
`d3c6ac175f47471cdfa50f05f1bf13f6acb92380315eaff6511869fe8a335dc6`.

## What was reproduced

The original analysis script was rerun from the compact 150-frame/450-row
inputs. It completed and wrote `recomputed/`, including the aligned feature
table, revised top-feature table, and CV metrics.

The historical two-panel plotting source that generated the locked PDF was not
present in the surviving repository. The existing `plot_shap_fig6.py` is a
short simplified summary wrapper and does not generate the attached figure.
Therefore `reproduce_fig6.py` is an explicit `RECONSTRUCTION`: it reads the
locked revised ranking table, uses the feature orders recovered from the PDF,
and renders panel (a), panel (b), and the combined figure. It does not claim
byte-identical historical generation.

Outputs:

- `locked_reconstruction/fig6_panel_a_reconstructed.pdf` and `.png`
- `locked_reconstruction/fig6_panel_b_reconstructed.pdf` and `.png`
- `locked_reconstruction/fig6_shap_reproduced_from_locked_table.pdf` and `.png`
- `locked_reconstruction/reproduction_manifest.json`

## Commands

Run the original analysis pipeline:

```bash
/Users/alina/Project/migrationbench-paper-repro/.venv/bin/python \
  /Users/alina/Project/migrationbench-paper-repro/scripts/shap_analysis.py \
  --xyz analysis/shap_fig6_reproduction_20261101/source_data/sampled_trajectory_Cr2_temp.xyz \
  --predictions analysis/shap_fig6_reproduction_20261101/source_data/sampled_trajectory_Cr2_temp_predictions_with_binding1.csv \
  --output analysis/shap_fig6_reproduction_20261101/recomputed
```

Render the locked-table reconstruction:

```bash
/Users/alina/Project/migrationbench-paper-repro/.venv/bin/python \
  analysis/shap_fig6_reproduction_20261101/reproduce_fig6.py \
  --ranking analysis/shap_fig6_reproduction_20261101/locked_source_tables/shap_top_features_revised.csv \
  --output-dir analysis/shap_fig6_reproduction_20261101/locked_reconstruction
```

## Reproducibility boundary

The current-environment rerun produced CV (R^2) values 0.9705, 0.8766, and
0.9811 for FT-600K, FT-MultiT, and Scratch. The locked paper table contains
0.9819, 0.8756, and 0.9730. The source-lineage note already records this
discrepancy; it is preserved here rather than overwritten or silently merged.
