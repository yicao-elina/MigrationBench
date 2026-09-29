# Sec. 3.5 / Fig. 6 SHAP source lineage

Status: canonicalized 2026-09-29.

The sole upstream source for the MigrationBench Sec. 3.5 and Fig. 6 results is
the X-FORCE asset under
`/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/X-FORCE/xforce/`.
The release copies the compact inputs into `data/raw_compact/revision1/shap/`.

| Release asset | Source | Rows/frames | MD5 / SHA256 |
|---|---|---:|---|
| `sampled_trajectory_Cr2_temp.xyz` | X-FORCE `data/shap/` | 150 × 82 atoms | source MD5 `8982c9d7dbcba18d7980257782f114b1` |
| `sampled_trajectory_Cr2_temp_predictions_with_binding1.csv` | X-FORCE `data/shap/` | 450 + header | source MD5 `8281288f64856364c1cb2aa8f3e9b319` |

The 600-row `...with_binding.csv` is a historical four-model input and is not
used here. The `2-concentration` workflow is unrelated doping-scenario analysis.
The Dual-X/ICML ID/OOD/NEB SHAP workflow is also unrelated and is not used for
Sec. 3.5.

The reproducible chain is:

```text
X-FORCE XYZ + three-model prediction CSV
  -> scripts/shap_analysis.py
  -> explicit image_index alignment
  -> 390 SOAP (r_cut=5, n_max=4, l_max=4) + 17 structural features
  -> absolute total-energy-error GBR surrogate
  -> 5-fold CV + TreeSHAP
  -> data/processed/shap_*_revised.csv
  -> scripts/plot_shap_fig6.py -> figures/publication/fig6_shap_revised.pdf
```

The manuscript values are generated only from the revised CSVs. The feature
label map is isolated in `scripts/shap_feature_labels.py`; it follows the
390-feature DScribe layout and keeps labels attached to the explicit feature
index rather than inferring labels from another SHAP workflow.

## Regeneration audit

The compact inputs were independently re-run on 2026-09-29 with the release
script. The run reproduced the expected shape, model set, and frame alignment,
but its CV values were not byte/numerically identical to the already audited
`data/processed/*_revised.csv` values. Therefore the existing revised CSVs and
publication PDF remain the locked paper outputs; the discrepancy is recorded
rather than silently overwriting the paper results. Before claiming bitwise
regeneration, the historical `shap_fig6_regenerate.py` implementation or its
exact feature preprocessing must be recovered and compared line-by-line.

The independent release-script check produced CV $R^2$ values 0.9705,
0.8766, and 0.9811 for FT--600K, FT--Multi--T, and Scratch, respectively,
versus the locked paper values 0.9819, 0.8756, and 0.9730. This is a
reproducibility discrepancy to resolve, not a result to conceal.
