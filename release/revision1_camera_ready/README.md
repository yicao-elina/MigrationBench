# Revision 1 camera-ready evidence

Executable sources and data behind the Revision 1 manuscript, SI and the
point-by-point response to Reviewer 1 (A1–A7, C8–C13). The compiled
manuscript sources live in `paper/revision1/`; this folder holds the
reproduction scripts, their inputs and the package notes.
`README_zh.md` is the original (Chinese) package note.

## Layout

| Path | Contents |
|---|---|
| `reproducibility/run_shap_full_campaign.py` | C13: SHAP analysis on the common 267-frame test set (12 seed runs, 407 features) |
| `reproducibility/plot_shap_top20_*.py`, `plot_shap_full_campaign.py`, `shap_feature_labels.py` | C13: top-20 figure renderer and feature-label mapping |
| `reproducibility/source_data/common_test_267.xyz`, `reproducibility/source_tables/` | C13 inputs: common XYZ and the 12 aligned per-frame evaluation tables |
| `reproducibility/recompute_feature_metrics.py`, `fig6_current_input.npz` | C11/C12: original-space silhouettes and t-SNE sensitivity on the 927-frame, 6,192-column matrix (the all-zero placeholder column is asserted and removed) |
| `reproducibility/neb_data_collector.py`, `neb_to_extended_xyz.py`, `build_zero_force_training_manifest.py` | C8: fail-closed exporters and the zero-force training audit (same bytes as `examples/` and `scripts/`) |
| `reproducibility/render_fig5_corrected.py` | Corrected Fig. 5 renderer |
| `../../paper/revision1/data_processed/current_release/` | Figure- and claim-specific source tables (MD status/horizons, parity, SHAP, feature audit, zero-force manifest) |
| `REVIEWER_COVERAGE.tsv` | Reviewer comment → manuscript/SI location and retained limitations |
| `READINESS_NOTE.md` | Evidence that remains open (Chinese) |
| `build.py`, `write_response.py`, `source_snapshot/` | Rebuild the five PDFs; regenerate the response from the pre-revision snapshot |

## Reproduce

Reference environment: Python 3.11 with `requirements-current.txt`
(exact versions matter for t-SNE and SHAP; newer scikit-learn/shap give
small numerical drift). Run from this folder:

```bash
uv venv -p 3.11 .venv && VIRTUAL_ENV=.venv uv pip install -r requirements-current.txt
```

```bash
.venv/bin/python reproducibility/run_shap_full_campaign.py --xyz reproducibility/source_data/common_test_267.xyz --source-tables reproducibility/source_tables --output shap_rerun
```

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python reproducibility/recompute_feature_metrics.py --output feature_audit_rerun
```

In the reference environment both runs reproduce
`paper/revision1/data_processed/current_release/` byte-for-byte
(`shap_seed_metrics.csv`, `shap_family_aggregate.csv`,
`shap_seed_feature_importance.csv`, `fig6_current_*.csv`). The SHAP result
is a reproduction of the current 267-frame analysis, not of the earlier
150-frame reconstruction.

Rebuild the PDFs (TeX Live/MacTeX with `latexmk`):

```bash
python3 build.py
```
