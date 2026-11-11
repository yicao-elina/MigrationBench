# 639 MLIP result data-quality audit

Status: `VERIFIED`. This audit checks the reconciled static-evaluation table before canonical-family merging.

## Dataset and grain

The intended grain is one row per candidate/model pair. The table has 1917 rows, 639 candidates, and 3 model IDs.
Candidate manifest and categorization each contain 639 and 639 unique candidates, respectively.

## Checks

- Duplicate candidate/model pairs: 0
- Exact duplicate rows: 0
- Successful rows: 1917; failed rows: 0
- Finite force rows: 1917
- Missing candidate IDs: 0; unexpected IDs: 0

## Findings and risk

The canonical candidate/model grain is complete for the historical 639×3 table when the status is VERIFIED. The wide reconciled CSV also retains merge-carried `_x`/`_y` and legacy source columns; these are schema contamination, not evidence of missing core evaluations. Downstream analysis must use the canonical candidate_id, model_id, candidate_group, window, energy_eV, force metrics, and evaluation_status columns, and the issue must be repaired or explicitly carried into the final manifest.

The 639 candidates are mechanism-focused, event-driven diagnostics rather than an unbiased random test set. Their stage labels come from the candidate manifest and must not be reinterpreted as newly observed DFT onset labels.

## Reproduction

`python /Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/canonical_model_matrix_20261016/data_quality_audit_20261016.py --results analysis/from_scratch_failure_attribution_20261011/mlip_candidate_campaign_20261013/results_reconciled/mlip_candidate_results_all.csv --candidate-manifest analysis/from_scratch_failure_attribution_20261011/mlip_candidate_campaign_20261013/candidate_manifest_639.csv --categorization analysis/from_scratch_failure_attribution_20261011/candidate_categorization.csv --output analysis/canonical_model_matrix_20261016/data_quality_audit_20261016`
Script SHA256: `19bc0dfe9ddfb5d458ab5d33458280fbfc0006acd3bc3cf5c8ca897d0d98eb03`
