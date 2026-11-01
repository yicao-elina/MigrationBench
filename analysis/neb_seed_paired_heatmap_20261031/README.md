# NEB seed-paired heatmap atlas

This bundle turns the current seed-resolved NEB evidence into publication-ready,
editable figures. The sliding block contains the complete current
`4 model families × 3 training seeds × 2 fixed DFT paths × 10-image grid`.
The migration block is a separate campaign source with three routes and two
observables; it is not silently treated as the same checkpoint universe. It
does not equate a self-relaxed MLFF-NEB path with the retained DFT geometry,
and it does not turn Foundation checkpoint variants into training-seed
replicates.

## What the two reference styles encode

The first attachment is a correlation bubble matrix: circle color is the sign
of a correlation, circle area is its magnitude, and asterisks normally denote
a significance test.  It is useful here for **descriptive similarity of full
image-wise error signatures**.  It is not appropriate to correlate a small
number of route-level MAEs alone: a few values cannot support a meaningful correlation
or p-value.  The real figure therefore omits asterisks because NEB image errors
are also serially correlated along a path.

The second attachment is a clustered heatmap. Its dendrogram merely reorders
the rows/columns so that similar profiles sit together. The revised real
error figure uses an explicit high-to-low mean-absolute-error order instead,
because that is the requested summary statistic; the synthetic reference
demo retains the dendrogram grammar. The full fixed-image error vectors,
rather than three summary scores, provide the real figure's features.

## Delivered figures

The renderer produces the following in PNG, editable PDF, and editable SVG.

- `fig_reference1_bubble_matrix_synthetic`: visual reconstruction of the first
  reference grammar with deterministic synthetic values.
- `fig_reference2_clustered_heatmap_synthetic`: visual reconstruction of the
  second reference grammar with deterministic synthetic values.
- `fig_neb_seed_performance_heatmap`: one total matrix containing a sliding
  fixed-path MAE block, a migration fixed-path barrier-proxy block, and a
  migration IDPP self-relaxed barrier-proxy block. Values are printed in eV;
  the main fill is standardized within each column for visual comparison.
  Missing campaign/model combinations are grey `NA`, never zero-filled.
- `fig_neb_seed_error_lineage_clustermap`: signed image-wise sliding errors
  sorted from high to low by each model-seed's mean absolute error. The top
  strip is the overall mean absolute error per image and the right strip is
  the row statistic used for sorting. This is a descriptive error map, not a
  causal training lineage.
- `fig_neb_seed_error_similarity_bubble_matrix`: Figure-1-style bubble matrix
  of the same full error signatures. Raw off-diagonal correlations are
  min-max normalized to `[0, 1]` for display; diagonal values are set to 1.
  Circle area and color use that normalized similarity, with no significance
  stars.

## Pairing rule

For a family comparison, `seed 123` is paired only with `seed 123` from the
other family on the same path and the same fixed DFT image grid.  The output
`NEB_SEED_MATCHED_FAMILY_EFFECTS.tsv` preserves all route-wise differences; it does
not claim p-values with `n = 3`.

## Input and current readiness

Copy or derive a tidy table that follows
[`raw_input/NEB_FIXED_PATH_INPUT_SCHEMA.tsv`](raw_input/NEB_FIXED_PATH_INPUT_SCHEMA.tsv).
The validator requires identical model/DFT geometry hashes, non-finite checks,
one row per `model_id × path_id × image_index`, three distinct seed labels per
family, and the fixed-path observable.

The complete current sliding four-family seed set is recovered from
`analysis/sliding_four_model_campaign_20261024/results/tables/fixed_profiles.csv`:
`Scratch-MultiT`, `Scratch-600K`, `FT-600K`, and `FT-MultiT`, each with seeds
123/234/345.  Foundation checkpoints are excluded.  The complete common
fixed-path coverage in this campaign is two routes: `pristine` and
`Cr-in-gap`; the third NEB route does not have the same four-family × three-seed
coverage and is therefore not pooled into this release.

The DFT reference is the campaign's retained `1.bulk` / `2.cr-in-gap` table
and carries `COMPLETED_BUT_SCF_WARNING`; model outputs are finite diagnostic
supporting evidence.  The input is consequently labelled
`PROVENANCE_PARTIAL`: these are real campaign-derived scores, but not an
unconditional publication-grade DFT validation claim.

### Total performance matrix provenance

The total matrix is built by `scripts/build_total_performance_metrics.py` from
the sliding route metrics and the migration `TIER1_RESULT_MATRIX.tsv`.

- Sliding columns are DFT-matched fixed-path `MAE_eV` for `pristine` and
  `Cr-in-gap`.
- Migration columns are `barrier_proxy_eV` for `in-gap diffusion`, `gap-to-QL`,
  and `within-QL`, shown separately for `FIXED_PATH_SINGLE_POINT` and
  `SELF_RELAXED_ENDPOINT_ONLY_NEB`.
- Self-relaxed summary rows select `initialization_method == IDPP`. The
  alternative `MIC_LINEAR` rows remain in the source audit but are not pooled.
- The migration campaign contains `From-scratch`, `FT-600K`, and `FT-MultiT`
  seeds, but no `Scratch-600K` model rows. The total table includes those
  expected Scratch-600K slots as `NOT_AVAILABLE_IN_SOURCE_CAMPAIGN` and the
  figure renders them as `NA`.
- Migration checkpoint hashes are not assumed identical to the sliding
  campaign hashes. Therefore the figure uses separate campaign-labelled rows
  and does not claim a single cross-task checkpoint identity.

Migration `barrier_proxy_eV` values are not DFT errors. They are a compact
route-level diagnostic/proxy and must not be interpreted as the sliding MAE
scale.

## Rebuild

From the repository root (with the project environment active):

```bash
PYTHONPATH=src python analysis/neb_seed_paired_heatmap_20261031/scripts/build_campaign_input.py \
  --campaign-root /Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/sliding_four_model_campaign_20261024 \
  --output analysis/neb_seed_paired_heatmap_20261031/raw_input/NEB_FIXED_PATH_CAMPAIGN_20261024.tsv

PYTHONPATH=src python analysis/neb_seed_paired_heatmap_20261031/scripts/build_total_performance_metrics.py \
  --sliding-metrics analysis/neb_seed_paired_heatmap_20261031/processed/NEB_MODEL_SEED_ROUTE_METRICS.tsv \
  --migration-matrix /Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/task3_endpoint_only_neb/neb_fixed_vs_self_matrix_20261019/derived_r01/TIER1_RESULT_MATRIX.tsv \
  --output analysis/neb_seed_paired_heatmap_20261031/processed/NEB_TOTAL_PERFORMANCE_METRICS.tsv

PYTHONPATH=src python analysis/neb_seed_paired_heatmap_20261031/scripts/render_neb_seed_paired_heatmaps.py \
  --input analysis/neb_seed_paired_heatmap_20261031/raw_input/NEB_FIXED_PATH_CAMPAIGN_20261024.tsv \
  --output-dir analysis/neb_seed_paired_heatmap_20261031/figures \
  --processed-dir analysis/neb_seed_paired_heatmap_20261031/processed \
  --total-metrics analysis/neb_seed_paired_heatmap_20261031/processed/NEB_TOTAL_PERFORMANCE_METRICS.tsv \
  --expected-paths 2 \
  --baseline-family 'Scratch-MultiT'
```

The checked-in campaign input is the real four-family, two-route table built
by `build_campaign_input.py`; no `--allow-synthetic-demo` flag is used for
this render.  The synthetic fixture and reference-style demos remain only as
isolated visual smoke-test assets and are explicitly named as synthetic.

## Scientific boundary

The sliding error figures compare only `FIXED_PATH_SINGLE_POINT` energies on
retained DFT images. The total performance matrix intentionally places
migration fixed-path and `SELF_RELAXED_ENDPOINT_ONLY_NEB` barrier proxies in
separate labelled blocks; those values are not pooled with sliding MAE and
are not DFT-validation errors. Neither `COMPLETED` nor finite MLFF output is
itself a DFT-validation gate.
