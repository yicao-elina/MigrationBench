# NEB seed-paired heatmap atlas

This bundle turns the current seed-resolved NEB evidence into publication-ready,
editable figures. The compact total-performance figure has seven model rows:
four trained families aggregated over seeds 123/234/345 and three matched
Foundation checkpoints (`MP-0B3`, `OMAT-0`, `MPA0`). Each trained-family
circle is split into three seed wedges. A companion seed-expanded figure has
15 rows: the 12 trained model-seed units plus the three single-checkpoint
Foundation rows. The migration columns are barrier errors against the
canonical DFT forward barrier, not raw barrier proxies.

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
- `fig_neb_seed_performance_heatmap`: one total matrix containing migration
  fixed-path barrier-error, migration IDPP self-relaxed barrier-error, and
  sliding fixed-path MAE blocks, in that order. Color is a column-normalized
  high-is-good score; trained-family circles are split into seed 123/234/345
  wedges and radius encodes available-seed coverage. Raw model barriers,
  canonical DFT barriers, errors, and scores remain in the processed tables.
- `fig_neb_seed_performance_heatmap_seed_expanded`: the same score grammar with
  one row per training seed/checkpoint (15 rows total), making seed-to-seed
  differences directly inspectable.
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
123/234/345. The same campaign contains three Foundation sliding checkpoints;
their DFT-matched MAE rows are retained in
`raw_input/sliding_foundation_20261024/fixed_metrics_per_model.csv`. The complete
common fixed-path coverage is two routes: `pristine` and
`Cr-in-gap`; the third NEB route does not have the same four-family × three-seed
coverage and is therefore not pooled into this release.

The DFT reference is the campaign's retained `1.bulk` / `2.cr-in-gap` table
and carries `COMPLETED_BUT_SCF_WARNING`; model outputs are finite diagnostic
supporting evidence.  The input is consequently labelled
`PROVENANCE_PARTIAL`: these are real campaign-derived scores, but not an
unconditional publication-grade DFT validation claim.

### Total performance matrix provenance and visual contract

The total matrix is built by `scripts/build_total_performance_metrics.py` from
the sliding route metrics, the migration `TIER1_RESULT_MATRIX.tsv`, the
Scratch-600K fill matrix, and
`raw_input/dft_reference/canonical_dft_barriers.tsv`.

- Sliding columns are DFT-matched fixed-path `MAE_eV` for `pristine` and
  `Cr-in-gap`.
- Migration columns are `abs(model barrier - canonical DFT forward barrier)`
  for `in-gap diffusion`, `gap-to-QL`, and `within-QL`, shown separately for
  `FIXED_PATH_SINGLE_POINT` and `SELF_RELAXED_ENDPOINT_ONLY_NEB`. Fixed rows
  use `forward_barrier_eV`; IDPP self-relaxed rows use `barrier_proxy_eV` as
  the endpoint-only migration barrier estimate.
- Self-relaxed summary rows select `initialization_method == IDPP`. The
  alternative `MIC_LINEAR` rows remain in the source audit but are not pooled.
- The canonical Scratch-600K fill campaign is included under
  `raw_input/scratch_600k_migration_20261101/`. It has all three registered
  seeds and all three routes: fixed-path is 9/9 finite; IDPP self-relaxed is
  7/9 `CONVERGED/PRIMARY_EVIDENCE`. The two nonconverged IDPP cells remain
  `NONCONVERGED/DIAGNOSTIC_ONLY` and render as `NA` in the main matrix.
- Migration checkpoint hashes are not assumed identical to the sliding
  campaign hashes. The compact figure therefore uses model-family rows for
  cross-task comparison, while the raw 24-unit table and source hashes retain
  the task-specific checkpoint identity.
- The final column order is frozen as: migration fixed pathway (`in-gap`,
  `gap→QL`, `within QL`), migration self-relaxed (`in-gap`, `gap→QL`,
  `within QL`), then sliding (`pristine`, `Cr-in-gap`).
- Each column is transformed independently across the 15 seed/checkpoint units
  to a lower-is-better min-max score
  in `[0, 1]`; yellow means high score and purple/blue means low score. This is
  a rank-like visual score, not a claim that MAE and barrier error share one
  physical unit.
- Bubble radius is `sqrt(n_available / n_expected)`: aggregate trained-family
  rows use an arithmetic mean over available seeds, while Foundation rows use
  one checkpoint (`1/1`). Missing or scientifically excluded cells are left
  blank; they are not zero-filled.
- The three Foundation rows are selected because the same checkpoint hashes are
  present in the Foundation sliding campaign and the T1 migration matrix. The
  migration self-relaxed block uses IDPP only and requires
  `CONVERGED/PRIMARY_EVIDENCE`.

The checked-in DFT reference carries `scf_warning=TRUE` and
`EXCLUDE_FROM_PUBLICATION` in its source registry. The figures therefore
preserve the error calculation and provenance but do not, by themselves,
upgrade the current DFT reference to a publication-grade validation claim.

## Rebuild

From the repository root (with the project environment active):

```bash
PYTHONPATH=src python analysis/neb_seed_paired_heatmap_20261031/scripts/build_campaign_input.py \
  --campaign-root /Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/sliding_four_model_campaign_20261024 \
  --output analysis/neb_seed_paired_heatmap_20261031/raw_input/NEB_FIXED_PATH_CAMPAIGN_20261024.tsv

PYTHONPATH=src python analysis/neb_seed_paired_heatmap_20261031/scripts/build_total_performance_metrics.py \
  --sliding-metrics analysis/neb_seed_paired_heatmap_20261031/processed/NEB_MODEL_SEED_ROUTE_METRICS.tsv \
  --migration-matrix analysis/neb_seed_paired_heatmap_20261031/raw_input/foundation_migration_20261019/TIER1_RESULT_MATRIX.tsv \
  --scratch600k-migration analysis/neb_seed_paired_heatmap_20261031/raw_input/scratch_600k_migration_20261101/processed/SCRATCH600K_MIGRATION_RESULT_MATRIX.tsv \
  --dft-reference analysis/neb_seed_paired_heatmap_20261031/raw_input/dft_reference/canonical_dft_barriers.tsv \
  --output analysis/neb_seed_paired_heatmap_20261031/processed/NEB_TOTAL_PERFORMANCE_METRICS.tsv

PYTHONPATH=src python analysis/neb_seed_paired_heatmap_20261031/scripts/build_total_performance_dot_metrics.py \
  --total-metrics analysis/neb_seed_paired_heatmap_20261031/processed/NEB_TOTAL_PERFORMANCE_METRICS.tsv \
  --foundation-migration analysis/neb_seed_paired_heatmap_20261031/raw_input/foundation_migration_20261019/TIER1_RESULT_MATRIX.tsv \
  --foundation-sliding analysis/neb_seed_paired_heatmap_20261031/raw_input/sliding_foundation_20261024/fixed_metrics_per_model.csv \
  --dft-reference analysis/neb_seed_paired_heatmap_20261031/raw_input/dft_reference/canonical_dft_barriers.tsv \
  --output analysis/neb_seed_paired_heatmap_20261031/processed/NEB_TOTAL_PERFORMANCE_DOT_METRICS.tsv \
  --seed-expanded-output analysis/neb_seed_paired_heatmap_20261031/processed/NEB_TOTAL_PERFORMANCE_SEED_EXPANDED_METRICS.tsv

PYTHONPATH=src python analysis/neb_seed_paired_heatmap_20261031/scripts/render_neb_seed_paired_heatmaps.py \
  --input analysis/neb_seed_paired_heatmap_20261031/raw_input/NEB_FIXED_PATH_CAMPAIGN_20261024.tsv \
  --output-dir analysis/neb_seed_paired_heatmap_20261031/figures \
  --processed-dir analysis/neb_seed_paired_heatmap_20261031/processed \
  --total-metrics analysis/neb_seed_paired_heatmap_20261031/processed/NEB_TOTAL_PERFORMANCE_METRICS.tsv \
  --dot-metrics analysis/neb_seed_paired_heatmap_20261031/processed/NEB_TOTAL_PERFORMANCE_DOT_METRICS.tsv \
  --seed-expanded-metrics analysis/neb_seed_paired_heatmap_20261031/processed/NEB_TOTAL_PERFORMANCE_SEED_EXPANDED_METRICS.tsv \
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
migration fixed-path and `SELF_RELAXED_ENDPOINT_ONLY_NEB` barrier errors in
separate labelled blocks; those values are not pooled with sliding MAE.
Neither `COMPLETED` nor finite MLFF output is itself a DFT-validation gate.
