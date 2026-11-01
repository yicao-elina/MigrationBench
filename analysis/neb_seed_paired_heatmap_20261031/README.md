# Fixed-path NEB seed-paired heatmap

This bundle turns the complete current `4 model families × 3 training seeds ×
2 fixed DFT paths × 10-image grid` table into publication-ready, editable figures.  It
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

The second attachment is a clustered heatmap.  Its dendrogram merely reorders
the rows/columns so that similar profiles sit together.  For this project, the
row tree is a *descriptive error-signature similarity tree*, not a causal
training genealogy.  The full fixed-image error vectors, rather than three
summary scores, provide its features.

## Delivered figures

The renderer produces the following in PNG, editable PDF, and editable SVG.

- `fig_reference1_bubble_matrix_synthetic`: visual reconstruction of the first
  reference grammar with deterministic synthetic values.
- `fig_reference2_clustered_heatmap_synthetic`: visual reconstruction of the
  second reference grammar with deterministic synthetic values.
- `fig_neb_seed_performance_heatmap`: one cell per model-seed × route; numbers
  are MAE in eV, while blue/red fill is route-standardized MAE for legibility.
- `fig_neb_seed_error_lineage_clustermap`: row-clustered signed error profiles
  across all fixed images, with route blocks retained in physical order.
- `fig_neb_seed_error_similarity_bubble_matrix`: Figure-1-style correlation
  map of the same full error signatures.

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

The complete current four-family seed set is recovered from
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

## Rebuild

From the repository root (with the project environment active):

```bash
PYTHONPATH=src python analysis/neb_seed_paired_heatmap_20261031/scripts/build_campaign_input.py \
  --campaign-root /Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/sliding_four_model_campaign_20261024 \
  --output analysis/neb_seed_paired_heatmap_20261031/raw_input/NEB_FIXED_PATH_CAMPAIGN_20261024.tsv

PYTHONPATH=src python analysis/neb_seed_paired_heatmap_20261031/scripts/render_neb_seed_paired_heatmaps.py \
  --input analysis/neb_seed_paired_heatmap_20261031/raw_input/NEB_FIXED_PATH_CAMPAIGN_20261024.tsv \
  --output-dir analysis/neb_seed_paired_heatmap_20261031/figures \
  --processed-dir analysis/neb_seed_paired_heatmap_20261031/processed \
  --expected-paths 2 \
  --baseline-family 'Scratch-MultiT'
```

The checked-in campaign input is the real four-family, two-route table built
by `build_campaign_input.py`; no `--allow-synthetic-demo` flag is used for
this render.  The synthetic fixture and reference-style demos remain only as
isolated visual smoke-test assets and are explicitly named as synthetic.

## Scientific boundary

The figures compare only `FIXED_PATH_SINGLE_POINT` energies on retained DFT
images.  They must not pool these scores with `SELF_RELAXED_ENDPOINT_ONLY_NEB`
barriers, nor use `COMPLETED` or finite MLFF output as a DFT-validation gate.
