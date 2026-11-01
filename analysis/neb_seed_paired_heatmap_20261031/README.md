# Fixed-path NEB seed-paired heatmap

This bundle turns a complete `4 model families × 3 training seeds × 3 fixed
DFT paths × image grid` table into publication-ready, editable figures.  It
does not equate a self-relaxed MLFF-NEB path with the retained DFT geometry,
and it does not turn Foundation checkpoint variants into training-seed
replicates.

## What the two reference styles encode

The first attachment is a correlation bubble matrix: circle color is the sign
of a correlation, circle area is its magnitude, and asterisks normally denote
a significance test.  It is useful here for **descriptive similarity of full
image-wise error signatures**.  It is not appropriate to correlate the three
route-level MAEs alone: three values cannot support a meaningful correlation
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
`NEB_SEED_MATCHED_FAMILY_EFFECTS.tsv` preserves all three differences; it does
not claim p-values with `n = 3`.

## Input and current readiness

Copy or derive a tidy table that follows
[`raw_input/NEB_FIXED_PATH_INPUT_SCHEMA.tsv`](raw_input/NEB_FIXED_PATH_INPUT_SCHEMA.tsv).
The validator requires identical model/DFT geometry hashes, non-finite checks,
one row per `model_id × path_id × image_index`, three distinct seed labels per
family, and the fixed-path observable.

The local `neb_fixed_vs_self_matrix_20261019` ledger is an excellent source of
the three retained paths and the seed-resolved fixed-path profiles, but it does
**not** currently supply an eligible four-family × three-seed DFT-matched
energy-error table: it has three seeded families, while its Foundation entries
are checkpoint variants with `training_seed = NA`.  Its current DFT references
also record an SCF-warning status.  This bundle therefore ships only a clearly
labelled synthetic fixture for visual/technical QA, not a scientific result.

## Rebuild

From the repository root (with the project environment active):

```bash
PYTHONPATH=src python analysis/neb_seed_paired_heatmap_20261031/scripts/make_synthetic_demo.py \
  --output analysis/neb_seed_paired_heatmap_20261031/fixtures/neb_seed_energy_errors_synthetic.tsv

PYTHONPATH=src python analysis/neb_seed_paired_heatmap_20261031/scripts/render_neb_seed_paired_heatmaps.py \
  --input analysis/neb_seed_paired_heatmap_20261031/fixtures/neb_seed_energy_errors_synthetic.tsv \
  --output-dir analysis/neb_seed_paired_heatmap_20261031/figures \
  --processed-dir analysis/neb_seed_paired_heatmap_20261031/processed \
  --allow-synthetic-demo
```

Replace the fixture with a `dataset_status=VERIFIED` input once the fourth
family and DFT geometry-matched reference profiles are available.  Omit
`--allow-synthetic-demo` for that production render.

## Scientific boundary

The figures compare only `FIXED_PATH_SINGLE_POINT` energies on retained DFT
images.  They must not pool these scores with `SELF_RELAXED_ENDPOINT_ONLY_NEB`
barriers, nor use `COMPLETED` or finite MLFF output as a DFT-validation gate.
