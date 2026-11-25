# Task-conditioned distribution atlas (2026-10-18)

This additive package contains the revised distribution diagnostic requested
for the reviewer response. The main atlas is intentionally limited to two
plot-native configuration-space panels; the RMSE history is a separate figure:

1. grouped train/validation/test SOAP/PCA distributions with two-dimensional
   density contours and marginal distributions;
2. MLIP-generated 600 K MD SOAP/PCA distributions overlaid on the grouped
   training density.

The separate four-model RMSE figure reconstructs the original Rockfish
validation histories from the native JSONL training logs, with one column per
family and energy/force rows. It is not mixed into the configuration-space
atlas because it has a different evidence grain and is descriptive single-run
training-history evidence.

The binary stability panel and the old in-figure diagnosis text have been
removed. Failure/CAR evidence is rendered separately in
`fig_failure_stage_grouped.*` so that a collapse trajectory is not confused
with a generic distribution shift.

Each atlas panel is rendered independently with the shared `panel_common.py`
recipe before composition. The standalone panel files are the preferred
manuscript-facing assets when a stacked composite would make labels or
legends too small; the composed atlas remains available as a reviewer-facing
overview.

## Main outputs

- [revised atlas PDF](figures/fig_task_conditioned_distribution_atlas.pdf)
- [revised atlas PNG](figures/fig_task_conditioned_distribution_atlas.png)
- [revised atlas SVG](figures/fig_task_conditioned_distribution_atlas.svg)
- [standalone panel A PDF](figures/fig_task_conditioned_panel_a.pdf)
- [standalone panel C PDF](figures/fig_task_conditioned_panel_c.pdf)
- [original four-model RMSE PDF](figures/fig_original_training_rmse_four_model.pdf)
- [standalone failure figure](figures/fig_failure_stage_grouped.pdf)
- [atlas caption](fig_task_conditioned_distribution_atlas_caption.tex)
- [grouped reference embedding](derived/grouped_reference_embedding.csv)
- [MD distribution points](derived/md_distribution_points.csv)
- [static error points](derived/static_error_points.csv)
- [Rockfish RMSE checkpoints](derived/rockfish_training_rmse_eval.csv)
- [Rockfish RMSE provenance manifest](derived/rockfish_training_rmse_manifest.json)
- [source inventory](tables/source_inventory.tsv)
- [claim boundary matrix](tables/claim_boundary_matrix.tsv)
- [v2 release manifest](task_conditioned_distribution_atlas_v2_manifest.json)

## Rebuild

The grouped SOAP/PCA fit is computed on Rockfish because the reproducible
ASE/DScribe environment is installed there. It uses the same descriptor
definition as the audit: Cr/Sb/Te, periodic SOAP, 5.0 Angstrom cutoff,
`n_max=4`, `l_max=4`, `sigma=0.5`, and inner averaging. The PCA fit is made on
the grouped training split only. `compute_soap_pca.py` records the transform
and the per-frame status. The standalone `plot_task_conditioned_panel_a.py`
and `plot_task_conditioned_panel_c.py` scripts render the configuration-space
panels using Plot Atlas. `parse_rockfish_training_rmse.py` extracts the native
JSONL validation checkpoints and `plot_original_training_rmse_four_model.py`
renders the separate RMSE history. `compose_task_conditioned_distribution_atlas.py`
then composes the two configuration-space recipes into the stacked overview.
`plot_distribution_atlas_v2.py` remains a
compatibility entry point for the atlas renderer; `plot_failure_stage_grouped.py`
renders the separate failure diagnostic.

The four grouped reference files are trajectory-level partitions: 1,786
train frames, 311 validation frames, and 267 test frames, with whole source
AIMD trajectories assigned to one partition. The fixed-path pathway-1--4
images are retained as a separate marker class because they are 61-atom
images rather than grouped AIMD cells.

## Evidence boundaries

- The reference PCA space is current grouped-split evidence, not the retained
  historical 521-row embedding.
- The archived static-error point table remains available for audit, but it is
  not the manuscript-facing RMSE figure. The manuscript-facing RMSE figure
  uses 87 original `mode=eval`, default-head checkpoints from four Rockfish
  run-123 logs. These curves are descriptive single-run validation histories,
  not grouped held-out test errors or seed uncertainty.
- MD SOAP/PCA points are MLIP-generated configuration descriptors, not DFT
  errors. A 1.2 Angstrom short-contact precheck prevents DScribe from
  allocating unbounded neighbor environments on collapse frames; those rows
  remain in the raw status output as `short_contact_skipped`.
- The continuation campaigns use the common MD velocity seed. Model/checkpoint
  branches therefore indicate model variation, not independent velocity
  replicas or calibrated uncertainty.
- The separate failure figure uses the audited 639-candidate MLIP-only
  diagnostic. Its committee spread is a disagreement signal, not a DFT error
  or calibrated uncertainty.
- Neither figure defines a universal model ranking. The intended claim is
  task-conditioned model selection.
