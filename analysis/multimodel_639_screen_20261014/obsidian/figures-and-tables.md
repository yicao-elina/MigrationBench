# Figures and tables

## Main candidate figure

![[../figures/fig_multimodel_639_overview.png]]

Panel mapping:

- **a** — median static maximum force by source stage and model checkpoint;
- **b** — stage-wise force distributions grouped by model family;
- **c** — candidate, training/validation/test and NEB SOAP-PCA embedding;
- **d** — 12-model force range versus distance to the DFT training distribution.

The panel-a stage ordering is the candidate-selection logic: early stable → stable pre-onset → precursor → threshold/terminal, with the 1000 ps stable target retained as a separate control.

## Data tables

- `../analysis/model_stage_summary_12models.csv` — model × stage force and energy summaries;
- `../analysis/candidate_force_spread_12models.csv` — per-candidate 12-model force spread;
- `../analysis/candidate_distribution_by_stage.csv` — candidate/reference descriptor distribution;
- `../analysis/candidate_source_coverage.csv` — source trajectory and run-key coverage;
- `../analysis/reference_distribution_summary.csv` — training/validation/test/NEB embedding summaries.

## Publication status

The figure is `RECONSTRUCTION` in `../figures/figure_manifest.json`: it is newly rendered from verified processed data using the Plot Atlas script and is not claimed as exact parity with a historical figure.
