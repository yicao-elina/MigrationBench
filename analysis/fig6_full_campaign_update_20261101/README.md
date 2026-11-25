# Figure 6 full-campaign update

This directory is an additive update of the legacy four-panel Figure 6 analysis. It uses the complete 639-candidate static campaign evaluated against 12 MLIP checkpoints from MigrationBench PR #11 at commit `476ca16`.

The source screen contains 7,668 finite, successful candidate/model pairs: 639 candidates x 12 models, with no duplicate candidate/model pair. The four panels retain the original diagnostic intent while using the scientific Plot Atlas formatting contract.

## Panels

| Panel | Content | Format change |
| --- | --- | --- |
| a | Median maximum force by candidate stage and model arm | Log-scaled heatmap with explicit family strip and cell values |
| b | Model force distributions by candidate stage | Family color plus arm hatch; the singleton stable-terminal case is a point |
| c | Candidate/reference PCA embedding | Near-origin inset keeps the candidate cloud readable while retaining the NEB cluster in the main axes |
| d | Training-set distance versus across-model force disagreement | Log-log diagnostic with the 100 eV/A guide line |

Each panel is exported independently as PNG, PDF, and SVG. The 2 x 2 composite is exported in the same three formats. PNG exports are 350 dpi; PDF and SVG remain editable vector outputs.

## Provenance boundary

This is a `RECONSTRUCTION` diagnostic of a static energy/force screen. It is not a 1 ns MD result, a DFT validation, or a causal attribution claim. The three Foundation entries in this historical screen are checkpoint variants rather than training seeds. The three Scratch entries are heterogeneous: two Lmax2 training-seed branches plus a baseline seed-345 branch. FT-600K and FT-MultiT are the only two groups that form same-family training-seed replicas.

The current SQLite registry was checked separately. It contains the six current Fine-tuning seed identities and six current Scratch seed identities, plus active Foundation identities and a historical MPA-0 comparator. The historical compiled OMAT comparator is retained in this figure as a diagnostic input but is explicitly marked `EXCLUDE_FROM_PUBLICATION` in the SQLite registry. Therefore SQLite alignment is intentionally partial for this historical 12-model screen.

The old PHATE/t-SNE reconstruction and the current SHAP campaign are preserved as separate artifacts and are not used as inputs here. In particular, the SHAP concentration plots are not Figure 6 source data.

## Re-run

```bash
python3 analysis/fig6_full_campaign_update_20261101/plot_fig6_full_campaign.py
```

The script validates the candidate count, model count, successful-row count, finiteness, and candidate/model uniqueness before plotting. Input and output hashes are recorded in `figure_manifest_family_grouped_v2.json`.

## Source files

The `source/` directory contains the immutable downloaded copies used by this run, including `MODEL_GROUPING.tsv`, the 12-model result table, candidate manifest, frame features, reference embedding, force-spread table, and the original campaign plotting script. The exact GitHub source commit and URLs are recorded in the manifest.
