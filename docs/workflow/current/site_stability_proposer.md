# Cr Site Stability Proposer

## Scope

Version 1 ranks candidate Cr environments using existing MACE-screened structures. It is not a DFT stability classifier and does not authorize an endpoint or QE submission by itself.

## Representation

For each structure, Cr is the center and all host displacement vectors use the periodic minimum-image convention. The 71-dimensional descriptor contains:

- species-resolved Gaussian radial densities for Sb and Te from 1.5 to 6.0 A;
- four nearest-neighbor distances per species;
- smooth coordination counts at 2.8, 3.2, 3.6, and 4.0 A;
- species-pair angular Legendre moments through order 4 inside 4.2 A;
- minimum Cr-host distance and a short-contact penalty.

The descriptor is invariant to global translation, global rotation, and permutation of atoms of the same species. An automated test verifies these invariances.

## Model And Leakage Control

A fixed Gaussian process operates after standardization and 20-component PCA. Validation uses five grouped folds. The split unit is a connected source-void family reconstructed from `dedup_clusters.csv`, so structures generated as distortions of the same initial void cannot appear on opposite sides of a fold.

This validation measures recovery of MACE screening energies, not agreement with DFT. It is separate from the manuscript MLFF-vs-DFT evaluation split.

## Grouped Validation

| Metric | Value |
|---|---:|
| Sites | 191 |
| Independent void-family groups | 48 |
| MAE | 0.623 eV |
| RMSE | 0.924 eV |
| Spearman rank correlation | 0.834 |
| Lowest-20% recall | 0.821 |
| Pairwise ordering accuracy for `|delta E| >= 0.2 eV` | 0.851 |
| Empirical 95% uncertainty coverage | 0.958 |

The rank and low-energy retrieval metrics are more relevant than RMSE for deciding which structures receive expensive DFT validation. The approximately 0.9 eV RMSE is too large for quantitative barrier prediction.

## DFT Label Gate

Only two selected candidate sites have historical `relax.out` files. Both outputs are older than their current `relax.in`, so the exact inputs that generated them are unavailable. They are labeled `historical_output_unlinked_to_current_input` and excluded from training.

The gate requires at least 20 independently traceable QE-relaxed local minima. Current count: `0/20`. Consequently:

- screening ranking: allowed;
- DFT stability claim: forbidden;
- automatic QE submission: forbidden;
- manuscript numeric use: forbidden.

## Prospective DFT Validation

The next DFT batch should be declared before results are known:

1. select six low-energy, descriptor-diverse proposals;
2. select six globally matched controls with the same source namespace and
   geometry class, unique void families, and standardized descriptor distance
   at most 2.0;
3. patch every input from the immutable common cell/template;
4. hash the final input before submission and preserve Slurm resource metadata;
5. use identical CPU resources and QE thresholds;
6. compare local-minimum success rate, ionic steps, SCF iterations, and basin collapse;
7. keep all 12 structures in an audit-only split until the prospective evaluation is complete.

This tests whether the proposer improves the yield of stable endpoints. It is distinct from the same-structure direct-versus-transformed A/B experiment, which tests relaxation speed.

This prospective batch is now frozen and running. Audit found that every
mirrored site-search input omitted `CELL_PARAMETERS` despite `ibrav=0`; none was
submitted directly. The corrected preparation injects one hash-bound reference
cell and produces 12 complete, byte-reproducible inputs with shared calculator
identity `03738b164d4bdccb`. Rockfish smoke `30854580` passed, and production
jobs `30854592`-`30854619` were all running in the initial health snapshot.
Full protocol and pair assignments are in
`docs/site_proposer_prospective_dft_validation.md`.

## Artifacts

- Raw mirror and hashes: `data_raw/site_search_qe/`
- Canonical endpoint graph: `data_processed/site_assignments/endpoint_selection_v3/`
- Descriptor and out-of-fold table: `data_processed/site_stability_proposer/v1_mace_screening/site_descriptors_and_oof_predictions.csv`
- Proposed queue: `data_processed/site_stability_proposer/v1_mace_screening/recommended_qe_sites.csv`
- Model manifest: `data_processed/site_stability_proposer/v1_mace_screening/proposer_manifest.json`
- Reproducible builder: `scripts/migrationbench/build_site_stability_proposer.py`
- Prospective preparation: `scripts/migrationbench/prepare_site_proposer_dft_validation.py`
- Frozen prospective batch: `data_processed/site_stability_proposer/prospective_dft_v5_s42/`
