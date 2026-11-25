# Evidence package — Foundation-model class ensemble 4×3 R03

This package reproduces the publication-style 4×3 figure with rows `Cr`, `Cu`, `Li`, `Na` and columns `in-gap diffusion`, `gap→QL`, `within-QL`. Colored curves are pointwise class means with sample-SD bands; hollow circles are class means. The screened QE/DFT references are black dashed lines with hollow diamonds.

## Final artifacts

- Figure PNG: `figures/foundation_class_4x3_R03/foundation_model_class_4x3_R03.png`
- Figure SVG: `figures/foundation_class_4x3_R03/foundation_model_class_4x3_R03.svg`
- Figure PDF: `figures/foundation_class_4x3_R03/foundation_model_class_4x3_R03.pdf`
- Delivery index: `DELIVERY_INDEX_foundation_class_4x3_R03.md`
- Figure manifest: `figure_data_manifest_foundation_class_4x3_R03.json`
- QA report: `QA_REPORT_foundation_class_4x3_R03.json`
- Caption: `CAPTIONS_foundation_class_4x3_R03.md`

## Reproduction scripts

Run from the repository root:

```text
python analysis/neb_cross_element_triptychs_20261024/build_cr_mh1_fixed_path_r01.py
python analysis/neb_cross_element_triptychs_20261024/build_foundation_class_4x3_r03.py
```

- `build_cr_mh1_fixed_path_r01.py`: evaluates the missing Cr `MACE-MH-1` profile on the three canonical fixed paths, 10 images per path, using the `omat_pbe` head; no relaxation is performed.
- `build_foundation_class_4x3_r03.py`: assembles the nine-model class means/SDs, adds all four screened QE references, and renders PNG/SVG/PDF plus independent panels.
- `build_foundation_class_4x3_r02.py`: inherited R02 assembly layer, including the Cr `MACE-MH-1` supplement.
- `build_foundation_class_4x3_r01.py`: reviewed visual grammar and original Cu/Li/Na campaign loader.

## Model-evaluation evidence

- Cr MH-1 supplement: `data/foundation_class_4x3_R02/Cr_mace_mh1_fixed_path_profiles_R02.tsv`
- Cr MH-1 supplement manifest: `data/foundation_class_4x3_R02/Cr_mace_mh1_fixed_path_manifest_R02.json`
- Combined model profiles used by R03: `data/foundation_class_4x3_R03/foundation_model_profiles_9models_4x3_R03.tsv`
- Class mean/SD profiles used by R03: `data/foundation_class_4x3_R03/foundation_model_class_profiles_4x3_R03.tsv`
- Foundation checkpoint directory: `/Users/alina/Downloads/mace-foundation-models`
- All checkpoint SHA256 values are recorded in `figure_data_manifest_foundation_class_4x3_R03.json`.

## Model data sources

- Cr TIER1 fixed-path profiles for the existing eight Cr checkpoints: `analysis/task3_endpoint_only_neb/neb_fixed_vs_self_matrix_20261019/derived_r01/TIER1_IMAGE_PROFILES.tsv`
- Cu/Li/Na nine-model campaign profiles: `data/foundation_model_profiles_9models_3x3.tsv`
- Canonical Cr fixed-path inputs used for MH-1: `analysis/task3_endpoint_only_neb/neb_fixed_vs_self_matrix_20261019/inputs/fixed_r01/cr_canonical_in_gap.extxyz`, `cr_canonical_gap_to_ql.extxyz`, `cr_canonical_within_ql.extxyz`
- Cr fixed-path input SHA256 values are recorded in `figure_data_manifest_foundation_class_4x3_R03.json`.

## QE/DFT reference sources

- Cr screened reference source: `data/Cr_fixed_path_representative_profiles.tsv`
- R03 combined reference table: `data/foundation_class_4x3_R03/foundation_model_qe_reference_profiles_4x3_R03.tsv`
- Cu/Li/Na reference source: `data/foundation_model_qe_reference_profiles_3x3.tsv`
- Original Cr/Cu/Li/Na reference provenance points to `analysis/neb_fixed_path_trajectory_figure_20261023/python/data/dft_profiles.tsv` where available.

## Scientific boundary

The package contains fixed-path single-point Foundation-model diagnostics and screened QE/DFT path references. Every retained reference path is marked `COMPLETED_BUT_SCF_WARNING`; the references are not unconditional converged DFT ground truth and the figure must not be read as a universal activation-barrier claim. The colored curves are not self-relaxed MLFF-NEB results.

R03 QA expects 1080 finite model rows, 360 class-profile rows, 120 reference rows, complete Cr `MACE-MH-1` coverage, independent y axes, and fixed-path-only scope.
