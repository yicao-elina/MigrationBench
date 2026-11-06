# Foundation-model class ensemble 3×3 — R02

Rows are Cu, Li, and Na. Columns are `in-gap diffusion`, `gap→QL`, and `within-QL`.

## Final composite

- [3×3 PNG](figures/foundation_class_3x3_R02/foundation_model_class_3x3_R02.png)
- [3×3 SVG](figures/foundation_class_3x3_R02/foundation_model_class_3x3_R02.svg)
- [3×3 PDF](figures/foundation_class_3x3_R02/foundation_model_class_3x3_R02.pdf)

## Model classification

- `MACE-MP` (n=5): MP-0b, MP-0b2 small/medium/large, MP-0b3 medium
- `MACE-OMAT-0` (n=2): OMAT-0 small/medium
- `MACE-MH` (n=2): MH-0/MH-1

The Agnesi filenames are MACE-MP release assets; they are not a separate Foundation family. This follows the official ACEsuit model mapping.

## Data and reproducibility

- [Build script](build_foundation_class_3x3_r01.py)
- [Raw 9-model profiles](data/foundation_class_3x3_R02/foundation_model_profiles_9models_3x3_R02.tsv)
- [Class means, individual points, and SD bands](data/foundation_class_3x3_R02/foundation_model_class_profiles_3x3_R02.tsv)
- [Screened QE reference profiles](data/foundation_class_3x3_R02/foundation_model_qe_reference_profiles_3x3_R02.tsv)
- [Figure manifest](figure_data_manifest_foundation_class_3x3_R02.json)
- [QA report](QA_REPORT_foundation_class_3x3_R02.json)
- [Caption and scientific boundary](CAPTIONS_foundation_class_3x3_R02.md)

Small colored circles are individual model values. Hollow circles are class means. The dashed black curve is a screened QE path-table reference; it retains SCF-warning status and is not converged DFT ground truth.
