# Foundation-model class ensemble 3×3 delivery

Rows are Cu, Li, and Na. Columns are `gap→QL`, `in-gap diffusion`, and `within-QL`.

## Final composite

- [3×3 PNG](figures/foundation_class_3x3/foundation_model_class_3x3_R01.png)
- [3×3 SVG](figures/foundation_class_3x3/foundation_model_class_3x3_R01.svg)
- [3×3 PDF](figures/foundation_class_3x3/foundation_model_class_3x3_R01.pdf)

## Editable standalone panels

All nine standalone panels are in `figures/foundation_class_3x3/` as PNG/SVG/PDF triples:

```text
panel_{Cu,Li,Na}_{gap_to_ql,within_gap,within_ql}.{png,svg,pdf}
```

## Data and reproducibility

- [Build script](build_foundation_class_3x3_r01.py)
- [9-model raw profiles](data/foundation_model_profiles_9models_3x3.tsv)
- [Class means and SD bands](data/foundation_model_class_profiles_3x3.tsv)
- [Screened QE reference profiles](data/foundation_model_qe_reference_profiles_3x3.tsv)
- [Figure manifest](figure_data_manifest_foundation_class_3x3_R01.json)
- [QA report](QA_REPORT_foundation_class_3x3_R01.json)
- [Caption and scientific boundary](CAPTIONS_foundation_class_3x3_R01.md)

## Model classes

`Agnesi` (n=4), `MACE-H` (n=2), `MACE-MP` (n=1), and `OMAT-0` (n=2).

The MACE-MP line is shown without an SD band because one model cannot provide an empirical within-class standard deviation. The dashed black trace is a screened QE path-table reference; it retains SCF-warning status and is not converged DFT ground truth.
