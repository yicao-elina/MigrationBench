# Publication report: Cr versus Cu/Li/Na NEB paths

## Scope

This package joins the three new Rockfish Cr-substitution paths to the existing nine Cu/Li/Na path-class panels: within-QL, in-gap diffusion, and gap → QL. Foundation-model values are same-geometry fixed-path single-point evaluations. The QE columns are path-table references, not certified converged barriers.

## Cr activation-energy readout

| Path class | Rockfish job | Forward QE (eV) | Reverse QE (eV) | Last complete iter. | Model range (eV) | Model median (eV) |
|---|---:|---:|---:|---:|---:|---:|
| gap → QL | 31916909 | 0.000000 | 0.095947 | 68 | 0.000000–0.072545 | 0.029836 |
| in-gap diffusion | 31916910 | 1.524299 | 0.675386 | 81 | 0.899060–1.549396 | 1.167452 |
| within-QL | 31916908 | 1.844900 | 2.346822 | 160 | 21.176812–29.735416 | 24.222283 |

## Results

The Cr within-QL path is the dominant model-disagreement case. Its Rockfish QE path-table forward activation is 1.8449 eV, while the ten-model barrier-like range is 21.176812–29.735416 eV. The discrepancy is concentrated at the interior images and should be interpreted as a strong model/geometry stress signal, not as evidence that the physical Cr barrier is 20–30 eV.

For Cr in-gap diffusion, the QE path-table value is 1.5243 eV and the Foundation-model range is approximately 0.899–1.549 eV, with the large-density Agnesi model closest to the screened QE reference. For Cr gap → QL, the QE path is monotone downhill relative to image 1 (forward 0.000 eV; reverse 0.095947 eV), and all model barrier-like values remain below 0.073 eV.

Against the existing Cu/Li/Na panel, within-QL paths are the most model-sensitive class: Cu is already a large-disagreement outlier, whereas Li and Na show much closer model envelopes. The gap-based classes have substantially smaller model spreads, although gap → QL is endpoint-direction dependent and should not be summarized by a forward barrier alone. The Cr pattern therefore extends the existing stress-test story rather than providing a single universal dopant ranking.

## Publication status

The three Rockfish Cr path-table outputs, each with ten images, are `COMPLETED`/`JOB DONE` with ten finite images, but each carries an SCF-convergence warning. Accordingly, Cr QE values are `COMPLETED_BUT_SCF_WARNING` and excluded from final converged-barrier claims. Figure 1 v3 is publication-safe as a provenance-labeled diagnostic/profile figure; a final Cr activation-energy claim requires rerunning the three paths until the SCF warning and path-convergence gates are both cleared.

## Reproducibility files

- `rockfish_vs_joshitha_contract_comparison.tsv`: numerical QE/NEB contract audit.
- `cr_trajectory_identity_audit.csv`: v2-to-v3 Cr trajectory mapping.
- `rockfish_cr_foundation_barrier_table.csv`: all 10 foundation-model evaluations per Cr path.
- `figure1_task_type_matrix_v3/figure1_task_type_matrix_v3.pdf`: updated figure-book panel.
