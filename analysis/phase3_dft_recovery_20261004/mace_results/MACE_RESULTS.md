# MACE reconstruction diagnostics

These are four-model single-point evaluations on the same reconstructed 81-atom local cells used by the additive Phase-3 DFT diagnostic.
They are not full-cell validation and absolute energy ranking across models is not scientifically interpreted.

| Case | Foundation-OMAT max force | From-scratch max force | Naive-FT max force | Multi-T max force |
|---|---:|---:|---:|---:|
| bounded_multiT_462000_step464000 | 58.3867 | 30.5571 | 47.3102 | 58.5472 |
| normal_461800_initial | 9.1353 | 4.1209 | 10.4492 | 12.2731 |
| runaway_foundation_461800_step462120 | 48.1287 | 100.1307 | 42.5450 | 48.6883 |
| runaway_scratch_461800_step462060 | 42.1240 | 81.2073 | 35.0423 | 38.7369 |

The full per-atom force arrays, energy values, model hashes, and input hashes are in `mace_reconstructed.json` and `mace_summary.csv`.

The Plot Atlas-rendered auxiliary figure is
`figures/mace_reconstruction_diagnostic.{png,svg,pdf}`. It is explicitly
labelled as a local-cell reconstruction diagnostic and is not DFT validation.
