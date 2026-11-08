# DFT-vs-MACE within-method comparison

All four QE cases passed the diagnostic gate. Energy columns are changes relative to `normal_461800_initial` within DFT or within the same MACE model; absolute cross-model energy ranking is not interpreted.

| Case | Model | ΔE DFT (eV) | ΔE MACE (eV) | DFT max | MACE max | Force MAE |
|---|---|---:|---:|---:|---:|---:|
| bounded_multiT_462000_step464000 | foundation-omat | 68.3721 | 46.2167 | 0.0883 | 58.3867 | 4.3094 |
| bounded_multiT_462000_step464000 | from-scratch | 68.3721 | 28.5534 | 0.0883 | 30.5571 | 2.5911 |
| bounded_multiT_462000_step464000 | naive-fine-tuning | 68.3721 | 40.7721 | 0.0883 | 47.3102 | 3.9216 |
| bounded_multiT_462000_step464000 | multi_T-fine-tuning | 68.3721 | 44.3336 | 0.0883 | 58.5472 | 4.5611 |
| normal_461800_initial | foundation-omat | 0.0000 | 0.0000 | 0.0815 | 9.1353 | 1.4155 |
| normal_461800_initial | from-scratch | 0.0000 | 0.0000 | 0.0815 | 4.1209 | 1.1628 |
| normal_461800_initial | naive-fine-tuning | 0.0000 | 0.0000 | 0.0815 | 10.4492 | 1.4645 |
| normal_461800_initial | multi_T-fine-tuning | 0.0000 | 0.0000 | 0.0815 | 12.2731 | 1.5762 |
| runaway_foundation_461800_step462120 | foundation-omat | 38.2513 | 24.9178 | 1.2568 | 48.1287 | 3.4398 |
| runaway_foundation_461800_step462120 | from-scratch | 38.2513 | 37.4292 | 1.2568 | 100.1307 | 7.9932 |
| runaway_foundation_461800_step462120 | naive-fine-tuning | 38.2513 | 22.6762 | 1.2568 | 42.5450 | 3.0286 |
| runaway_foundation_461800_step462120 | multi_T-fine-tuning | 38.2513 | 23.6390 | 1.2568 | 48.6883 | 3.1983 |
| runaway_scratch_461800_step462060 | foundation-omat | 45.6747 | 29.8855 | 0.0844 | 42.1240 | 3.7418 |
| runaway_scratch_461800_step462060 | from-scratch | 45.6747 | 41.6112 | 0.0844 | 81.2073 | 7.3583 |
| runaway_scratch_461800_step462060 | naive-fine-tuning | 45.6747 | 27.8491 | 0.0844 | 35.0423 | 3.4436 |
| runaway_scratch_461800_step462060 | multi_T-fine-tuning | 45.6747 | 28.5161 | 0.0844 | 38.7369 | 3.6786 |

Scope: `RECONSTRUCTION / REPRODUCED_SCOPE_LIMITED`; these 81-atom local cells are not formal full-cell DFT validation.
