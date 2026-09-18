# NEB Energy Barrier Snapshot

Generated from current synced MigrationBench artifacts. For QE rows, `Max error/fmax` is the maximum over movable images; frozen-endpoint errors are retained separately in the CSV/JSON diagnostics. Values marked running/unconverged/quarantined are not final manuscript references.

## Em Summary

| Path | Branch/source | Em (eV) | Status | Max error/fmax | Iteration |
|---|---|---:|---|---:|---:|
| `1-2` | `mb_mlff_cpu_1_2_s42_30791344` / MACE/MLFF | 9.827962 | mlff_proxy_basin_collapse_or_large_relax | 29.609216 |  |
| `1-2` | `pipeline_config_current_reference` / historical_reference_or_config | 2.790171 | unconverged |  |  |
| `1-3` | `mb_mlff_cpu_1_3_s42_30791345` / MACE/MLFF | 0.882818 | mlff_proxy_basin_collapse_or_large_relax | 2.115137 |  |
| `1-3` | `pipeline_config_current_reference` / historical_reference_or_config | 4.003362 | unconverged |  |  |
| `1-4` | `pipeline_config_current_reference` / historical_reference_or_config | 0.336050 | numerically_stationary_unconverged |  |  |
| `1-5` | `mb_mlff_cpu_1_5_s42_30791346` / MACE/MLFF | 0.499037 | mlff_proxy_basin_collapse_or_large_relax | 19.357830 |  |
| `1-6` | `mb_mlff_cpu_1_6_s42_30791347` / MACE/MLFF | 0.452755 | mlff_proxy_basin_collapse_or_large_relax | 2.251724 |  |
| `1-6` | `mlff_1-6_repair_idpp_s42` / MACE/MLFF | 0.452755 | mlff_proxy_basin_collapse_or_large_relax | 2.251724 |  |
| `1-6` | `mb_qe16_direct_pf_s42_30791500` / QE | 0.820400 | stopped_unconverged | 1.576222 | 12 |
| `1-6` | `mb_qe16_direct_r2_s42_30791828` / QE | 0.820400 | stopped_unconverged | 0.408386 | 65 |
| `1-6` | `mb_qe16_direct_r3_s42_30812496` / QE | 0.820400 | running_unconverged | 2.471425 | 103 |
| `1-7` | `mb_qe17_direct_pf_s42_30791508` / QE | 0.557933 | stopped_unconverged | 0.869559 | 13 |
| `1-7` | `mb_qe17_direct_r2_s42_30791829` / QE | 0.557933 | stopped_unconverged | 0.436896 | 71 |
| `1-7` | `mb_qe17_direct_r3_s42_30812497` / QE | 0.557933 | running_unconverged | 1.615549 | 108 |
| `1-7` | `pipeline_config_current_reference` / historical_reference_or_config | 0.582586 | unconverged |  |  |
| `81_neb_1` | `historical_2D_421_neb_1` / historical_QE_candidate | 1.034728 | historical_candidate_quarantine | 0.355916 |  |
| `81_neb_2` | `historical_2D_421_neb_2` / historical_QE_candidate | 1.561262 | historical_candidate_quarantine | 0.635720 |  |
| `81_neb_3` | `historical_2D_421_neb_3` / historical_QE_candidate | 1.691053 | historical_candidate_quarantine | 0.231831 |  |
| `81_neb_4` | `historical_2D_421_neb_4` / historical_QE_candidate | 2.063858 | historical_candidate_quarantine | 0.526984 |  |
| `81_neb_5` | `historical_2D_421_neb_5` / historical_QE_candidate | 1.414868 | historical_candidate_quarantine | 0.319104 |  |
| `victor_neb2` | `victor_historical_V-ST_neb2` / QE | 109.440398 | running_unconverged | 380.372065 | 4 |
| `victor_neb2_copy` | `victor_historical_V-ST_neb2_copy` / QE | 627.040883 | quarantine_overflow | 475.185787 | 1 |

## Profile Figures

- [00_1-2__mb_mlff_cpu_1_2_s42_30791344.svg](/Users/alina/Project/26MIgrationBench/data_processed/neb_energy_profiles/00_1-2__mb_mlff_cpu_1_2_s42_30791344.svg)
- [02_1-3__mb_mlff_cpu_1_3_s42_30791345.svg](/Users/alina/Project/26MIgrationBench/data_processed/neb_energy_profiles/02_1-3__mb_mlff_cpu_1_3_s42_30791345.svg)
- [05_1-5__mb_mlff_cpu_1_5_s42_30791346.svg](/Users/alina/Project/26MIgrationBench/data_processed/neb_energy_profiles/05_1-5__mb_mlff_cpu_1_5_s42_30791346.svg)
- [06_1-6__mb_mlff_cpu_1_6_s42_30791347.svg](/Users/alina/Project/26MIgrationBench/data_processed/neb_energy_profiles/06_1-6__mb_mlff_cpu_1_6_s42_30791347.svg)
- [07_1-6__mlff_1-6_repair_idpp_s42.svg](/Users/alina/Project/26MIgrationBench/data_processed/neb_energy_profiles/07_1-6__mlff_1-6_repair_idpp_s42.svg)
- [08_1-6__mb_qe16_direct_pf_s42_30791500.svg](/Users/alina/Project/26MIgrationBench/data_processed/neb_energy_profiles/08_1-6__mb_qe16_direct_pf_s42_30791500.svg)
- [09_1-6__mb_qe16_direct_r2_s42_30791828.svg](/Users/alina/Project/26MIgrationBench/data_processed/neb_energy_profiles/09_1-6__mb_qe16_direct_r2_s42_30791828.svg)
- [10_1-6__mb_qe16_direct_r3_s42_30812496.svg](/Users/alina/Project/26MIgrationBench/data_processed/neb_energy_profiles/10_1-6__mb_qe16_direct_r3_s42_30812496.svg)
- [12_1-7__mb_qe17_direct_pf_s42_30791508.svg](/Users/alina/Project/26MIgrationBench/data_processed/neb_energy_profiles/12_1-7__mb_qe17_direct_pf_s42_30791508.svg)
- [13_1-7__mb_qe17_direct_r2_s42_30791829.svg](/Users/alina/Project/26MIgrationBench/data_processed/neb_energy_profiles/13_1-7__mb_qe17_direct_r2_s42_30791829.svg)
- [14_1-7__mb_qe17_direct_r3_s42_30812497.svg](/Users/alina/Project/26MIgrationBench/data_processed/neb_energy_profiles/14_1-7__mb_qe17_direct_r3_s42_30812497.svg)
- [27_victor_neb2__victor_historical_V-ST_neb2.svg](/Users/alina/Project/26MIgrationBench/data_processed/neb_energy_profiles/27_victor_neb2__victor_historical_V-ST_neb2.svg)
- [28_victor_neb2_copy__victor_historical_V-ST_neb2_copy.svg](/Users/alina/Project/26MIgrationBench/data_processed/neb_energy_profiles/28_victor_neb2_copy__victor_historical_V-ST_neb2_copy.svg)

## Files

- Summary CSV: `/Users/alina/Project/26MIgrationBench/data_processed/neb_energy_profiles/neb_energy_summary.csv`
- Summary JSON: `/Users/alina/Project/26MIgrationBench/data_processed/neb_energy_profiles/neb_energy_summary.json`
- HTML index: `/Users/alina/Project/26MIgrationBench/data_processed/neb_energy_profiles/neb_energy_profiles.html`
