# Foundation grouped benchmark reproducibility index

| Artifact | Evidence |
|---|---|
| Frozen grouped test | `data/processed/foundation_grouped/equilibrium_split_summary.csv`; SHA256 `d49cf4b2ab49fc1e521d314c5828cf5dcaea6853e70cecb2727ffd0678bb27ab` |
| Checkpoint manifest | `data/processed/foundation_grouped/models.tsv`; preflight evidence `preflight_manifest.csv` |
| Source/release ledger | `data/processed/foundation_grouped/source_release.tsv`; official family documentation URL and checkpoint provenance notes |
| Preflight | Slurm array `31483607`, 10/10 terminal `COMPLETED 0:0` |
| Test equilibrium | Slurm array `31483362`; MH-1 scientific result from repaired job `31482442` retained in the summary |
| Valid equilibrium | Slurm array `31483778`, 10/10 terminal `COMPLETED 0:0` |
| NEB profiles | 50 raw JSON profiles in `data/processed/foundation_grouped/neb/`; materialized summary `neb_summary_complete.csv` |
| DFT reference | `data/processed/foundation_grouped/dft_1-4_neb_energies.csv`; provisional path 1-4 barrier `0.336050 eV` |
| Test/valid metrics | `data/processed/foundation_grouped/equilibrium_split_summary.csv` |
| NEB metrics | `data/processed/foundation_grouped/neb_summary_complete.csv` |
| Representative profile diagnostics | `data/processed/foundation_grouped/special_profile_diagnostics.csv` |
| Main comparison | `data/processed/foundation_grouped/foundation_vs_grouped_comparison.csv` |
| Publication figures | `paper/revision1/figures/foundation_grouped/path_1-4_energy_profiles_corrected.{png,svg,pdf}` and `*_family_aggregated.{png,svg,pdf}` |
| Reproduction script | `scripts/plot_foundation_neb_profiles.py` |
| Reviewer text | `docs/audit/foundation_grouped_benchmark/foundation_benchmark_report.md` |

The Slurm records and failed MH-1 MACE 0.3.12 deserialization evidence are retained on Rockfish; the repaired MH-1 run uses the isolated `python_pkgs` MACE 0.3.16 runtime and explicit `omat_pbe` head.
