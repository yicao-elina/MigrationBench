# Historical grouped Foundation-vs-NEB audit (2026-10-03)

## Verdict

The finite numerical profiles are reproducible, but the retained local QE `neb.out` files do not provide sufficient completion evidence to promote the five 61-atom paths to formal DFT reference answers. The historical combined table's `dft_quality=pass` label is therefore too strong for the evidence bundle currently available.

## Re-extracted QE path tables

| Path | Images | Last-table barrier (eV) | Max last-table error (eV/A) | JOB DONE | Explicit convergence marker | Status |
|---|---:|---:|---:|---|---|---|
| 1-2 | 5 | 2.790171 | 3.212886 | False | False | conditional / exploratory |
| 1-3 | 5 | 4.003362 | 5.364093 | False | False | conditional / exploratory |
| 1-4 | 5 | 0.336050 | 0.482407 | False | False | conditional / exploratory |
| 1-5 | 5 | 1.099583 | 4.673162 | False | False | conditional / exploratory |
| 1-7 | 5 | 0.582586 | 0.291742 | False | False | conditional / exploratory |

For path `1-4`, the re-extracted barrier is `0.336050 eV`, matching the historical `0.336050 eV` value to numerical precision. This validates the arithmetic of the retained final image table, not the missing convergence evidence.

## Foundation profile checks

The audit found 50/50 expected JSON profiles. All finite profiles have internally consistent `relative_energies_eV = energies_eV - energies_eV[0]` records: 50/50.

The model-side fixed-path calculations are therefore usable as finite diagnostic evaluations. Their barrier errors must be reported against a conditional/exploratory QE reference until the QE completion and force/path convergence evidence is recovered.

## Provenance limitation

The retained grouped-benchmark mirror does not contain the original QE input files for these five 61-atom paths. Functional/SOC/D3/k-point/pseudopotential and convergence settings cannot be independently revalidated from this bundle alone. The raw output hashes and extracted rows are retained in `qe_path_audit.csv`; Foundation JSON hashes and profile checks are retained in `foundation_profile_audit.csv`.

## Source-of-truth files

- Campaign: `/Users/alina/Documents/Codex/foundation_grouped_benchmark_20261001`
- QE source outputs: `data/dft_paths/*-neb.out`
- Foundation outputs: `results/neb/*.json`
- Historical combined table: `analysis/combined_all_paths_barrier_results.csv`
