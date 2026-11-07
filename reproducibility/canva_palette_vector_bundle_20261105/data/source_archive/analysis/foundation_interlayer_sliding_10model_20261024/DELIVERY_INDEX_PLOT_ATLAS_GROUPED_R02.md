# Grouped interlayer-sliding figure — Plot Atlas R02

## Update

R02 is an additive typography and axis-label update based on the supplied reference PDF `fig4b_interlayer_sliding_pristine_fixed_path_20261024_yellow_scratch_R04.pdf`.

- x-axis title: `Reaction Coordinate`
- y-axis title: `ΔE (eV)` retained
- all visible text and numeric labels: increased by approximately 2 pt from R01 and set bold
- panel titles, panel letters, legend text, tick labels, and axis labels: bold
- plot spines and tick marks: visibly strengthened
- class mean lines and pointwise sample-SD uncertainty bands: unchanged
- scientific grouping: `MACE-MP`, `MACE-OMAT-0`, and `MACE-MH`
- reference trace: screened QE/DFT path table

## Main 2×2 figure

- [PNG](results/figures/plot_atlas_grouped_r02/interlayer_sliding_10model_grouped_2x2_plot_atlas_r02.png)
- [SVG](results/figures/plot_atlas_grouped_r02/interlayer_sliding_10model_grouped_2x2_plot_atlas_r02.svg)
- [PDF](results/figures/plot_atlas_grouped_r02/interlayer_sliding_10model_grouped_2x2_plot_atlas_r02.pdf)

## Standalone panels

- [Fixed bulk — PNG](results/figures/plot_atlas_grouped_r02/fixed_bulk_grouped_plot_atlas_r02.png) · [SVG](results/figures/plot_atlas_grouped_r02/fixed_bulk_grouped_plot_atlas_r02.svg) · [PDF](results/figures/plot_atlas_grouped_r02/fixed_bulk_grouped_plot_atlas_r02.pdf)
- [Fixed Cr-in-gap — PNG](results/figures/plot_atlas_grouped_r02/fixed_cr_in_gap_grouped_plot_atlas_r02.png) · [SVG](results/figures/plot_atlas_grouped_r02/fixed_cr_in_gap_grouped_plot_atlas_r02.svg) · [PDF](results/figures/plot_atlas_grouped_r02/fixed_cr_in_gap_grouped_plot_atlas_r02.pdf)
- [Self bulk — PNG](results/figures/plot_atlas_grouped_r02/self_bulk_grouped_plot_atlas_r02.png) · [SVG](results/figures/plot_atlas_grouped_r02/self_bulk_grouped_plot_atlas_r02.svg) · [PDF](results/figures/plot_atlas_grouped_r02/self_bulk_grouped_plot_atlas_r02.pdf)
- [Self Cr-in-gap — PNG](results/figures/plot_atlas_grouped_r02/self_cr_in_gap_grouped_plot_atlas_r02.png) · [SVG](results/figures/plot_atlas_grouped_r02/self_cr_in_gap_grouped_plot_atlas_r02.svg) · [PDF](results/figures/plot_atlas_grouped_r02/self_cr_in_gap_grouped_plot_atlas_r02.pdf)

## Provenance and reproduction

- [R02 provenance manifest](manifests/FIGURE_PROVENANCE_PLOT_ATLAS_GROUPED_R02.json)
- [Reproduction script](scripts/plot_interlayer_grouped_profile_atlas_r01.py)

From the repository root:

```bash
python analysis/foundation_interlayer_sliding_10model_20261024/scripts/plot_interlayer_grouped_profile_atlas_r01.py \
  --root /Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/foundation_interlayer_sliding_10model_20261024
```

## Scientific boundaries

- Fixed-path panels remain `FINITE_DIAGNOSTIC` single-point diagnostics.
- Self panels remain endpoint-only MLIP-NEB path observables, not DFT MEP barriers.
- QE/DFT reference traces retain their warning-bearing provenance and are not silently promoted to converged barriers.
- The MatPES-PBE OMAT-FT checkpoint remains grouped under `MACE-OMAT-0`.
