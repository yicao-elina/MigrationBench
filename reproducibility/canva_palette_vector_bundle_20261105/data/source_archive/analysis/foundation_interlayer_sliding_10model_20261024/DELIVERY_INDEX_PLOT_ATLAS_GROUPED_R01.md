# Grouped MACE class profile reconstruction R01

This is the requested strict-format reconstruction of the campaign using the
same three-class grammar as the reference example.

## Model grouping

- `MACE-MP`: Agnesi-L, Agnesi-MD, Agnesi-SD, Agnesi-M, MP-0b3
- `MACE-OMAT-0`: OMAT-M, OMAT-S, MatPES-OMAT-FT
- `MACE-MH`: MH-0, MH-1

For each class, the plotted line is the pointwise mean across member models;
the translucent band is the pointwise sample SD (`ddof=1`).

## Main figure

- `results/figures/plot_atlas_grouped_r01/interlayer_sliding_10model_grouped_2x2_plot_atlas_r01.{png,svg,pdf}`

Panel layout is 2 systems × 2 methods: bulk/Cr-in-gap by fixed-path/self-
relaxed endpoint-only. Four standalone panels are also emitted in the same
directory.

## Reproduction

```bash
python analysis/foundation_interlayer_sliding_10model_20261024/scripts/plot_interlayer_grouped_profile_atlas_r01.py \
  --root /Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/foundation_interlayer_sliding_10model_20261024
```

## Scientific boundaries

- Fixed-path remains `FINITE_DIAGNOSTIC` and is not a self-relaxed DFT MEP barrier.
- Self-relaxed remains the accepted endpoint-only MLIP NEB path quantity.
- QE remains a screened `SCF_WARNING` reference, not unconditional ground truth.
- MatPES-PBE OMAT-FT is included in the `MACE-OMAT-0` class as `MatPES-OMAT-FT`.

Output hashes and source-table hashes are recorded in
`manifests/FIGURE_PROVENANCE_PLOT_ATLAS_GROUPED_R01.json`.
