# Phase 1 static same-configuration force comparison

## Scope

This analysis evaluates the four existing MACE models on the same four periodic
configurations. It is a force/stress consistency comparison, not a DFT accuracy
benchmark: no DFT reference was used and no model is treated as ground truth.

## Inputs and identity checks

The common configurations were selected from the real Rockfish continuation
file
`/scratch16/pclancy3/yi/mace_four_model_MD_continuation_20260930/1-from-scratch/segment_from_000200000.extxyz` by exact `step` metadata, not by filename or assumed frame index. The selected steps are 460000, 461800, 462000, and 462200. The source contains 4,001 frames and has SHA256
`38140e3252961d7e2b470232a179ffbeb65d29f2dca4eb0e1d42c9d94c0b45c7`.

Each selected input was checked for 2,050 atoms, finite positions, periodicity
`T T T`, constant cell, and unchanged species order. The verified composition is
50 Cr, 800 Sb, and 1,200 Te. `momenta:R:3` and `forces:R:3` are present in the
source extxyz schema; momenta are retained as provenance and are not used in the
single-point calculations. All Cr atoms are treated as key atoms, using the
1-based indices recorded in `inputs/snapshot_manifest.json` rather than an
assumed dopant mapping.

## Model evaluation

Each snapshot was evaluated once with each existing model (16 evaluations total)
using the Rockfish MACE environment and a single A100 GPU. The run was submitted
as Slurm job 31511914. The evaluator recorded model path, observed SHA256, input
SHA256, device/dtype, elapsed time, energy, forces, stress availability, finite
checks, raw `.npz` output, and any exception. Units are eV for energy, eV Å^-1
for forces, and eV Å^-3 for stress; stress comparisons are also reported in GPa
using 160.21766208 GPa per eV Å^-3.

## Metrics

Forces are compared pairwise across models and by leave-one-model-out residuals
against the mean force of the other three models. Reported force metrics include
component MAE/RMSE/max-absolute difference and vector-norm mean/max difference.
Key-atom metrics are force norms over the verified Cr set. Local disagreement is
defined as the atoms within 3.0 Å minimum-image distance of the atom with the
largest pairwise force-vector spread at each step. Geometry diagnostics include
cell volume, minimum periodic pair distance, and local-shell population.

Absolute energies are retained, but cross-model energy offsets are not interpreted
as physical ranking because the Foundation-OMAT energy zero differs from the
Cr-specific models by approximately 1.9 million eV for this 2,050-atom cell.
Within-model energy changes and pairwise force/stress differences are the
interpretable quantities in this phase.
