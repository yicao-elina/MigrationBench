# MACE-CANDIDATE-EVAL

## Article target
Results/SI: candidate selection and model-evaluation coverage audit

## Caption
A, runtime coverage across the nine registered model families; B, successful versus runtime-failure rows by candidate type; C, maximum-force distribution for the three model labels with successful rows.

## Results
The table contains 975 rows: 195 successful evaluations and 780 runtime failures. The successful rows are limited to Foundation, FT-600K, and MPA0 labels; the grouped-seed, Scratch, and FT-MultiT branches are not scientifically failed model evaluations here because their recorded runtime error is the missing NVIDIA driver.

## Discussion
This figure is a campaign/runtime coverage audit and a force-scale diagnostic. It should be used to explain why the current candidate-evaluation ablation cannot support a complete model-family ranking or grouped-seed conclusion.

## Scientific boundary
- The runtime failure traceback identifies a missing NVIDIA driver; do not relabel those rows as model-quality failures.
- Absolute energies are not comparable across model families; this package uses status and force-scale fields only.
- The 65-candidate manifest and selection provenance remain the source of candidate identity.
