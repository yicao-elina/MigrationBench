# Exploratory non-equilibrium NEB-path sampling

Date: 2026-10-03 (corrected energy-unit rebuild)

## Scope and interpretation

This is an exploratory single-point/path-sampling analysis, not a converged NEB barrier calculation. The source NEB outputs do not expose the inner electronic SCF residual history, so the number of inner SCF iterations cannot be recovered from `neb.out`. The auditable counts below are NEB outer path iterations and recorded image self-consistency calls.

The DFT path energy is treated as a fixed-geometry electronic-energy sample only when the last two complete energy tables are numerically stable under the screening rule `max per-image |ΔE| < 0.1 eV`. This is a stability heuristic; it does not prove electronic SCF convergence, force convergence, equilibrium, or a reliable DFT activation barrier. The saved `sb2te3.path` energies are stored in Hartree and were converted to eV with `27.211386245988`; Cartesian coordinates were converted from Bohr to Å with `0.529177210903`.

## DFT audit

| path | outer iterations | image self-consistency calls | max last-table ΔE (eV) | barrier ΔE (eV) | exploratory eligible |
|---|---:|---:|---:|---:|---|
| 2L-Cu__neb | 45 | 442 | 19.062111 | -16.590291 | no |
| 2L-Cu__neb-QLs | 79 | 788 | 0.050366 | -0.050416 | yes |
| 2L-Cu__neb-gap-diffuse | 42 | 420 | 0.006842 | -0.003178 | yes |
| 2L-Cu__neb-gap2QL | 95 | 950 | 0.001316 | -0.000132 | yes |
| 2L-Li__neb-QLs | 42 | 414 | 0.001862 | -0.001745 | yes |
| 2L-Li__neb-gap-diffuse | 47 | 469 | 0.004969 | -0.001793 | yes |
| 2L-Li__neb-gap2Ql | 51 | 505 | 0.000491 | +0.000008 | yes |
| 2L-Na__neb-QLs | 42 | 329 | 0.154644 | -0.123898 | no |
| 2L-Na__neb-gap-diffuse | 42 | 413 | 0.007797 | -0.003418 | yes |
| 2L-Na__neb-gap2QL | 55 | 539 | 0.004236 | -0.000089 | yes |

Eight of ten paths pass the exploratory `<0.1 eV` energy-stability screen. The Cu `neb` and Na `neb-QLs` paths are excluded because their final energy tables are not stable enough.

The two excluded paths remain in the complete diagnostic set. For `2L-Cu__neb`, the penultimate-to-final per-image differences are 0.006135, 0.647049, 1.860266, 4.156051, 6.538434, **19.062111**, 8.033137, 1.942854, 0.747535, and 0.004299 eV for images 1–10; the maximum is at image 6 and the interior images 4–8 all exceed 1 eV. For `2L-Na__neb-QLs`, the corresponding differences are 0.010375, 0.034977, 0.077067, 0.103012, 0.129757, 0.134274, **0.154644**, 0.130188, 0.043976, and 0.008486 eV; the maximum is at image 7 and images 4–8 exceed 0.1 eV. The first path is numerically unstable; the second is a borderline stability failure rather than the same type of breakdown.

## Sampling and MACE evaluation

For each eligible path, 10 images were evaluated in three forms:

- `base`: checkpoint path geometry;
- `perturb_a_dx020`: `( +0.20, 0.00, 0.00 ) Å` displacement;
- `perturb_b_tilt020`: `( -0.10, +0.14, +0.10 ) Å` displacement.

This produced 240 DFT-labeled exploratory frames. Foundation and MPA0 completed 240/240 single-point evaluations each. FT-600K completed 0/240 because its deployed model does not support the required elements: Cu `Z=29`, Li `Z=3`, and Na `Z=11`.

All MACE/DFT profile comparisons use within-model relative energies, subtracting image 1 of the same path and variant. Absolute energies are not compared across models, and the results are not presented as DFT-accuracy validation.

## Main results

For the eight eligible paths, base-profile median MAE relative to the screened DFT path is:

| model | median profile MAE | median absolute barrier difference |
|---|---:|---:|
| Foundation | 0.0606 eV | 0.0541 eV |
| MPA0 | 0.1154 eV | 0.1418 eV |

The largest discrepancies are concentrated in the high-curvature `2L-Cu__neb-QLs` profile: its base barrier-like errors are 23.351 eV and 25.332 eV for Foundation and MPA0, respectively. The `2L-Li__neb-QLs` errors are much smaller (0.015 and 0.107 eV). The three gap2QL paths are comparatively consistent, with absolute base barrier-like errors of 0.0018–0.0549 eV for Foundation and 0.0741–0.1768 eV for MPA0.

The perturbation response is model-dependent and strongly path-dependent. Across the eight paths, the median absolute per-image response is 0.0090 eV for Foundation `perturb_a`, 0.0128 eV for Foundation `perturb_b`, 0.0169 eV for MPA0 `perturb_a`, and 0.0146 eV for MPA0 `perturb_b`. Large responses occur on the same problematic paths, so they should be interpreted as sensitivity/OOD diagnostics rather than equilibrium phonon-like perturbation tests.

## Files

Publication-ready outputs are in `processed/figures/`: `figure_1_stability_parity_perturbation.{pdf,svg,png}`, `figure_2_relative_energy_profiles.{pdf,svg,png}`, `table_1_model_summary.{tex,csv}`, `table_s1_dft_stability.{tex,csv}`, `table_s2_perturbation_response.{tex,csv}`, and `captions_and_discussion.md`.

- `scf_neb_iteration_energy_audit.csv`: outer-iteration, image-call, and final-table stability audit.
- `scf_neb_image_energy_deltas.csv`: per-image penultimate/final energy differences.
- `path_sampling/path_sampling_manifest.csv`: all 240 sampled frames and hashes.
- `processed/mace_dft_relative_image_metrics.csv`: image-level DFT/MACE relative energies and forces.
- `processed/mace_dft_relative_path_summary.csv`: path-level MAE and barrier comparison.
- `processed/mace_perturbation_response.csv`: perturbation response by image.
- `processed/model_status.csv`: model success/failure audit.

The next scientifically clean step is to use this as a screening/diagnostic dataset, while reserving the terms “converged NEB barrier” and “DFT reference accuracy” for a separate campaign with explicit electronic and force convergence evidence.

The complete follow-up matrix evaluates all ten saved NEB paths, including both excluded paths, with all nine local Foundation models (900/900 successful single points). The canonical path classes are `in_gap_diffusion` (three paths), `gap_to_QL` (three paths), and `within_QL` (three paths); the bare Cu `neb` path is retained as an explicitly unclassified control. See `all_dft_foundation_models/`, `all_path_diagnostic_framework/`, and `all_path_structural_diagnostics/` for the merged figures and class-stratified diagnostics.
