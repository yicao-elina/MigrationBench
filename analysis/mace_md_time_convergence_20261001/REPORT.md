# MACE-MD time-convergence audit: Scratch and FT-MultiT

Date: 2026-10-01

## Scope

Analyzed the continuation segments beginning at step 200,000 (200 ps), using the 2050-atom fixed-cell trajectory. The extxyz files contain 50 Cr atoms; MSD is computed as the mean of per-Cr squared in-plane displacements after PBC unwrapping, not as the squared displacement of the Cr center of mass. The reported conversion is `D_xy = slope / 4`.

The legacy Scratch source lacks step metadata. Its time axis is reconstructed from the documented 200-step save interval; continuation steps are read from the absolute-step field.

## Results

### Scratch

The trajectory reaches 497.2 ps after the 200 ps starting point, but it is not scientifically valid through 500 ps. At step 489,000 (289 ps after the analysis start), all 50 Cr atoms show a simultaneous approximately 56 Å frame-to-frame in-plane jump. This is a trajectory discontinuity or runaway event, not ordinary diffusion noise.

Before that event, cumulative fits are effectively non-diffusive and have near-zero R²:

| window | Dxy (cm²/s) | R² |
|---:|---:|---:|
| 50 ps | -1.23e-9 | 0.001 |
| 100 ps | 8.45e-10 | 0.003 |
| 200 ps | 5.73e-10 | 0.005 |

The 500 ps value (`4.99e-2 cm²/s`) is dominated by the discontinuity and must be rejected. The 50 ps block slopes are near zero through 250 ps and jump to approximately `0.095 cm²/s` after the anomaly. This model fails the current trajectory-integrity gate; no diffusivity conclusion should be drawn from this segment.

### FT-MultiT

The available continuation reaches 252.6 ps after the 200 ps starting point, so a 500 ps test is not yet available. Cumulative Dxy fits are:

| window | Dxy (cm²/s) | R² |
|---:|---:|---:|
| 50 ps | 1.26e-6 | 0.924 |
| 100 ps | 1.41e-6 | 0.950 |
| 200 ps | 1.36e-6 | 0.957 |
| available 252.6 ps | 1.37e-6 | 0.969 |

The 50 ps chronological blocks are `1.26e-6`, `0.61e-6`, `0.95e-6`, `1.41e-6`, and `0.84e-6 cm²/s`. The cumulative estimate changes by only about 8% from 50 to 200 ps and about 1% from 200 to 252.6 ps, but individual block estimates vary by roughly a factor of 2.3. Therefore 200 ps looks approximately stable at the cumulative-fit level, but the block uncertainty is not small enough to call the result fully converged.

## Decision

- Do not use the Scratch 500 ps MSD or diffusivity; first diagnose the step-489,000 discontinuity and rerun/repair from the last valid checkpoint.
- FT-MultiT is the stronger candidate: its cumulative MSD slope is approximately stable by 200 ps, but it still needs the 500 ps branch and replica uncertainty before final reporting.
- The current analysis is a time-convergence diagnostic, not final diffusivity evidence: it has one trajectory per model and no hopping-event identity audit.

## Artifacts

- [Scratch convergence JSON](./from-scratch/time_convergence.json)
- [Scratch 50 ps blocks](./from-scratch/blocks_50ps.csv)
- [Scratch convergence figure](./from-scratch/time_convergence.png)
- [FT-MultiT convergence JSON](./multi-T/time_convergence.json)
- [FT-MultiT 50 ps blocks](./multi-T/blocks_50ps.csv)
- [FT-MultiT convergence figure](./multi-T/time_convergence.png)
