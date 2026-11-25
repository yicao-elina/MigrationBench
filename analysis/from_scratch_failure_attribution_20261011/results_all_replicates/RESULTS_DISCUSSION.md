# From-scratch failure attribution: audited Results and Discussion

## Scope and protocol

This audit preserves the original from-scratch campaign and evaluates exact-protocol reproductions in separate roots. The protocol uses the same 2050-atom fully periodic input, model-family/training-seed pairing, initial momenta, `md_seed=20261005`, 600 K Langevin NVT, 1 fs timestep, friction 0.01 fs^-1, 200-step saved-frame interval, and 1,000,000-step target. The scientific gate is `T > 5000 K` or recorded `max |F| > 100 eV Å^-1` or `d_min < 0.5 Å` or a non-finite value.

Threshold onset is the first saved frame that crosses a recorded gate. It is not automatically a physical phase transition, and a trajectory stopped at that frame is classified as unstable rather than assigned a diffusion coefficient.

## Replica ledger

- training seed 123: original: unstable_threshold, reproduce-1: complete_to_target, reproduce-2: unstable_threshold
- training seed 234: original: unstable_threshold, reproduce-1: unstable_threshold, reproduce-2: unstable_threshold
- training seed 345: original: unstable_threshold, reproduce-1: unstable_threshold, reproduce-2: unstable_threshold

## Threshold-onset results

| source | family | training seed | onset (ps) | trigger | max T (K) | max recorded |F| (eV Å^-1) | min distance (Å) |
|---|---|---:|---:|---|---:|---:|---:|
| original | ablation_lmax_2 | 123 | 795.6 | temperature+force+minimum_distance | 7733.6 | 180.9 | 0.476 |
| reproduce-1 | ablation_lmax_2 | 123 | 1000.0 | not_observed | 750.6 | 3.9 | 2.235 |
| reproduce-2 | ablation_lmax_2 | 123 | 111.4 | force | 3681.7 | 313.1 | 0.600 |
| original | ablation_lmax_2 | 234 | 52.4 | force | 2094.6 | 134.2 | 0.905 |
| reproduce-1 | ablation_lmax_2 | 234 | 99.6 | force+minimum_distance | 2094.7 | 282.4 | 0.329 |
| reproduce-2 | ablation_lmax_2 | 234 | 37.6 | force | 1702.4 | 116.9 | 0.769 |
| original | baseline | 345 | 83.0 | force | 820.9 | 108.7 | 0.695 |
| reproduce-1 | baseline | 345 | 386.6 | force | 4888.4 | 1528.2 | 0.535 |
| reproduce-2 | baseline | 345 | 19.8 | force | 835.9 | 124.1 | 0.759 |

## Reproducibility and trigger classification

The initial-state audit passed exact positions/momenta/cell/species/PBC checks for 9/9 available trajectories; this separates onset differences from an input-state mismatch.
- Seed 123 (ablation_lmax_2) has 3 audited run(s), onset range 111.4–1000.0 ps and saved-frame spread 888.6 ps; observed triggers are `force; temperature+force+minimum_distance`.
- Seed 234 (ablation_lmax_2) has 3 audited run(s), onset range 37.6–99.6 ps and saved-frame spread 62.0 ps; observed triggers are `force; force+minimum_distance`.
- Seed 345 (baseline) has 3 audited run(s), onset range 19.8–386.6 ps and saved-frame spread 366.8 ps; observed triggers are `force`.

These rows test repeatability of the model/input/protocol combination. They do not merge the baseline seed-345 model with the two `ablation_lmax_2` model branches as if they were one neural-network parameterization.

## Precursor and configuration-distribution evidence

- Seed 123: precursor/threshold selected frames outside the training q99 gate = 100.0%; median distance to train = 12.460; median distance to NEB = 615.711; Cr–Te coordination median changed from 5.34 in stable-pre to 5.26 near threshold.
  Observed stable-pre to precursor/threshold median shifts: Cr_Te_coord_3p2_mean: -0.08; CrTe_bond_median_A: +0.005; CrTe_angle_mean_deg: -0.112; min_CrTe_A: -0.0151; Cr_velocity_median_A_ps: +0.697. These are descriptive recurrence checks, not causal effects.
- Seed 234: precursor/threshold selected frames outside the training q99 gate = 100.0%; median distance to train = 12.205; median distance to NEB = 615.809; Cr–Te coordination median changed from 5.12 in stable-pre to 5.04 near threshold.
  Observed stable-pre to precursor/threshold median shifts: Cr_Te_coord_3p2_mean: -0.08; CrTe_bond_median_A: +0.00183; CrTe_angle_mean_deg: -0.109; min_CrTe_A: -0.0277; Cr_velocity_median_A_ps: -0.996. These are descriptive recurrence checks, not causal effects.
- Seed 345: precursor/threshold selected frames outside the training q99 gate = 100.0%; median distance to train = 11.976; median distance to NEB = 615.770; Cr–Te coordination median changed from 5.32 in stable-pre to 5.24 near threshold.
  Observed stable-pre to precursor/threshold median shifts: Cr_Te_coord_3p2_mean: -0.08; CrTe_bond_median_A: +0.00524; CrTe_angle_mean_deg: -0.0586; min_CrTe_A: -0.1; Cr_velocity_median_A_ps: +1.35. These are descriptive recurrence checks, not causal effects.

The descriptor analysis is a standardized PCA nearest-neighbour diagnostic fitted on training descriptors. A frame outside a training q95/q99 gate is evidence of observed distribution novelty, not calibrated uncertainty and not proof that the training set caused the runaway. NEB proximity is also a geometric comparison, not evidence that the MD trajectory follows a physical NEB mechanism.

## Evidence boundary and limitations

The strongest supported conclusion is limited to whether the same recorded model/input/initial-state/protocol combination repeatedly crosses the operational instability gate. A common precursor may be reported only when the same feature direction is present across independent runs; a final threshold frame alone is insufficient. The threshold frame is saved at the configured 0.2 ps resolution and can combine force, temperature, and short-distance criteria.

No Dxy is reported for unstable branches. HFACF, thermal conductivity, and physical phase-transition claims require separate heat-flux/DFT/convergence evidence. Optional DFT snapshots are prepared as inputs only and are not silently submitted or interpreted here.
