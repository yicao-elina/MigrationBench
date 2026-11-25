# Phase 2 integrator/thermostat sensitivity campaign

This is a new, isolated campaign. It does not reuse or overwrite the
canonical retry-2 outputs. Rockfish root:
`/scratch16/pclancy3/yi/phase2_integrator_sensitivity_20261003`.

Slurm jobs `31583430` (array 0--23) and `31586193` (array 24--47) together
contain 48 runs: two exact handoff snapshots (461800, 462000), four fixed
model files, and six settings:

| setting | integrator | dt (fs) | Langevin friction (fs^-1) |
|---|---|---:|---:|
| `nve_dt0p5` | VelocityVerlet | 0.5 | 0 |
| `nve_dt1p0` | VelocityVerlet | 1.0 | 0 |
| `nve_dt2p0` | VelocityVerlet | 2.0 | 0 |
| `langevin_f0_dt1p0` | Langevin | 1.0 | 0 |
| `langevin_f0p001_dt1p0` | Langevin | 1.0 | 0.001 |
| `langevin_f0p01_dt1p0` | Langevin | 1.0 | 0.01 |

Every run retains the same 2 ps duration, 10-step saved-frame interval,
600 K target when a bath is present, unresampled initial momenta, input
identity checks, and Phase-2 scientific gates. `audit_sensitivity.py` should
only be run after all 48 output directories contain complete 202-row metrics
and `completion.json`.

Interpretation:

- divergence only under Langevin would support a bath contribution;
- persistence under NVE but recovery at smaller dt would support timestep
  sensitivity;
- persistence under NVE, weak bath, and small dt supports a stronger
  model-PES instability interpretation;
- any apparent stability still requires checking local environment identity
  and, for the final mechanism claim, a minimum-distance-gated DFT comparison.
