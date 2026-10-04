# Phase 2 integrator/thermostat sensitivity results

## Completion gate

Jobs 31583430 (array 0--23) and 31586193 (array 24--47) completed 48/48
tasks with exit code 0. The remote root contains 48 `metrics.csv` files and
48 `completion.json` files; every metrics file has 202 saved rows. The audit
also finds consistent input, positions, momenta, and cell hashes within each
of the two snapshot groups.

## Scientific result

| Input step | Foundation-OMAT | From-scratch | Naive-FT | Multi-T |
|---:|---|---|---|---|
| 461800 | 6/6 runaway | 6/6 runaway | 6/6 stable | 6/6 stable |
| 462000 | 6/6 runaway | 6/6 runaway | 6/6 bounded / structural shift | 6/6 bounded / structural shift |

Foundation-OMAT and from-scratch cross the catastrophic gates under NVE,
Langevin with zero friction, weak friction (0.001 fs^-1), current friction
(0.01 fs^-1), and the tested dt range (0.5, 1.0, and 2.0 fs). dt=0.5 fs
delays some crossings but does not restore a finite bounded handoff. This
rules against attributing the observed failure only to the current Langevin
bath and supports a stronger model-dependent PES-instability interpretation
for these states.

This remains a short-range counterfactual stability result. The tested
integrators do not prove thermostat/timestep irrelevance, do not prove that
the initial configurations are DFT minima, and do not identify the original
long-trajectory chemical mechanism. Naive-FT and Multi-T are stable for the
461800 state and bounded/structurally shifted for the 462000 state; neither
classification is a diffusivity or DFT-validation result.

For the two unstable models, the first-gate timing remains finite across all
six settings rather than appearing only at the current Langevin condition:

| Input | Model | First-gate range (absolute step) | Median first-gate step | Gate types |
|---:|---|---:|---:|---|
| 461800 | Foundation-OMAT | 461970--462210 | 462030 | force (5/6), velocity (1/6) |
| 461800 | From-scratch | 461910--462210 | 462020 | force (6/6) |
| 462000 | Foundation-OMAT | 462010--462040 | 462020 | force (6/6) |
| 462000 | From-scratch | 462020--462060 | 462050 | force (6/6) |

The `dt=0.5 fs` runs generally delay the first gate but do not restore a
bounded trajectory. This supports rejecting a failure explanation based only
on the current thermostat or timestep, while retaining the stated limitation
that numerical settings can change the detailed crossing time and later
trajectory.

The machine-readable evidence is in `audit_20261004_final/`, especially
`sensitivity_summary.csv`, `sensitivity_summary.json`, and
`audit_summary.json`.
