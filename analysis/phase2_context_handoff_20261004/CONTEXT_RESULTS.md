# Phase-2 context handoff results — 460000 / 462200

This additive campaign closes the P0 context-snapshot scope gap. It uses the
same four-model, 2 ps protocol as the canonical 461800/462000 handoff: no
velocity resampling, 600 K Langevin, friction 0.01 fs^-1, 1 fs timestep, 2000
steps, and metrics every 10 steps.

## Audit result

All 8/8 runs completed with `completed=true`, `last_step=source_step+2000`,
202 metric rows, finite required metrics, and 1616 frame-aligned rows in
`frame_aligned_metrics.csv`. For both snapshots, input, position, momentum,
and cell hashes are identical across the four models. The serialized first
frame matches the exact source frame in species, positions (1e-6 A), momenta
(1e-6 amu A ps^-1), cell (1e-8 A), PBC, atom count, and source step. The exact
source file SHA256 also matches every run's `handoff_input.json`.

| Snapshot | Foundation-OMAT | From-scratch | Naive-FT | Multi-T-FT |
|---|---|---|---|---|
| 460000 | runaway, first force gate 460180 | stable short handoff | stable short handoff | stable short handoff |
| 462200 | runaway at source step | runaway at source step | runaway at source step | runaway at source step |

The 462200 state is already a stress/collapse context state at the handoff
boundary: its input minimum distance is 0.629 A, below the 1.5 A physical
minimum-distance gate. Its four classifications are therefore diagnostic
counterfactuals only and do not enter diffusion, DFT input selection, or the
formal causal claim. The 460000 anomaly window gives the additional local
record: Foundation-OMAT first crosses the force gate at step 460180 with Te
atom 582 as the maximum-force atom (221.410 eV A^-1, minimum distance 2.113
A); the subsequent 460190/460200 frames show the force/contact cascade. The
other 460000 models have no catastrophic gate crossing in the 2 ps window.
The canonical causal comparison remains the identical 461800/462000 handoff.

The extracted local windows and Cr/Sb/Te neighborhoods are in
`context_anomaly_local_neighborhood.json`; they contain the 460000
460160--460200 window and 462200 source/+10/+20-step context frames. Because
462200 is already below the physical distance gate at its source step, it is
explicitly excluded from the DFT reconstruction package.

| Snapshot | Model | First diagnostic step | Max-force atom at window peak | Peak force (eV A^-1) | Minimum distance (A) |
|---:|---|---:|---|---:|---:|
| 460000 | Foundation-OMAT | 460180 | Te 582 | 221.410 | 2.113 |
| 462200 | Foundation-OMAT | source | Sb 230 | 468.446 | 0.629 |
| 462200 | From-scratch | source | Sb 601 | 404.831 | 0.976 |
| 462200 | Naive-FT | source | Sb 230 | 4111.177 | 0.629 |
| 462200 | Multi-T-FT | source | Sb 229 | 1860.615 | 0.629 |

Machine-readable evidence is in `audit_results.json`, `audit_results.csv`, and
`frame_aligned_metrics.csv`; the audit is rerunnable with
`python audit_context_handoff.py`.
