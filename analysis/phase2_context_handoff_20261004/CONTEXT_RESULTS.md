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

The 462200 state is already a stress/context state at the handoff boundary;
its four classifications are diagnostic counterfactuals and do not enter the
canonical diffusion or formal DFT claim. The 460000 result strengthens the
context evidence that model response depends on the selected state, while the
canonical causal comparison remains the identical 461800/462000 handoff.

Machine-readable evidence is in `audit_results.json`, `audit_results.csv`, and
`frame_aligned_metrics.csv`; the audit is rerunnable with
`python audit_context_handoff.py`.
