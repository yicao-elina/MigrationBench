# Phase 2/3 paper-ready evidence package (2026-10-03)

## Main result

Under identical positions, momenta, cell, species, timestep, target
temperature, and Langevin protocol, the two input states separate by model
within the first 0.02--0.32 ps of the 2 ps handoff. Foundation-OMAT and the
from-scratch model cross the predefined catastrophic gate, whereas naive-FT
and Multi-T remain finite for the 461800 state and remain bounded, although
structurally shifted, for the 462000 state. This is evidence for a large
model-dependent dynamical-stability difference near these states. It is not a
claim that the initial state is a DFT minimum, nor does it by itself identify
the original long-trajectory mechanism.

## Canonical 2 ps handoff table

The first-divergence step is the earliest sampled frame at which any gate was
crossed: maximum force >= 100 eV Å^-1, maximum velocity >= 1000 Å ps^-1,
minimum pair distance < 1.5 Å, temperature >= 2000 K, total-energy span >=
250 eV, maximum unwrapped displacement >= 50 Å, or at least 100 atoms more
than 5 Å from their initial unwrapped positions.

| Input step | Model | Classification | First gate crossing | Δt from input | Gate at first crossing |
|---:|---|---|---:|---:|---|
| 461800 | Foundation-OMAT | runaway | 462120 | 0.32 ps | max force >= 100 eV Å^-1 |
| 461800 | From-scratch | runaway | 462060 | 0.26 ps | max force >= 100 eV Å^-1 |
| 461800 | Naive-FT | stable short handoff | — | — | none |
| 461800 | Multi-T | stable short handoff | — | — | none |
| 462000 | Foundation-OMAT | runaway | 462020 | 0.02 ps | max force >= 100 eV Å^-1 |
| 462000 | From-scratch | runaway | 462050 | 0.05 ps | max force >= 100 eV Å^-1 |
| 462000 | Naive-FT | bounded / structural shift | — | — | none |
| 462000 | Multi-T | bounded / structural shift | — | — | none |

The canonical trajectories contain 202 frames per run (initial frame plus
200 saved integration frames). First-frame normalized hashes match the input
positions, momenta, cell, species, and PBC for all 8/8 runs. The aligned
frame table and hash audit are retained in `results/`. The additive context
campaign extends the same frame-aligned and initial-state audit to the
neighboring retained snapshots 460000 and 462200; it contains 8/8 complete
runs and 1,616 aligned rows in
`analysis/phase2_context_handoff_20261004/frame_aligned_metrics.csv`.

## Context-snapshot audit

The neighboring snapshots provide a bounded context check rather than a new
causal claim. Starting from 460000, Foundation-OMAT crosses the force gate at
step 460180 while from-scratch, naive-FT, and Multi-T remain finite and below
all catastrophic gates for the 2 ps handoff. The 462200 state is already a
stress/collapse context state at the handoff boundary: its input minimum
distance is 0.629 Å, below the 1.5 Å physical gate, and all four models cross
a diagnostic force gate at the source step. Those four runs are therefore
retained only as counterfactual context and are not evidence that the models
create the initial stress state; they are excluded from DFT input selection.
In the physically admissible 460000 context window, Foundation-OMAT first
crosses the force gate at step 460180, with Te atom 582 as the maximum-force
atom, while the other three models remain below the catastrophic gates for 2
ps. For both snapshots, the input, position, momentum, cell, species, and PBC
checks pass across all four models. The complete audit and local neighborhoods
are in `analysis/phase2_context_handoff_20261004/CONTEXT_RESULTS.md`,
`audit_results.json`, and `context_anomaly_local_neighborhood.json`.

Taken together with the canonical 461800/462000 handoff, the publishable
conclusion is: under exactly matched positions, momenta, cell, species and MD
parameters, Foundation-OMAT and from-scratch can undergo catastrophic
divergence within 2 ps while naive-FT/Multi-T remain finite or only structurally
shifted. This supports a model-dependent dynamical-stability difference near
the tested states, but does not prove that the initial state was a DFT minimum
or that the original long-trajectory mechanism has been uniquely identified.

## Mechanism-level interpretation

The first anomalous region is local and model-dependent rather than an
endpoint-only numerical artifact. For the 461800 input, Foundation-OMAT's
first force-gate frame is dominated by Cr atom 1310, while from-scratch is
dominated by Sb atom 601. For the 462000 input, the corresponding atoms are
Cr 245 for Foundation-OMAT and Sb 229 for from-scratch. In each case the
force excursion precedes the later sub-1.5 Å contact and thermal/displacement
runaway in the saved 10-step frames. The neighborhood records, including
Cr/Sb/Te neighbor identities and distances before, at, and after the first
gate, are in `results/first_anomaly_local_neighborhood.jsonl`.

This supports the narrower statement that the model forces amplify a local
perturbation differently on the same initial state. It does not support
assigning a unique chemical mechanism until the integrator/thermostat audit
and minimum-distance-gated DFT comparison are complete.

## Phase 1 boundary

The Phase 1 static-force comparison is a same-geometry cross-model response
test. Its 0.629 Å minimum-distance frame is an explicitly nonphysical
instability regime and is excluded from formal DFT or diffusion statistics.
Static force disagreement does not, by itself, prove that the common input
geometry had already collapsed; the manuscript should state this caveat
directly.

## Phase 3 DFT boundary

The real 81-atom QE array jobs 31537708 and 31537709 ended as `CANCELLED`
after 00:47:53. No accepted DFT single-point result with finite energy and
forces, explicit SCF convergence, normal QE termination, and verified
identity/cell provenance is available from that array. The 81-atom local-tile
inputs are documented approximations and are not a substitute for a formal
full-cell or identity-preserving DFT comparison. Accordingly, the formal DFT
validation gate remains **0 accepted structures** and no DFT-based causal
claim is made here.

## Completed P1 sensitivity gate

The isolated Rockfish arrays 31583430 (indices 0--23) and 31586193 (indices
24--47) completed 48 runs with 48/48 `completion.json`, 202/202 metric rows
per run, and zero failed Slurm tasks. The matrix covers Foundation-OMAT,
from-scratch, naive-FT, and Multi-T at both input states, with NVE dt = 0.5,
1.0, and 2.0 fs, Langevin friction = 0, 0.001, and 0.01 fs^-1 at dt = 1 fs.
All 48 input records retain consistent positions/momenta/cell hashes within
each snapshot group.

| Input step | Foundation-OMAT | From-scratch | Naive-FT | Multi-T |
|---:|---|---|---|---|
| 461800 | 6/6 runaway | 6/6 runaway | 6/6 stable | 6/6 stable |
| 462000 | 6/6 runaway | 6/6 runaway | 6/6 bounded / shift | 6/6 bounded / shift |

Foundation-OMAT and from-scratch therefore remain runaway across NVE, zero
friction, weak friction, current friction, and the tested timestep range.
Reducing dt to 0.5 fs delays some first gate crossings (for example, both
models at step 461800), but does not restore a finite, bounded handoff. The
result is inconsistent with a failure caused only by the current Langevin
bath, and supports a stronger model-dependent PES-instability interpretation
for these states. It does not prove that timestep and thermostat effects are
irrelevant, and it does not identify the original long-trajectory mechanism.

Across the six settings, Foundation-OMAT first crosses a gate at steps
461970--462210 for the 461800 input and 462010--462040 for 462000; the
corresponding from-scratch ranges are 461910--462210 and 462020--462060.
The first gate is force-dominated in 5/6 Foundation-OMAT runs at 461800 and
all from-scratch runs, with one Foundation-OMAT 461800 run first crossing the
velocity gate. This timing spread is a numerical-sensitivity result, not a
claim that the exact gate step is invariant.

Naive-FT and Multi-T remain stable for the 461800 state across all six
settings. For 462000 they remain below the catastrophic gates across all six
settings but show bounded structural displacement; this is a bounded/shifted
counterfactual, not formal structural or diffusivity validation.

## Additive local-cell force diagnostic

To locate a DFT-comparable onset without using the nonphysical 0.629 Å frame,
we constructed four provenance-tracked 81-atom local-cell reconstructions:
the normal 461800 input, the Multi-T bounded/shifted 2 ps endpoint, and the
first force-gate frames for Foundation-OMAT and from-scratch from the 461800
handoff. All four pass the local 1.5 Å minimum-distance gate and have
full-trajectory-to-local mapping error below 6×10^-13 Å. Evaluating the same
local cells with the four MACE models gives a within-model force-response
diagnostic: the from-scratch model's maximum force increases by 19.7–24.3×
relative to the normal local cell at its runaway-onset reconstruction, while
Foundation-OMAT increases by 4.6–5.3×. These values are not cross-model
absolute energy rankings and are not a substitute for DFT; the full table,
model hashes, and force arrays are retained in the Phase 3 recovery package.

The corresponding QE array is isolated and remains separately gated. Until
its outputs show finite energy and all 81 forces, explicit SCF convergence,
normal `JOB DONE` termination, and matching input hashes, this local-cell
diagnostic is reported as `RECONSTRUCTION / REPRODUCED_SCOPE_LIMITED` and the
formal DFT gate remains zero.

## Figure and provenance outputs

- `figures/phase2_handoff_timeseries.{png,svg,pdf}`: four aligned diagnostics,
  both input states, and first-gate markers.
- `figures/phase2_stability_gate.{png,svg,pdf}`: model × input-state gate
  summary with first crossing steps.
- `figures/plot_env.json`: exact renderer, Plot Atlas source commit, Python,
  NumPy, Matplotlib, palette, and command.
- `results/frame_aligned_metrics.csv`: all 1,616 saved metric rows, not only
  endpoint summaries.
- `results/first_divergence_by_run.csv/.json`: first crossing and endpoint
  diagnostics.
- `results/first_frame_hash_audit.jsonl`: trajectory-first-frame versus input
  normalized hash checks.
- `results/first_anomaly_local_neighborhood.jsonl`: atom/local-neighborhood
  mechanism records.
