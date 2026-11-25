# Unified MD protocol and comparison boundary

## Current campaign

The formal from-scratch matrix contains 14 model artifacts from five ablation
families and three training seeds. Each run has a separate output directory,
model SHA256, input SHA256, training seed, independent MD seed, fixed 2050-atom
identity/PBC/cell checks, trajectory, checkpoint, metrics and status manifest.
The `ablation_correlation_3/seed123` artifact is blocked because legacy and
later scratch records overlap under the same source directory.

The protocol is fresh-start Langevin NVT at 600 K, 1 fs timestep, friction
0.01 fs^-1, saving every 200 steps. The common input is the read-only
`0-foundation-mace_mp` evaluation structure, not the protected old continuation
root. The three complete 20 ps runs are screening-pass candidates only; the
selected 1 ns runs were baseline/seed345 and lmax2/seed123,seed234. All three
selected 1 ns branches terminated at `instability_threshold` before 1,000,000
steps and are retained as `FAILURE_MODE`, not as production transport data.

## Observable definitions

`timeseries.csv` contains potential, kinetic and total energy, temperature,
maximum force, sampled minimum distance, and pressure when stress is available.
Cr positions are unwrapped through fixed-cell fractional coordinates. `msd.csv`
contains both 3D Cr MSD and explicitly XY Cr MSD; `Dxy` uses the 2D Einstein
factor 4. `vacf.csv` is the normalized Cr velocity VACF after removing the
per-frame Cr center-of-mass velocity. Candidate hops are displacement events
between saved frames and are not formal site hops without site mapping.

The original repository's stress-based heat-flux convention is retained as an
explicit approximation. The new runner's extxyz files contain stress readable
through ASE, so the unified postprocessor records pressure and can construct
HFACF/running κ with the exact model-independent formula. If stress is absent,
the output remains `unavailable`; no surrogate is substituted.

## Comparison boundary

The original paper CSVs are copied into `reference_original_paper/` with their
source path and SHA256 recorded in `campaign_manifest.json`. The older
FT-600K/FT-MultiT continuation metrics are kept as a labeled comparison
source. They do not share the new fresh-start input, model registry, or start
step, so their numerical transport values are not treated as same-protocol
replica statistics. Final figures distinguish `FORMAL_SAME_PROTOCOL`,
`REFERENCE_PROTOCOL_DIFFERENT`, `SCREENING_ONLY`, and `UNSTABLE_NO_DXY`. The
terminal-failure figure additionally uses `BLOCKED` and labels all finite
MSD/VACF/κ traces as diagnostics only.

## Scientific limits

Finite output, a complete step count, and a successful scheduler state are not
transport convergence. Dxy, HFACF and κ require window, block and replica
uncertainty checks. An unstable model has a retained failure-onset record and
is reported as `unstable — no Dxy`; stress-derived running κ remains a
repository diagnostic unless the full scientific gates pass.
