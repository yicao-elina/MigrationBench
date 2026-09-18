# Prospective Site Transform Versus Direct DFT Relaxation

## Scientific Question

For a frozen, representative Cr candidate site, does a deterministic local
geometry transform reduce the QE work needed to reach the same repeat-validated
DFT minimum relative to direct relaxation from the unmodified coordinate?

This is a paired initialization-efficiency experiment. It does not test whether
the transformed coordinate has a lower initial energy, and a branch reaching a
different basin is a mechanism change rather than a speedup.

## Transform

Only the Cr coordinate changes. The host, cell, atom ordering, pseudopotentials,
cutoffs, k points, convergence thresholds, CPU resources, and seed are identical
within a pair. With periodic minimum-image displacement, the transformed point is

`r* = argmin_(||MIC(r-r0)|| <= 0.35 A) J(r)`

`J(r) = lambda ||MIC(r-r0)||^2 + sum_j softplus((d_safe-d_j(r))/sigma)^2`,

where `lambda=2`, `d_safe=2.80 A`, and `sigma=0.25 A`. Projected gradient descent
uses an initial step of `0.03`, at most 200 iterations, and a gradient tolerance
of `1e-6`. The smooth contact term is a physics-motivated geometric prior, not a
surrogate DFT energy. A lower `J` therefore cannot be interpreted as a lower DFT
energy.

## Frozen Six-Pair Batch

| Candidate | Cr shift (A) | Minimum Cr-host distance, before -> after (A) | Geometric objective, before -> after | Direct job | Transformed job |
|---|---:|---:|---:|---:|---:|
| `unique_119` | 0.006047 | 2.799912 -> 2.804868 | 2.779697 -> 2.779172 | `30854592` | `30861875` |
| `unique_093` | 0.048114 | 2.666022 -> 2.674210 | 5.340567 -> 5.305587 | `30854597` | `30861880` |
| `unique_088` | 0.226957 | 2.602868 -> 2.609474 | 5.084904 -> 4.672118 | `30854603` | `30861881` |
| `unique_200` | 0.090157 | 2.643089 -> 2.706794 | 4.346899 -> 4.216004 | `30854608` | `30861894` |
| `unique_166` | 0.091876 | 2.643399 -> 2.695148 | 4.632714 -> 4.496431 | `30854612` | `30861902` |
| `unique_098` | 0.234812 | 2.619408 -> 2.614593 | 5.002268 -> 4.576918 | `30854617` | `30861915` |

The total smooth contact objective decreases for all six candidates. The nearest
distance for `unique_098` decreases slightly while remaining above 2.60 A because
the objective trades all local contacts against the displacement tether; the
method does not claim that every individual Cr-host distance increases.

Every job uses calculator identity `03738b164d4bdccb`, two CPU MPI ranks, 160 GB,
a 24-hour Slurm limit, QE `max_seconds=84600`, and `nstep=200`. The direct jobs are
also the proposal arm of the prospective stable-site experiment; their computation
and later direct repeat are shared, not duplicated.

The semantic input-pair audit independently verifies every frozen pair. All six
pass: input line counts and atom ordering match; cells and all 80 host coordinates
are identical; the only changed text roles are the collision-avoiding QE `prefix`
and one Cr coordinate; the measured periodic Cr displacement matches the transform
manifest within `1e-9 A`; both input hashes and the calculator identity match.
The queryable evidence is
`data_processed/site_stability_proposer/prospective_transform_ab_v1_s42/semantic_pair_audit/input_pair_audit.json`.

## Speed Measurement

For pair `i`, the primary reported ratios are

`S_ionic,i = N_ionic,direct / N_ionic,transformed`

and

`S_SCF,i = N_SCF,direct / N_SCF,transformed`.

Ratios greater than one favor the transformed initialization. We also record the
first ionic step at which maximum atomic force crosses 0.50, 0.20, 0.10, and
0.05 eV/A, plus the first sustained crossing after which no later evaluated
structure rebounds above the threshold. For both definitions we retain ionic
steps and cumulative SCF iterations. We also record cumulative walltime,
CPU-hours, energy drop, and the complete force and energy curve. Walltime is
secondary because node performance can vary. A sustained crossing on a running
prefix remains diagnostic because a future step can still rebound; it becomes
formal only after the complete relaxation passes every gate below.

A numerical speedup is released only when both arms:

1. finish BFGS relaxation with maximum force at or below 0.05 eV/A;
2. pass independent, hash-bound same-calculator repeat relaxation;
3. end in the same basin under periodic Cr distance, host RMSD, and local-distance
   signature gates;
4. carry exact QE input, Slurm job, `pw.x`, and cumulative runtime provenance for
   every cost-bearing segment; and
5. use a calculator identity accepted by the N24 production-method gate.

Otherwise the result is censored, rejected, or reported as a basin switch. A
running-prefix ratio is diagnostic only.

The direct and transformed arms produce separate acceptance batches because
their repeat jobs have separate parent IDs and namespaces. The comparator takes
both artifacts explicitly through `--direct-endpoint-acceptance` and
`--transformed-endpoint-acceptance`, verifies each member against its own parent,
and stores both paths in every dataset row. The legacy `--endpoint-acceptance`
option is accepted only for an already combined batch and cannot be mixed with
the arm-specific options.

When endpoint discovery produces a broader repeat-status table containing both
proposal and matched-control jobs, the A/B repeat planner creates a read-only,
source-hash-bound status view containing exactly the six direct proposal endpoint
keys. The underlying QE repeats are reused; no scientific output is copied or
recomputed. Acceptance is then rederived from that exact status view so its
first/repeat status hashes match the six-pair estimand. Rockfish CPU Slurm job
`30864774` validated this shared-repeat path in the full suite (`103 passed`,
zero skipped, exit `0:0`).

The arm-specific interface passed the complete self-contained suite on a real
Rockfish CPU node in Slurm job `30864698` (`103 passed`, zero skipped, exit
`0:0`). Preceding job `30864689` is retained as failed infrastructure evidence:
an accidental root-level duplicate of the test file was collected instead of
the correctly nested copy. Removing only those newly created duplicates restored
the intended environment; no scientific input or output was changed.

## Current Status

All 12 first-relax jobs were submitted on Rockfish. The six transformed jobs
passed the immediate health snapshot. Under the three-hour polling rule, no
transformed job is queried again before `2026-09-13T22:01:00Z`; therefore no DFT
speed result is yet available. The full paired monitor, comparator, dataset
builder, and independent-repeat planner passed Rockfish CPU Slurm job `30862405`
with `102 passed`, zero skipped, and exit `0:0`.

At the cadence-authorized `2026-09-13T20:00:11Z` direct/control snapshot, all
12 jobs were healthy and running after about three hours but remained inside
their first SCF, with only 2-3 electronic iterations and zero ionic steps. The
state planner therefore returned `waiting_for_first_relax`; no restart, repeat,
or duplicate submission was made. This is a performance concern, not evidence
of SCF divergence or endpoint instability.

To test CPU parallel efficiency without disturbing those calculations, diagnostic
job `30864875` uses the exact `unique_119` structure and Hamiltonian in a single
SCF with 4 MPI ranks and 4 k-point pools. It has a three-hour walltime,
`max_seconds=9900`, 180 GB, fixed seed 42, and `label_eligibility=false`. Its
calculator identity remains `03738b164d4bdccb`. Only after measured throughput
shows an advantage may later clean continuations adopt that resource layout;
the currently running 2-rank jobs are not cancelled.

The scaling probe is registered in `configs/active_qe_scf_scaling_jobs.json` and
is analyzed by `analyze_qe_scf_scaling_probe.py`. The decision records iteration
rates, completed-SCF status, walltime, and CPU-hours. It recommends 4r/4p only
for future clean continuations when the probe has no scientific failure, contains
at least two electronic iterations, and reaches at least 1.5 times the baseline
iteration rate. Rockfish CPU Slurm job `30864905` passed all 104 tests with zero
skips.

## Outputs

- Frozen transform inputs and full per-point metadata:
  `data_processed/site_stability_proposer/prospective_transform_ab_v1_s42/`
- Pair comparator output after an authorized snapshot:
  `data_processed/site_stability_proposer/prospective_transform_ab_v1_s42/comparison/`
- Evaluation-only Hugging Face staging rows:
  `data_processed/hf_dataset_staging/site_proposer_transform_ab_v1_s42/`
- Canonical updater: `scripts/migrationbench/update_site_proposer_transform_ab.py`
- Repeat planner: `scripts/migrationbench/plan_site_transform_ab_repeats.py`
- Semantic input-pair audit: `scripts/migrationbench/audit_site_transform_input_pairs.py`
