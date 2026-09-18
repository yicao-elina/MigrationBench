# Prospective DFT Validation Of The Cr Site Proposer

## Scientific Question

Does the frozen MACE-label site proposer increase the probability that an
expensive QE relaxation reaches an accepted local minimum, compared with
geometry-matched controls that were not recommended?

This is an endpoint-discovery experiment. It is deliberately separate from
the same-structure direct-versus-coordinate-transform experiment in
`docs/endpoint_site_transform_ab_experiment.md`. Different starting sites cannot
be used to claim a relaxation speedup for one fixed structure.

## Same-Structure Transform Arm

A separate paired experiment now asks whether a deterministic local-coordinate
transform reduces DFT relaxation work for the six frozen proposal sites. It is
not part of the proposal-versus-control yield estimand. For each proposal
coordinate `r0`, only Cr is changed by solving

`r* = argmin_(||MIC(r-r0)|| <= 0.35 A) [2||MIC(r-r0)||^2 + sum_j softplus((2.80 A-d_j(r))/0.25 A)^2]`.

The direct proposal remains the original frozen QE input. The transformed arm
uses the same host, cell, pseudopotentials, cutoffs, k points, spin treatment,
ionic settings, resources, seed, and stopping budget. The six shifts are
`0.006`, `0.048`, `0.227`, `0.090`, `0.092`, and `0.235 A`; all transformed
starting structures retain minimum Cr-host distances above `2.60 A`.

The joint input audit finds one complete calculator identity,
`03738b164d4bdccb`, across all direct and transformed inputs. Rockfish jobs
`30861875`, `30861880`, `30861881`, `30861894`, `30861902`, and `30861915`
are the transformed branches paired to direct proposal jobs `30854592`,
`30854597`, `30854603`, `30854608`, `30854612`, and `30854617`.

The primary work measures are cumulative ionic steps and SCF iterations.
Walltime is secondary. A speedup is defined only after both branches converge
below `0.05 eV/A` and are assigned to the same repeat-accepted basin; otherwise
the result is censored or classified as a basin switch. These new prospective
DFT outputs remain excluded from the frozen v1 proposer.

`scripts/migrationbench/update_site_proposer_transform_ab.py` is the canonical
updater. It validates the exact six direct/transformed job bindings, syncs each
run under `data_processed/cluster`, accumulates restart work, verifies every
runtime segment, consumes repeat-acceptance records when available, and writes
the comparison CSV, force curves, and `summary.json`. Rockfish job `30862166`
passed the complete `99`-test suite for this state machine.

Every update also rebuilds the evaluation-only `relaxation_pairs` staging table
under `data_processed/hf_dataset_staging/site_proposer_transform_ab_v1_s42`.
Rockfish job `30862305` passed all `100` tests for the integrated table builder
and fail-closed speedup/leakage rules.

Direct and transformed repeats use explicit `direct` and `geomvoid` lineage
tags in their branch ID, Slurm job name, QE prefix, and manifest. This prevents
two members with the same site/image/seed from adopting the same repeat job.
Rockfish job `30862388` passed all `101` tests after this correction.

## Input Audit And Repair

All 245 mirrored site-search `relax.in` files use `ibrav=0` but omit
`CELL_PARAMETERS`. They are not directly executable QE inputs. The descriptor
model was still geometrically defined because its builder separately supplied
`data_raw/site_search_qe/2L_octo_Cr1.in` as the reference cell.

The prospective preparation script now fails closed if a source already has a
cell, requires exactly one reference-cell block, injects that immutable block,
and records the source, reference, cell-block, and final-input SHA-256 values.
The final 12 inputs preserve every source atomic coordinate exactly.

## Frozen Selection And Matching

The proposer and its out-of-fold predictions are frozen before any prospective
DFT output exists. Candidate rows with historical output, linked or unlinked,
are excluded. Proposal/control pairs use distinct connected void families.

For proposal `p` and control `c`, the assignment cost is

`C(p,c) = 25 L(p,c) + ||z_p-z_c||_2 + 0.25 |E_MACE,p-E_MACE,c|`,

where `z` is the standardized 71-dimensional invariant descriptor and `L=0`
requires the same source namespace and geometry label. A global one-to-one
assignment is solved jointly rather than greedily. The registered analysis set
requires `L=0` and descriptor distance at most `2.0`. Among feasible six-pair
sets, the one with the smallest total recommendation rank is frozen.

| Pair | Proposal | Control | Geometry | Descriptor distance | MACE energy delta, control-proposal (eV) | Jobs |
|---|---|---|---|---:|---:|---|
| 1 | `unique_119` | `unique_104` | octahedral | 0.460 | -0.000793 | `30854592 / 30854596` |
| 2 | `unique_093` | `unique_135` | cn7_te4 | 0.480 | -0.003204 | `30854597 / 30854600` |
| 3 | `unique_088` | `unique_078` | cn7_te4 | 0.526 | +0.016205 | `30854603 / 30854604` |
| 4 | `unique_200` | `unique_204` | cn8_te4 | 1.119 | -0.004547 | `30854608 / 30854611` |
| 5 | `unique_166` | `unique_042` | cn7_te3 | 1.056 | +0.002899 | `30854612 / 30854615` |
| 6 | `unique_098` | `unique_129` | cn6_te3 | 1.180 | -0.001557 | `30854617 / 30854619` |

## Calculator And Execution

Rockfish CPU smoke `30854580` regenerated the batch from raw mirrored inputs
and passed the selection/cell test. The 12 Rockfish and local final QE inputs
are byte-identical. Calculator audit result:

- identity: `03738b164d4bdccb`;
- composition: `Cr1Sb32Te48`;
- cutoff: `70/280 Ry`;
- k points: `2x2x2`, shifted `1 1 1`;
- spin: collinear `nspin=2`, no SOC;
- 12 inputs, zero parse errors, zero incomplete identities.

Rockfish CPU smoke `30854874` then validated the offline comparator (`3 passed
in 3.53 s`). The comparator withholds yield estimates while any branch is
running or restartable and requires each accepted row's local input hash,
runtime-recorded input hash, and `pw.x` binary provenance to be complete.

Each production relaxation uses 2 MPI ranks, 160 GB, 24 hours, `nstep=200`, and
`max_seconds=84600`. All output is under `/scratch16/pclancy3/yi`. The initial
post-submission snapshot found all 12 jobs running. The next health query is not
allowed before `2026-09-13T19:58:00Z`.

## Estimands And Acceptance

The primary estimate is paired repeat-validated local-minimum yield. Secondary
outputs are cumulative ionic steps, cumulative SCF iterations, and final-basin
deduplication. A successful first relaxation is `repeat_required`, not a final
stable-site label. Every first-relax success receives a from-scratch repeat
under the same calculator. Final acceptance additionally requires the repeat
force, displacement, energy, geometry, parent-job, status-hash, calculator,
runtime, and exact-input gates. Clean max-seconds or max-step results are
continued and counted over their entire lineage.

Prospective outputs remain an audit-only holdout for the frozen v1 proposer.
They may train a later, separately versioned DFT model only after the v1
prospective evaluation is reported. They are not added to the current
cross-validation folds and cannot influence this batch's selection.

## Provenance

- Authoritative batch: `data_processed/site_stability_proposer/prospective_dft_v5_s42/`
- Preparation script: `scripts/migrationbench/prepare_site_proposer_dft_validation.py`
- Comparator: `scripts/migrationbench/compare_site_proposer_dft_validation.py`
- Submitted registry: `data_processed/site_stability_proposer/prospective_dft_v5_s42/submitted_endpoint_relax_jobs.json`
- Passing Rockfish smoke: `data_processed/cluster/mb_siteprep_s42_30854580/`
- Comparator Rockfish smoke: `data_processed/cluster/mb_siteprep_s42_30854874/`
- Repeat-required comparator smoke: `data_processed/cluster/mb_siteprep_s42_30857156/`
- Endpoint-stage planner smoke: `data_processed/cluster/mb_siteprep_s42_30857186/`
- Strict acceptance/batch/basin Rockfish smoke: `data_processed/cluster/mb_epaccept_sm_s42_30856509/`
- Failed missing-fixture smoke retained at `data_processed/cluster/mb_siteprep_s42_30854347/`

Local development batches v1-v4 are non-authoritative: v1 exposed the missing
cell, v2 added the cell but retained local path coupling, v3 fixed portability,
and v4 exposed a poor greedy/control-caliper design. None was submitted to QE.

The comparator interface accepts `--endpoint-acceptance` pointing to the
repeat-relax `endpoint_acceptance_batch.json`. Without it, any successful first
relaxation remains censored and `formal_result_available` stays false. Rockfish
job `30857156` validated accepted, rejected-repeat, and missing-repeat behavior
under Python 3.9 (`3 passed`, exit `0:0`).

`plan_endpoint_discovery_progress.py` is the canonical read-only stage planner.
It requires exact frozen-batch coverage and emits one of: wait, continue clean
first relaxations, prepare repeats, continue clean repeats, build acceptance,
or assign basins and run the final comparator. It never submits or reclassifies
a job on its own. Rockfish CPU job `30857186` validated this progression together
with the comparator (`4 passed`, exit `0:0`).

For the six-pair same-structure transform experiment,
`plan_site_transform_ab_repeats.py` first extracts exactly the six proposal
parents from the 12-site discovery batch and excludes all six matched controls.
It then plans the direct and `geomvoid` repeat arms independently, using distinct
arm tags while retaining the exact parent job IDs. The direct repeats are shared
with endpoint discovery and must be submitted only once; the two workflows adopt
the same resulting acceptance records rather than launching duplicate QE work.
Rockfish CPU Slurm job `30862405` validated this rule as part of the complete
suite (`102 passed`, zero skipped, exit `0:0`).

The two arms intentionally build separate repeat-acceptance batches. The final
A/B updater passes them as `--direct-endpoint-acceptance` and
`--transformed-endpoint-acceptance`; every comparison and Hugging Face staging
row retains both artifact references. This prevents a completed experiment from
being blocked by a nonexistent combined file and preserves each arm's parent-job
binding.

If the discovery repeat-status file contains proposal and control jobs, the
paired planner filters it by exact `(path_id, image_index_qe)` keys into a
read-only six-proposal view and records the source file SHA-256. The acceptance
builder operates on that view, while all QE repeat directories remain shared.
Rockfish job `30864774` passed the complete 103-test suite for this behavior.
