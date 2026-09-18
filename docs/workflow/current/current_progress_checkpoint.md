# Current Progress Checkpoint

Date: 2026-09-15 UTC

## What Is Solid Now

- The most serious reviewer issue is identified: the manuscript and SI mix fixed-geometry NEB-path scoring, self-relaxed/self-consistent NEB-path scoring, and possibly inconsistent reference-barrier interpretations.
- The fixed-path versus self-relaxed distinction plausibly explains why Fig. 3 and Fig. S3 can report different Foundation-model barriers and qualitative assessments. The distinction is a protocol explanation, not yet a numeric validation: the `~0.7 eV` fixed-path error and `0.41 eV` self-NEB claim still require lossless provenance/recalculation before use.
- The current defensible 1-4 fixed-path DFT candidate is about `0.336050 eV`, with convergence caveats.
- The manuscript should not use the old Fig. S3-style `0.41 eV Foundation exceptional` claim as an audited final value.
- Existing scripts can parse QE NEB outputs, detect QE overflow/path-length failures, generate QE NEB inputs from explicit images, run MLFF/MACE NEB, monitor active Slurm jobs, classify paths, and package dataset records.
- Trust-region MACE now has a separate, fail-closed QE handoff role. A schema-1.1
  matched bundle binds both QE inputs, both handoff manifests, the candidate,
  the dual-potential MLFF result, and its comparator artifact. Rockfish job
  `30853141` passed all 82 tests for this chain.
- Historical `1-4` trust-region jobs `30853021` (seed 42) and `30852994`
  (seed 52) completed identically. The optimization potential converged in 31
  steps, retained a `2.426 A` minimum pair distance, and reduced migrant
  coordinate drift from `0.750 A` in the unrestrained control to `0.168 A`.
  The base-MACE path itself is not converged (`0.190 eV/A` true-force maximum),
  and the endpoints are unverified, so this is a curvature donor only.
- After the trust jobs terminated, the frozen-runner boundary was removed.
  Rockfish job `30861741` passed the synchronized full suite with `97 passed`
  and zero skips; failed deployment smoke `30861664` is retained as evidence of
  a missing remote A/B-comparator sync, not a scientific calculation.
- The path-level QE-NEB A/B comparator is implemented and validated. It totals
  path and SCF work over restart lineages and reports speedup only after both
  branches converge to the same PBC-aware final mechanism under an accepted
  calculator identity. Rockfish job `30853151` passed all 84 tests.
- MACE pre-NEB has been shown to work as a practical initializer for at least the 1-7 continuation path; it does not yet replace DFT evidence.
- Path repair is now an explicit preconditioning gate: suspect intermediate images are first checked for minimum distances and all-atom/image-to-image jumps, then optionally repaired by physics heuristics before MACE or QE refinement.
- QE preflight failures were diagnosed: the first serious failure was memory pressure, and later 2-rank/160G preflights parsed and started SCF but timed out because the walltime was short.
- The checkpoint-scoped leakage audit is complete for all 1,110 current `1-6/1-7` QE frames against the declared fine-tuning train/valid/test files. No exact, SOAP-threshold, or compatible-local-RMSD overlap was detected; the result does not cover unavailable foundation pretraining data.
- The canonical QE barrier atlas now contains 14 traceable profiles and zero accepted final barriers. Current `1-6/1-7` snapshots are endpoint-dominated and therefore remain diagnostic.
- A same-structure prospective A/B batch now pairs the six frozen site-proposer
  coordinates with bounded PBC-aware transformed coordinates. All 18 audited
  direct/control/transformed inputs share calculator identity
  `03738b164d4bdccb`; transformed jobs `30861875/80/81/94`, `30861902/15`
  reached `max_seconds` during their first SCF, as did the 12 direct/control
  jobs. All have zero ionic steps, so this test has not produced a relaxation
  speed result. This tests relaxation efficiency separately from the
  proposal-versus-control stable-site yield.
- The prospective transform updater fails closed on incomplete six-pair
  coverage, missing direct-job bindings, missing repeat acceptance, different
  final basins, incomplete runtime provenance, or an unaccepted calculator.
  Rockfish Slurm job `30862166` passed all `99` tests.
- The selected-point mathematical transform is now a real prospective DFT A/B
  experiment, not a proposal. Six frozen Cr sites are paired with six bounded
  periodic local-void projections; the host atoms, cell, atom order, calculator
  identity, resources, and seed are identical. The Cr shifts span
  `0.006047-0.234812 A`. Direct and transformed jobs are running; a speedup will
  be reported only after both arms independently converge, repeat into the same
  basin, and pass the calculator gate.
- The 61-atom k-point check failed: image-4 minus image-2 changes from
  `+0.830367 eV` at Gamma to `-0.615659 eV` at `2x2x1`, a `-1.446026 eV`
  change and an ordering reversal. Gamma-only barriers are now quarantined as
  final references. The subsequent `3x3x1`/`4x4x1` contrast also failed the
  `0.02 eV` gate: `-0.807902` versus `-0.904860 eV`, a `0.096958 eV`
  difference. The completed `4x4x1`/`5x5x1` contrast is `-0.904860` versus
  `-0.894609 eV`, differing by `0.010251 eV`; this passes the `0.02 eV` gate,
  so `4x4x1` is the accepted numerical mesh. Cutoff-at-`4x4x1` also passed:
  the `50/200` and `100/400 Ry` contrasts differ by only `0.000938 eV`, so
  `50/200 Ry` is accepted. Nonspin jobs `30879759/61` and collinear-Cr jobs
  `30879762/64` completed and passed SCF. The collinear fixed-geometry
  image4-image2 contrast is `-0.692803 eV`, differing from the nonspin
  `-0.904860 eV` contrast by `0.212056 eV`; treat spin as a separate physical
  model, not as a pooled numerical variant. Explicit-SOC jobs `30879766/68`
  were cancelled on 2026-09-15 UTC per user instruction to pause SOC
  unification; their partial outputs are diagnostic only.
- For `81_neb_4`, all ten independent-image SCFs passed. The gated launcher
  submitted production QE NEB job `30879749`; it parsed all ten warm-started
  images and entered iteration 1. Warmup energies remain non-label diagnostics.

## What Is Not Finished

- 1-7 does not yet have a final converged QE NEB barrier.
- 81-atom deep-penetration paths remain candidates, not manuscript-grade references.
- Multi-seed model retraining and uncertainty bars are not finished.
- Any future checkpoint, training-file, benchmark-frame, or mechanism-group revision must rerun the leakage audit.
- Corrected MD protocol reruns are not complete.
- SHAP perturbation validation is not complete.
- Final figures and response letter still consume placeholders until upstream data gates close.
- No direct-versus-trust-region QE-NEB speedup exists yet: repeated 1-4
  endpoints, endpoint-bound MACE reruns, and the N24 calculator gate must pass
  before the matched DFT pair can be submitted.
- No direct-versus-transformed endpoint-relaxation speedup can be stated: all
  18 prospective jobs stopped inside the first SCF with zero ionic steps. The
  next gate is electronic-SCF stabilization, followed by matched bulk reruns,
  repeat validation, and same-basin assignment.
- No current Gamma-only QE barrier is manuscript-grade. Numerical convergence
  accepts `50/200 Ry` and `4x4x1`; collinear spin is now shown to materially
  change the fixed-geometry contrast, and explicit SOC is intentionally paused
  pending a later user decision.

## Working Strategy

1. Use MACE/MLFF NEB as a path preconditioner.
2. Convert the resulting images into explicit-image QE `neb.in`.
3. Submit long CPU/parallel QE NEB on Rockfish under `/scratch16/pclancy3/yi`.
4. Set QE `max_seconds` slightly below Slurm walltime so jobs stop cleanly.
5. Continue each path from its previous run directory until convergence or scientific rejection.
6. Store every run with a manifest and sync outputs back for local parsing.

## Current Machine-Readable Status

Latest monitor output: `state/qe_neb_job_status.json`. This file is generated by `scripts/migrationbench/monitor_qe_neb_jobs.py` and stores squeue/sacct status, synced local output path, parsed QE metrics, overflow flags, and action classification for each active job.

## Current Priority Order

1. Finish 1-7 in-gap QE restart chain.
2. Select one or more representative deep-penetration paths from 81-atom candidates.
3. Formalize endpoint/path sampling using continuous descriptors.
4. Rebuild tables/figures from generated data.
5. Write reviewer responses from the same upstream data nodes.

## Newly Submitted MLFF Pre-NEB Jobs

Four CPU MLFF preconditioning jobs have been submitted for historical unconverged paths.

| Job id | Path | Purpose | Initial status |
|---|---|---|---|
| `30791344` | `1-2` | deep-penetration MLFF pre-NEB | `COMPLETED 0:0` |
| `30791345` | `1-3` | deep-penetration MLFF pre-NEB | `COMPLETED 0:0` |
| `30791346` | `1-5` | deep-penetration sanity/preconditioning | `COMPLETED 0:0` |
| `30791347` | `1-6` | auxiliary path preconditioning | `COMPLETED 0:0` |

These jobs are not manuscript evidence yet. Their acceptance gate is: produce `mlff_neb_manifest.json`, `mlff_neb_images.extxyz`, a finite barrier proxy, no unphysical energy spike, and internal-image forces low enough to justify QE refinement.

## Victor NEB Parse Result

Located Victor V-ST NEB outputs are currently quarantined:

| Victor path | Parsed state | Forward activation | Max image error | Decision |
|---|---|---:|---:|---|
| `V-ST/neb` | no complete activation table | placeholder | placeholder | rerun or inspect input only |
| `V-ST/neb2` | unconverged/nonphysical | `109.440398 eV` | `380.372065 eV/A` | quarantine |
| `V-ST/neb2 copy` | unconverged/nonphysical | `627.040883 eV` | `475.185787 eV/A` | quarantine |

## Endpoint Selection Output

The original Victor/Joshitha endpoint graph (`191` sites, `12052` pairs, `24` representatives) was initially superseded because `5696` pairs crossed source files before physical compatibility was proved, and IDs were not source-namespaced. The conservative v2 graph isolated the two files until stronger evidence became available.

The raw site-search archive now proves that Victor and Joshitha are assignment namespaces for one `Sb2Te3 4x2x1` search: they share one pipeline manifest, host structure, QE template, reference cell, and one 245-input directory; all 245 inputs have composition `Sb32Te48Cr1`. The canonical v3 graph therefore uses explicit `system_id=sb2te3_4x2x1` while retaining collaborator namespaces. It contains `191` namespaced sites, `12052` physically allowed same-system pairs, `5696` cross-namespace pairs, and `24` representatives. The old graph remains noncanonical because its key schema was unsafe; v2 remains the audit record of the fail-closed interim state.

A Cr-centered periodic radial/angular proposer was grouped by connected source-void family and evaluated on all 191 sites. Against MACE screening energies it achieves `0.924 eV` grouped RMSE, Spearman `0.834`, lowest-20% recall `0.821`, pairwise ordering accuracy `0.851` for energy separations >=`0.2 eV`, and empirical 95% uncertainty coverage `0.958`. This is useful for screening but not a DFT stability model: only two candidate-site outputs exist, both predate their current inputs and are rejected as unlinked provenance; zero independent DFT minima pass the current gate.

## QE Calculator Identity Audit

An audit of `377` traceable QE inputs found `12` calculator identities and zero inputs with explicit SOC (`noncolin` and `lspinorb` both true). The current 61-atom NEB family uses `50/200 Ry`, Gamma-only, `0.0147 Ry` smearing, no spin and no SOC. The current 81-atom NEB family uses `70/280 Ry`, `2x2x2`, `0.005 Ry`, no spin and no SOC. Site-search inputs use `70/280 Ry`, `2x2x2`, `0.0147 Ry`, collinear `nspin=2`, and no SOC. These conflict with the SI's global `100/400 Ry`, `4x4x1`, SOC statement. The accepted response requires an endpoint/saddle convergence and physical-model sensitivity matrix before choosing between correcting the Methods and recomputing results. See `docs/qe_calculator_identity_audit.md`.

Cutoff-sensitivity stage 1 is running on exact `1-6/path108` images 2 and 4. Historical-profile jobs are `30850191/30850192`; `100/400 Ry` jobs are `30850193/30850195`. All are CPU jobs with 24 h walltime and `max_seconds=84600`. Client timeout caused duplicate submissions `30850194/30850197`; both were promptly cancelled, recorded in `submission_correction.json`, and excluded from the active registry. Submission now uses a Rockfish-side lock and immutable receipt to prevent recurrence.

## Hidden-Basin Audit And Endpoint Validation

The dependency-free topology audit under `data_processed/path_topology_audit/` analyzed `14` real historical/current paths, found `9` internal low-energy basin candidates, and generated `23` directed segments. All `28` inspected endpoints remain `unverified_frozen_neb_endpoints` until standalone relaxation.

| Path | Current QE forward | Image-2 energy relative to image 1 | Topology decision |
|---|---:|---:|---|
| `1-6` | `0.820400 eV` | `-1.231737 eV` | hidden basin; both current segments are endpoint-dominated |
| `1-7` | `0.557933 eV` | `-1.114792 eV` | hidden basin; both current segments are endpoint-dominated |

These forward values are diagnostic endpoint-referenced snapshots, not accepted single-saddle barriers.

Six standalone fixed-cell QE relaxations were submitted on Rockfish with `2` CPU MPI ranks, `160G`, `24:00:00`, `max_seconds=84600`, and seed `42`. A later provenance audit established that these inputs are r3 startup geometries, not the latest NEB iteration geometries. They remain valid controls, but cannot validate the latest hidden basin:

| Path | Images | Slurm jobs | State at first monitor |
|---|---|---|---|
| `1-6` r3 | `1, 2, 5` | `30843292`, `30843300`, `30843307` | running, first SCF |
| `1-7` r3 | `1, 2, 5` | `30843324`, `30843325`, `30843348` | running, first SCF |

Their manifests and an explicit correction ledger are under `data_processed/endpoint_relax_inputs/priority_1-6_1-7_s42/`. `monitor_qe_relax_jobs.py` syncs results and writes every complete ionic step with energy and atomic forces to `relax_iterations.extxyz`.

The canonical in-gap `1-4` candidate now has independent fixed-cell
relaxations submitted for historical images 1, 4, and 5 as Rockfish jobs
`30852638`, `30852853`, and `30852854`. An input audit found and fixed two
provenance defects before submission: historical `pw_1.in` is a calculator/cell
template rather than the final image-1 geometry, and the old generator omitted
explicit `nstep`, falling back to QE's 50-step default. All three submitted
inputs now draw coordinates from the same five-frame `sb2te3.xyz` with exact
`0 A` coordinate reproduction, remove the legacy variable-cell namelist, set
`nstep=200`, and set `max_seconds=84600` under a 24-hour allocation. The updated
generator passed 73 tests in real Rockfish Slurm job `30852857`.

## Paired Geometry-Transform Experiment

A deterministic periodic local-void projection was applied to Cr only, with the host fixed and displacement constrained to `0.25 A`. The objective is the sum of a displacement tether and smooth short-distance penalties. Image 1 is a symmetry-fixed point with zero displacement for both paths, so it was not resubmitted.

| Path/image | Cr shift | Minimum Cr-host distance before -> after | Direct job | Transformed job |
|---|---:|---:|---|---|
| `1-6 / 2` | `0.250 A` | `2.558 -> 2.655 A` | `30843300` | `30843835` |
| `1-6 / 5` | `0.143 A` | `2.591 -> 2.591 A` | `30843307` | `30843837` |
| `1-7 / 2` | `0.250 A` | `2.535 -> 2.666 A` | `30843325` | `30843840` |
| `1-7 / 5` | `0.132 A` | `2.581 -> 2.577 A` | `30843348` | `30843841` |

All four transformed jobs are running on Rockfish with the same QE/resource protocol as the direct controls. This first experiment is now explicitly labeled `r3 startup geometry`; it does not carry the latest NEB energy label.

A replacement exact-provenance experiment was generated from `sb2te3.path108` for `1-6` and `sb2te3.path112` for `1-7`, which are the latest complete iterations currently present in `neb.out`. The `.pathN` absolute energies agree with the same-iteration `neb.out` tables within `2e-4 eV`. Direct and transformed variants use identical templates and resources, and differ only in the Cr coordinate transform.

| Path/image | Source iteration | Cr shift | Minimum Cr-host distance before -> after | Direct job | Transformed job |
|---|---:|---:|---:|---|---|
| `1-6 / 2` | `108` | `0.250 A` | `2.471 -> 2.643 A` | `30844024 -> 30851800` | `30844035 -> 30851803` |
| `1-6 / 5` | `108` | `0.143 A` | `2.591 -> 2.591 A` | `30844025 -> 30851807` | `30844038 -> 30851806` |
| `1-7 / 2` | `112` | `0.250 A` | `2.485 -> 2.600 A` | `30844039 -> 30851802` | `30844065 -> 30851801` |
| `1-7 / 5` | `112` | `0.132 A` | `2.581 -> 2.577 A` | `30844053 -> 30851808` | `30844069 -> 30851809` |

The original eight exact-provenance jobs ended cleanly at QE's default 50 ionic steps without BFGS convergence. Their r2 children above continue from the latest printed coordinates with explicit `nstep=200`, `max_seconds=84600`, and cumulative lineage accounting. Formal speedup remains pending until both members converge to the same basin. QE path histories are exported under `data_processed/qe_neb_path_history/`, preserving 110 iterations for `1-6` and 114 for `1-7` as extxyz plus a per-image energy table and source hashes.

All future continuation and launch paths explicitly export the immutable remote
code root. Rockfish Slurm smoke `30852040` passed all 69 tests. Because the
currently active r2 jobs were submitted before per-job runtime capture was
enabled, they remain diagnostic unless a later continuation or independent
repeat records the required `runtime_provenance.json` alongside the calculator,
endpoint, and path acceptance artifacts.

The earlier reported per-atom forces for this experiment were invalid because
the parser overwrote QE's total Hellmann-Feynman force block with the separately
printed DFT-D3 contribution. The corrected parser ignores the D3 diagnostic
block. Reanalysis leaves zero formal speedups: two pairs have incomplete direct
branches; `1-6/image5` reaches different final basins; and the same-basin
`1-7/image5` pair is slower after transformation (`109` versus `92` ionic steps,
`1863` versus `1554` SCF iterations). The historical rows also lack full runtime
provenance, so these remain diagnostics rather than claims.

An independent selected-point A/B experiment is active for clearance-defect
images `1-3/image2`, `1-3/image3`, and `1-5/image2`. The transformed branch
moves only Cr by a deterministic minimum-image projection until all Cr-host
distances are at least `1.8 A`; direct and transformed calculations otherwise
use the same fixed-cell QE protocol. At `2026-09-13T15:26Z`, all six jobs were
healthy and the BFGS-step prefixes were `25/23`, `28/27`, and `29/29` for
direct/transformed. No pair is yet eligible for a speedup because both members
must converge below `0.05 eV/A` and finish in the same basin.

The new provenance-complete MACE runs for `1-4`, `1-7`, and `1-8` finished, but
all failed the strict QE-handoff gate through energy collapse or Cr path
discontinuity. They are retained as negative task-relevant model evidence, not
as DFT initializers. Rockfish test job `30852910` passed all `74` current tests.

The complete restart lineage is additionally joined with each image's `PW.out` under `data_processed/qe_neb_full_history/`. This yields `545` DFT-labeled frames for `1-6` and `565` for `1-7`. Every completed iteration has coordinates, total energy, independent PW atomic forces, potential-energy gradients, scalar NEB residuals, SCF work, reaction coordinate, stable IDs, and source hashes. Cross-check maxima are `6.81e-8 eV` for PW/path energy and `1.29e-7 eV/A` for `gradient + force`. Only the currently incomplete iterations `109` and `113` lack NEB residuals.

These records are normalized into five Hugging Face staging tables under `data_processed/hf_dataset_staging/qe_neb_history_v1/`: `1110` configurations, `1110` calculations, `1110` optimization iterations, `2` paths, and `2` derived-barrier audit records. The publication gate is closed because both paths remain unconverged and their endpoints are unverified. The fail-closed validator passes with zero duplicate primary keys, zero broken foreign keys, zero array-shape failures, zero split-group violations, zero identical structures crossing splits, and zero manuscript-gate violations. All current records are isolated in the `audit` split; no training/evaluation split has been assigned.

Nine historical MACE NEB trajectories have also been reconstructed at every
optimizer iteration and normalized into separate audit bundles. The 61-atom
bundle contains `8390` rows across four paths; the 81-atom bundle contains
`24030` rows across five paths. Both validators pass with zero relational,
array, force-semantics, split, leakage, or manuscript-gate failures. Because the
old runs did not capture runtime identity, their records preserve null runtime
fields and an explicitly assumed legacy protocol; they cannot be promoted to
training data or final barrier references without a separate release decision.

## 81-Atom QE-Ready Inputs

PBC-aware continuous geometry classification is now complete for all five historical paths. All five lie in the deep-penetration region; this confirms `neb_2`, `neb_3`, and `neb_4`, and resolves the former uncertainty for `neb_1` and `neb_5`. The canonical descriptor table is `data_processed/path_classification/81_atom_paths_v2_pbc.csv`. These are geometry classifications only, not convergence claims.

The historical per-image energy profiles are now rendered as five SVGs under `data_processed/neb_energy_profiles/`. Every curve remains labeled `historical_candidate_quarantine` because maximum image errors are `0.232-0.636 eV/A`, above the final gate.

The direct `81_neb_4` preflight showed that its first SCF, rather than NEB geometry, is the immediate bottleneck. Two standalone 24 h CPU warmup pilots are running: image 1 job `30845691` and image 2 job `30845692`, each using 4 MPI ranks, 4 k-point pools, 180 GB, and QE `max_seconds=84600`. Their temporary `degauss=0.01 Ry` and robust mixing are preconditioning settings only; production NEB restores `0.005 Ry` and reconverges.

All five historical 81-atom `2D_421_Diffusion_traj` paths now have MACE-preconditioned QE explicit-image inputs generated under `/scratch16/pclancy3/yi/revision1_migrationbench_inputs/`.

| Path | Remote QE input dir | MLFF sanity | Priority |
|---|---|---|---|
| `81_neb_1` | `mb_qe81_neb_1_mace_r1_s42` | barrier `1.916 eV`, internal fmax `0.156 eV/A` | medium-high |
| `81_neb_2` | `mb_qe81_neb_2_mace_r1_s42` | barrier almost zero, internal fmax `0.501 eV/A` | low |
| `81_neb_3` | `mb_qe81_neb_3_mace_r1_s42` | barrier zero, internal fmax `0.910 eV/A` | low |
| `81_neb_4` | `mb_qe81_neb_4_mace_r1_s42` | barrier `0.309 eV`, internal fmax `0.160 eV/A` | medium |
| `81_neb_5` | `mb_qe81_neb_5_mace_r1_s42` | barrier zero, internal fmax `0.107 eV/A` | low-medium, inspect path |

Recommended order after active `1-7`/`1-6`: submit `81_neb_1` or `81_neb_4` first. Keep `81_neb_2`, `81_neb_3`, and `81_neb_5` in the difficult/diagnostic queue unless we need broader deep-penetration coverage.

## QE-Ready Inputs Generated From New MLFF Jobs

The newly completed MLFF jobs have been converted into QE explicit-image inputs on Rockfish.

| Path | Remote QE input dir | Priority | Caveat |
|---|---|---|---|
| `1-2` | `/scratch16/pclancy3/yi/revision1_migrationbench_inputs/mb_qe1_2_mace_r1_s42` | medium | large MLFF relaxation; inspect before long QE |
| `1-3` | `/scratch16/pclancy3/yi/revision1_migrationbench_inputs/mb_qe13_mace_r1_s42` | medium | large MLFF relaxation; QE input ready |
| `1-5` | `/scratch16/pclancy3/yi/revision1_migrationbench_inputs/mb_qe1_5_mace_r1_s42` | low/difficult | MLFF internal force remains very high |
| `1-6` | `/scratch16/pclancy3/yi/revision1_migrationbench_inputs/mb_qe1_6_mace_r1_s42` | quarantined | first QE table was nonphysical; do not continue this branch |

These inputs satisfy the infrastructure requirement that every QE `neb.in` is paired with a manifest explaining where its images came from. The MACE-derived `1-6` branch is explicitly quarantined because its first QE activation table was nonphysical, indicating bad path initialization rather than ordinary slow convergence.

## 1-7 Branch Status

The first long restartable QE refinement for path `1-7` from MACE-relaxed images was cancelled and quarantined after the first activation table became nonphysical. A direct-historical preflight branch has been generated and submitted instead.

| Branch | Job/Input | Status | Evidence | Decision |
|---|---|---|---|---|
| MACE-derived QE | Slurm `30791312`, `mb_qe17_mace_r1_s42` | `CANCELLED` | activation energies overflowed; image 2/3 energies about `1700 eV` above endpoints; max all-atom MIC step in source MACE images `5.047 A` | quarantine, do not restart |
| direct historical QE preflight | Slurm `30791508`, `mb_qe17_direct_pf_s42` | clean QE stop; Slurm marked `FAILED` because the wrapper inherited QE stop status | final preflight table: forward `0.557933 eV`, max movable-image error `0.869559 eV/A`, barrier drift last three `0.005534 eV`, no overflow, last complete iteration `13` | 24 h restart continuation submitted |
| direct historical QE restart r2 | Slurm `30791829`, `mb_qe17_direct_r2_s42` | clean QE stop; still unconverged | final r2 table: forward `0.557933 eV`, max movable-image error `0.436896 eV/A`, barrier drift last three `0.000000 eV`, no overflow, last complete iteration `71` | 48 h restart continuation submitted |
| direct historical QE restart r3 | Slurm `30812497`, `mb_qe17_direct_r3_s42` | `RUNNING` | parent run dir `/scratch16/pclancy3/yi/revision1_migrationbench_runs/mb_qe17_direct_r2_s42_30791829`; QE `max_seconds=171000` for 48 h walltime | monitor to convergence or next clean restart |

Direct input directory: `/scratch16/pclancy3/yi/revision1_migrationbench_inputs/mb_qe1_7_direct_hist_s42`. Local synchronized input snapshot: `cluster/mb_qe1_7_direct_hist_s42_input/`.

## Quarantined QE Job

Path `1-6` from the first MACE-preconditioned images was submitted, but the first QE activation table was nonphysical. This branch is no longer a continuation candidate.

| Field | Value |
|---|---|
| Slurm job id | `30791362` |
| Job name | `mb_qe1_6_mace_r1_s42` |
| Run directory | `/scratch16/pclancy3/yi/revision1_migrationbench_runs/mb_qe1_6_mace_r1_s42_30791362` |
| Input directory | `/scratch16/pclancy3/yi/revision1_migrationbench_inputs/mb_qe1_6_mace_r1_s42` |
| Slurm walltime | `24:00:00` |
| QE `max_seconds` | `84600` |
| Status | `CANCELLED` after diagnosis |
| Reason | forward activation `340.825195 eV`, max image error `923.019648 eV/A`, and earlier path-length overflow; this is a bad DFT initialization, not a converging NEB segment |

## 1-6 Recovery Branches

The `1-6` path now has two recovery branches so we can distinguish bad MLFF path preparation from a real hard DFT path.

Convergence metric correction: QE tables include diagnostic errors for frozen endpoints, but those endpoints are not optimized by NEB. The canonical `max_image_error_eV_A` now means the maximum over `frozen=F` images; `max_image_error_all_eV_A` preserves the all-image diagnostic. This changes the `1-6` r2 convergence residual from the formerly reported all-image `1.076934 eV/A` to the correct movable-image `0.408386 eV/A`; the `0.820400 eV` candidate barrier is unchanged and remains unconverged. The hash-backed correction table is under `data_processed/qe_neb_metric_corrections/`.

Future QE NEB segments use `continue_qe_neb.py` and `submit_slurm_qe_neb_restart_verified.sh`. The orchestrator refuses any parent not classified `clean_max_seconds_restartable`, hashes its `neb.in`, `neb.out`, latest `.pathN`, and status snapshot, and atomically replaces the parent in `active_qe_jobs.json`. On the compute node, only the parent `out/` directory is copied; every file is SHA-256 inventoried before and after `rsync --checksum`, and any diff aborts before `neb.x`. Real Rockfish Slurm smoke job `30850065` completed in 5 seconds with exit `0:0`, identical three-file fixture inventories, `inventory_match=true`, empty diff, and both Slurm logs archived inside the scratch run directory. The old restart-chain submission examples are legacy only.

| Branch | Job/Input | Status | Acceptance Gate |
|---|---|---|---|
| geometry repair + second MACE | Slurm `30791499`, `mb_mlff_cpu_1_6_repair_idpp_s42` | `COMPLETED`; quarantined | geometry passed, but MACE still drops internal images by `17-19 eV`; do not make QE input from this branch |
| direct historical QE preflight | Slurm `30791500`, `mb_qe16_direct_pf_s42` | clean QE stop; Slurm marked `FAILED` because the wrapper inherited QE stop status | final preflight table: forward `0.820400 eV`, max movable-image error `1.576222 eV/A`, barrier drift last three `0.000000 eV`, no overflow, last complete iteration `12` |
| direct historical QE restart r2 | Slurm `30791828`, `mb_qe16_direct_r2_s42` | clean QE stop; still unconverged | final r2 table: forward `0.820400 eV`, max movable-image error `0.408386 eV/A`; all-image diagnostic `1.076934 eV/A` comes from frozen endpoint 5; barrier drift `0.000000 eV`, last complete iteration `65` |
| direct historical QE restart r3 | Slurm `30812496`, `mb_qe16_direct_r3_s42` | `RUNNING` | parent run dir `/scratch16/pclancy3/yi/revision1_migrationbench_runs/mb_qe16_direct_r2_s42_30791828`; QE `max_seconds=171000` for 48 h walltime |

Repair test files are synced locally under `cluster/repair_tests_20260911_020649/`. The best geometric repair so far is `idpp`/`smooth-all`: max migrant step `0.678 A`, total migrant path `2.713 A`, and minimum pair distance `2.305 A`.

## 81-Atom Deep-Penetration QE Probes

Two 81-atom deep-penetration representative paths are now being tested in QE. Both are preflight branches until they produce a physical first activation table.

| Path | Slurm job | Input dir | Status | Why this one | Caveat |
|---|---|---|---|---|---|
| `81_neb_1` | `30791401`, `mb_qe81n1_mace_r1_s42` | `/scratch16/pclancy3/yi/revision1_migrationbench_inputs/mb_qe81_neb_1_mace_r1_s42` | clean QE stop but no complete table | best 81-atom MLFF sanity among deep candidates, but 10-image/81-atom QE is too slow in this setup | triage before restart; image 2 SCF reached after long walltime, no activation table |
| `81_neb_4` MACE-derived | `30791512`, `mb_qe81n4_pf_s42` | `/scratch16/pclancy3/yi/revision1_migrationbench_inputs/mb_qe81_neb_4_mace_r1_s42` | `CANCELLED`; quarantined | QE showed initial path length overflow | do not restart this branch |
| `81_neb_4` direct historical | `30791514`, `mb_qe81n4_direct_pf_s42` | `/scratch16/pclancy3/yi/revision1_migrationbench_inputs/mb_qe81_neb_4_direct_hist_s42` | clean QE stop but no complete table | historical geometry very clean; image 1 SCF did not finish within 6 h segment | triage before restart; do not keep in active monitor |

The `81_neb_4` recovery now uses independent image SCFs to remove the serial first-image bottleneck. Pilot jobs `30845691` (image 1) and `30845692` (image 2) use 4 CPU MPI ranks, 4 k-point pools, 180 GB, 24 h, and QE `max_seconds=84600`. Expansion to images 3-10 is gated on both pilots reaching finite converged SCFs with charge-density/XML restart artifacts. The production assembler has passed a fail-closed 9/10-image rejection test and a complete 10/10-image smoke test; it restores `degauss=0.005 Ry`, sets `max_seconds=171000` for a 48 h Slurm job, and requires production reconvergence before any energy is eligible as a barrier label.

The tested scripts are mirrored on Rockfish at `/scratch16/pclancy3/yi/revision1_migrationbench_pipeline/scripts/migrationbench/`; both `pw.x` and `neb.x` resolve from the QE 7.3.1 CPU module there.

The post-pilot expansion is now a single config-driven action: `expand_qe_scf_warmup_batch.py --config configs/qe_scf_warmup_expansion_81_neb_4_s42.json --submit`. It passed four smoke gates: current unaccepted pilots cause zero submissions; an accepted synthetic gate prepares exactly images 3-10; repeated preparation preserves every input hash; and simulated Slurm submission registers eight unique jobs atomically with zero duplicate submissions on rerun. A separate disconnect simulation recovered all eight jobs by exact Rockfish job name without calling `sbatch` again. The warmup acceptance gate is `JOB DONE + SCF convergence + finite energy + charge-density.dat + data-file-schema.xml`; wavefunctions are inventoried but regenerated by production NEB.

Clean-timeout continuation is implemented by `continue_qe_scf_warmup.py` and `submit_slurm_qe_scf_warmup_restart.sh`. A continuation writes into a new job directory, seeds it from the immutable parent charge density/XML, regenerates wavefunctions, and records the parent status/input/manifest hashes. It refuses active, failed, or artifact-incomplete parents. A simulated submission verified one atomic registry entry, one `sbatch` call across repeated invocations, and an unchanged continuation-manifest hash. Both scripts pass syntax checks in the real Rockfish QE 7.3.1 CPU environment.

The final warmup-to-production transition is implemented by `launch_qe_neb_from_scf_warmups.py` with canonical config `configs/qe_neb_launch_81_neb_4_scfwarm_s42.json`. The real current 2/10 status was verified to fail closed without creating an output directory. A synthetic 10/10 accepted status produced a 10-source lineage manifest and a production input with `degauss=0.005 Ry`, `startingpot='file'`, 4 CPU ranks, 4 k-point pools, 180 GB, 48 h, and `max_seconds=171000`. Simulated submission produced one active-QE registry row and no duplicate submission or hash drift on rerun. The launcher and wrapper pass checks in Rockfish's QE 7.3.1 CPU environment; no real production NEB has been submitted before the gate.


## Victor Standardized QE Preflight

The first Victor V-ST path has entered the common MigrationBench QE preflight workflow.

| Path | Slurm job | Source | Status | Gate |
|---|---|---|---|---|
| `victor_01_neb_v2` | `30791542`, `mb_victor01_v2_pf_s42` | `01_neb_v2.out` endpoint-only Victor branch regenerated into 5 explicit images | clean QE stop but no complete table | image 1 SCF did not finish within 6 h segment; triage before restart |

## Auditable QE Barrier Atlas

The canonical 14-path DFT snapshot is rendered under `data_processed/neb_barrier_atlas/`: seven historical 61-atom paths, five historical 81-atom paths, and the current locally synced `1-6_qe_r3` and `1-7_qe_r3` branches. Each row retains the raw `neb.out` SHA-256, source modification time, last complete iteration, QE-reported forward/reverse activation energies, discrete image-table barriers, endpoint energy difference, peak-image location, force residuals, and convergence status.

No row currently passes the unified final-reference gate. The historical curves remain quarantined; current `1-6_qe_r3` and `1-7_qe_r3` are endpoint-dominated, running/unconverged snapshots. The atlas therefore reports candidate energy spans only and must not be consumed as a final manuscript table.

## Prospective Stable-Site DFT Batch

Audit found that all 245 mirrored site-search QE inputs used `ibrav=0` without
an explicit cell. Direct reuse is forbidden. The corrected prospective v5
preparation injects one immutable reference cell, preserves source coordinates,
uses global proposal/control matching over 12 distinct void families, and keeps
all new DFT outputs outside the frozen v1 proposer data.

Rockfish CPU smoke `30854580` completed successfully. The 12 final inputs are
byte-identical between local and Rockfish generation and share complete
calculator identity `03738b164d4bdccb`. Jobs `30854592`-`30854619` were all
running in the initial post-submission snapshot. They use 2 MPI ranks, 160 GB,
24 hours, `nstep=200`, and `max_seconds=84600`; no further health query is
allowed before `2026-09-13T19:58:00Z`. This batch estimates stable-endpoint
discovery yield, not same-structure coordinate-transform speedup.
Comparator smoke `30854874` also completed (`3 passed in 3.53 s`), validating
input/runtime hash binding and suppression of formal yield estimates for
censored or restartable branches.

The downstream acceptance software was then exercised as one complete chain
on a Rockfish CPU node. Job `30856509` completed with exit `0:0` in three
seconds: two synthetic first/repeat relax pairs passed exact input, Slurm job,
and `pw.x` runtime-provenance checks; batch acceptance reported `2/2` accepted;
complete-linkage assignment returned two distinct basins. This validates the
fail-closed machinery only. The 12 real prospective jobs remain censored until
their own first and repeat relaxations finish.

The prospective comparator previously counted a successful first relaxation as
accepted yield, which conflicted with the canonical repeat-relax endpoint gate.
That inconsistency is fixed: a first success is now `repeat_required`; formal
yield is released only after a hash-bound repeat acceptance or rejection exists
for every first success. Rockfish CPU job `30857156` validated the corrected
accepted, rejected-repeat, and missing-repeat branches (`3 passed in 3.65 s`,
exit `0:0`).

The paired direct-versus-transformed repeat planner now isolates exactly the six
proposal parents from the 12-site batch, excludes the six matched controls, and
assigns separate `direct` and `geomvoid` namespaces. Direct proposal repeats are
canonical shared computations: endpoint discovery and the transform A/B analysis
must reference the same repeat job and acceptance artifact, never submit it twice.
Rockfish CPU Slurm job `30862405` completed in 14 seconds with `102 passed`, zero
skipped, and exit `0:0`.

At the authorized `2026-09-13T20:00:11Z` prospective direct/control checkpoint,
all 12 jobs remained healthy and running, but each was still in the first SCF
after about three hours: 2-3 electronic iterations and zero ionic steps. The
canonical planner returned `waiting_for_first_relax`; no continuation or repeat
was submitted. A non-label 4-rank/4-pool single-SCF performance diagnostic for
the identical `unique_119` input was submitted as Rockfish job `30864875` with
three hours, `max_seconds=9900`, 180 GB, and unchanged calculator identity
`03738b164d4bdccb`. It does not replace or cancel any scientific job.

At the `2026-09-13T23:37Z` checkpoint all 12 direct/control jobs and all six
transformed jobs remained healthy but still had zero ionic steps; each was in
its first SCF after 3-5 electronic iterations. Probe `30864875` failed by OOM
after one SCF iteration, so `4 ranks / 4 pools / 180 GB` is rejected for this
81-atom Hamiltonian and the probe remains label-ineligible. The comparison
pipeline now records sustained force-threshold crossings and cumulative SCF
work per force evaluation; Rockfish job `30872198` passed all 107 tests.

The A/B comparator and updater now accept separate direct and transformed
repeat-acceptance batches, retaining both hash-bound sources in every comparison
and dataset row. Rockfish full-suite job `30864698` passed all 103 tests with
zero skips. Job `30864689` is archived as an infrastructure-only failure caused
by an accidental root-level duplicate test file during synchronization; the
replacement removed those duplicates and did not alter scientific data.

At the cadence-authorized `2026-09-13T23:35:04Z` checkpoint, historical `1-4`
images 1 and 5 had ended with QE BFGS markers but failed the benchmark force
gate at `0.07358` and `0.11058 eV/A`. Strict force-refinement jobs `30872019`
and `30872029` now continue their exact last force-evaluated structures with
24 h walltime and `max_seconds=84600`. Image 4 job `30852853` remains running
at 64 ionic steps and `0.39983 eV/A` maximum force.

At `2026-09-14T03:04Z`, main jobs `30812497` (`1-7`) and `30812496` (`1-6`)
remained healthy but unconverged at iterations 170 and 160, with movable-image
residuals `0.191044` and `0.257913 eV/A`. Their `0.557933/0.820400 eV`
forward activations remain diagnostic.

The force parser audit found that DFT-D3 component forces had overwritten QE's
actual total per-atom forces. After correction, no-symmetry image-1 job
`30877468` and image-5 job `30872029` are accepted first relaxations with
maximum total forces `0.040615` and `0.002138 eV/A`. Hash-bound repeat jobs
`30880454/30880456` are running. Redundant over-tight image-5 job `30879756`
was cancelled; image-1 job `30879755` ended after a BFGS trust-radius failure
despite an already tiny `0.001675 eV/A` force. Image-4 job
`30852853` ended after an SCF failure and is quarantined for review rather than
automatically continued. The monitor now synchronizes every declared parent
before cumulative lineage accounting; Rockfish test job `30877455` passed all
108 tests.

At `2026-09-14T03:07Z`, all 12 prospective direct controls and all six
transformed arms remained healthy in their first SCFs, so no formal speedup is
yet calculable. The older exact-iteration diagnostic A/B set is now terminal:
zero pairs satisfy convergence, force, same-basin, provenance, and calculator
gates, so it yields no formal speedup and will not receive further compute.

At `2026-09-14T13:38Z`, `1-7/1-6` stopped cleanly at complete iterations
`199/191` with movable residuals `1.137328/0.178825 eV/A`; neither is converged
and their activation values remain diagnostic. Verified restart segments
`30880322/30880323` are running with 48 h walltime and `max_seconds=171000`.
The Slurm monitor's accounting lookup now uses an explicit seven-day window,
preventing active or recent jobs from being falsely classified when Slurm's
default accounting window omits them.

At `2026-09-15T07:42Z`, restart jobs `30880322/30880323` remained healthy and
unconverged at complete segment iterations `46/50`, with movable-image residuals
`0.378317/0.478149 eV/A`. Production `81_neb_4` job `30879749` remained in its
first NEB iteration without a complete activation table. Material-transfer jobs
`30878879/30878880` reached complete iterations `6/4` with residuals
`0.911180/0.958397 eV/A`; they use a different calculator identity and stay
outside the production barrier pool.

At `2026-09-15T07:46Z`, all 18 prospective direct/control/transformed jobs were
terminal after clean `max_seconds` stops during their first SCF. They completed
`15-20` electronic iterations, zero ionic steps, and no force evaluation, so
neither stable-site yield nor transformed-versus-direct relaxation speed is
defined. The earlier 4-rank/4-pool probe OOMed; performance-only CPU job
`30920087` now tests the identical `unique_119` Hamiltonian using 2 ranks,
2 k-point pools, 180 GB, 24 h, and `max_seconds=84600`. Its output is never label
eligible. Parser-regression job `30880482` passed all `109` tests on Rockfish.
