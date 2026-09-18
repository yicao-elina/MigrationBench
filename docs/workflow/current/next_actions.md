# Next Actions

## Data Quality Quarantine

- Do not use `3.fine-tuning/2-layer/data/neb_extxyz_outputs` as a DFT-label source. Its historical extractor has a factor-of-two energy-unit error, labels path gradients as forces with the wrong sign and scale, and promotes `.path0` placeholders to labels.
- Use `configs/historical_neb_extxyz_quarantine.json` as the denylist and `parse_qe_neb_full_history.py` as the replacement. Regenerate any benchmark that consumed the affected `energy` or `forces` fields.
- Cross-supercell Cr-centered structure leakage audit Slurm job `30845544` passed for all `1110` current `1-6/1-7` QE evaluation frames: no exact global/local fingerprints, no local RMSD <= `0.05 A`, and no SOAP cosine distance <= `0.0001` were found in the declared fine-tuning train/valid/test files. Attempts `30844881` and `30845220` remain invalid failures, not passes. This result does not establish anything about an unknown foundation-model pretraining corpus.

## Active Monitoring

- Use `scripts/migrationbench/monitor_qe_neb_jobs.py --jobs-file configs/active_qe_jobs.json` to sync compact Rockfish outputs, parse `neb.out`, and write `state/qe_neb_job_status.json`.
- After a new complete iteration or restart segment is synced, run `scripts/migrationbench/export_qe_neb_full_histories.py --config configs/qe_neb_full_history_exports.json` to refresh the lossless per-iteration extxyz/CSV assets. Update the lineage config whenever r4 or later is created.
- Rebuild `data_processed/hf_dataset_staging/qe_neb_history_v1/` with `build_neb_history_dataset.py`, then require `validate_migrationbench_dataset.py` to pass before any Hugging Face upload. Current unconverged records stay in the isolated `audit` split and are not eligible for training or manuscript reference values.
- Monitor verified restart jobs `30880322` (`1-7` r4) and `30880323` (`1-6` r4) every 3 hours. At `2026-09-15T07:42Z` they were healthy at complete segment iterations `46/50`, with movable residuals `0.378317/0.478149 eV/A`; neither was converged. Their current `0.557933/0.820400 eV` activations remain diagnostic.
- As of 2026-09-11T13:15Z, `30791508` and `30791500` stopped cleanly with physical but unconverged activation tables. They have been continued as 24 h restart jobs `30791829` and `30791828` with QE `max_seconds=84600`.
- As of 2026-09-12T13:28Z, `30791829` and `30791828` stopped cleanly with physical but unconverged activation tables. They have been continued as 48 h restart jobs `30812497` and `30812496` with QE `max_seconds=171000`.
- `30791514` (`81_neb_4` direct) and `30791542` (`Victor 01_neb_v2`) stopped cleanly but produced no complete activation table; keep them in triage, not the active monitor list.
- `30791401` (`81_neb_1`) also stopped cleanly after about 23.5 h with no complete activation table; keep it in triage, not the active monitor list.
- If a job ends by `max_seconds`, use its run directory as the parent for the next restart segment.
- Submit that next segment only through `continue_qe_neb.py`, and only when the monitor classification is `clean_max_seconds_restartable`. Use explicit resources: current 61-atom `1-6/1-7` chains use 2 ranks, 160 GB and 1 k-point pool; the 81-atom warm-start chain uses the resources pinned in its production manifest. The verified wrapper hashes every file under the parent and copied `out/` directories and aborts before QE if they differ.
- If a job fails, inspect `neb.out`, Slurm stderr, memory, and path geometry before resubmitting.
- Also monitor the six endpoint/basin validation jobs through `scripts/migrationbench/monitor_qe_relax_jobs.py` and `state/qe_endpoint_relax_job_status.json`. Do not poll more often than the three-hour heartbeat unless a material failure is reported.
- Monitor `1-4` historical-path endpoint/basin jobs `30852638` (image 1),
  `30852853` (image 4), and `30852854` (image 5) from their path-specific
  registry. Do not first poll before `2026-09-13T18:07Z`. All three derive
  coordinates from the same multi-frame `sb2te3.xyz`; `pw_1.in` supplies only
  calculator and cell metadata. Continue only clean stops, and require the
  independent repeat-relax gate before declaring a stable endpoint.
- The corrected parser excludes separately printed DFT-D3 component forces.
  Image-1 job `30877468` and image-5 job `30872029` are accepted first
  relaxations at `0.040615/0.002138 eV/A`. Monitor their hash-bound repeat jobs
  `30880454/30880456`. Redundant job `30879756` was cancelled and terminal
  over-tight job `30879755` is diagnostic only. Image-4 job `30852853` ended in SCF nonconvergence;
  keep it quarantined for explicit review and do not auto-continue it. Only a
  force-gated result may proceed to the independent repeat-relax stage.
- The original image-5 jobs `30843307` and `30843348` ended by maximum ionic steps, not BFGS convergence. Continue their exact latest printed geometries through `30851387` and `30851388`; preserve separate hashes for the latest printed and last force-evaluated structures.
- The original image-1 jobs `30843292` and `30843324` reported QE BFGS convergence but failed the benchmark maximum-atom-force gate (`0.07493` and `0.07312 eV/A`). Strict refinements `30851617` and `30851619` use `etot_conv_thr=1e-6 Ry` and `forc_conv_thr=5e-4 Ry/Bohr`.
- Do not promote the running `1-6` or `1-7` forward activation values while their endpoints are frozen/unverified and the path topology contains a hidden image-2 basin.
- All ten `81_neb_4` standalone SCFs are accepted. Production QE NEB job `30879749` is running from their hash-bound charge densities with `degauss=0.005 Ry`, 4 ranks/4 pools, 180G, 48 h, and `max_seconds=171000`. Monitor it every three hours through the active QE registry; warmup energies are never barrier labels.
- For a row classified `clean_timeout_needs_restart`, use `continue_qe_scf_warmup.py` with the timed-out parent job ID and a new attempt number. It creates a new run directory, copies only charge/XML, records parent hashes, uses `startingpot='file'`, regenerates wavefunctions, and refuses a still-active or scientifically failed parent.
- After all ten `81_neb_4` images pass the warmup gate, run `python3 scripts/migrationbench/launch_qe_neb_from_scf_warmups.py --config configs/qe_neb_launch_81_neb_4_scfwarm_s42.json --submit`. Do not assemble or submit it manually. The launcher pins the latest accepted attempt for every image, restores `degauss=0.005 Ry`, regenerates wavefunctions, sets QE `max_seconds=171000` below the 48 h walltime, and registers the job atomically.

## Pilot Path Before Scaling

1. Finish standalone relaxations of images `1`, `2`, and `5` for current `1-6` and `1-7`.
2. For every first-relax row accepted by the monitor, generate and submit a
   same-calculator repeat relaxation with `prepare_qe_repeat_relaxations.py`.
   Reject structures that fail BFGS, force, repeat-displacement, energy,
   geometry, provenance, or calculator gates; compose accepted endpoint pairs
   with `build_endpoint_pair_acceptance.py`, then assign basin IDs after
   deduplicating accepted structures under the configured basin metric.
3. Select one accepted endpoint pair with a physically meaningful displacement and an expected interior saddle.
4. Generate historical-subpath, warped-path, IDPP, and linear baselines with complete manifests.
5. Run MACE NEB for each initializer and cluster the relaxed mechanisms.
6. Escalate the best non-collapsed representative to restartable QE NEB and continue until the `0.03/0.02 eV/A` path-force and `0.02 eV` drift gates pass.
7. Export every MACE/QE image and optimizer iteration before adding more candidate mechanisms.

## Geometry-Transform A/B Test

- Direct controls: `30843300`, `30843307`, `30843325`, `30843348`.
- Constrained local-void variants: `30843835`, `30843837`, `30843840`, `30843841`.
- Refresh direct and transformed status with `monitor_qe_relax_jobs.py`, then regenerate `data_processed/endpoint_relax_comparison/geomvoid025_vs_direct/` with `compare_endpoint_relaxation_speed.py`.
- Primary efficiency metrics are ionic steps and accumulated SCF iterations. Walltime is secondary because the paired jobs run on different nodes.
- If final basins differ, record a basin-switch outcome and do not calculate speedup.
- Treat the original jobs above as r3-startup-geometry controls only; see `provenance_correction.json`.
- Canonical exact-iteration pairs are `30844024/30844035`, `30844025/30844038`, `30844039/30844065`, and `30844053/30844069`.
- Their source iterations are `1-6:108` and `1-7:112`, with exact `.pathN` hashes in each manifest. Monitor them every three hours, not continuously.
- Accept an acceleration claim only if both variants finish BFGS, have final max force <= `0.05 eV/A`, and pass the same-basin geometry gate. Compare ionic steps and accumulated SCF iterations first; walltime is secondary.
- Current prefix evidence is mixed: both image-2 transformed branches have fewer saved ionic steps and similar/lower force, while both image-5 transformed branches show force excursions. All transformed starts are `0.166-0.184 eV` above their direct controls. Keep the conclusion pending and use `docs/endpoint_site_transform_ab_experiment.md` as the canonical experiment definition.
- The three-hour heartbeat now regenerates pair tables and force curves whenever endpoint status or ionic-step counts change.

## QE Protocol Identity Gate

- Do not pool energies across the 61-atom `50/200 Ry Gamma`, 81-atom `70/280 Ry 2x2x2`, site-search spin-polarized, and Victor `100/400 Ry 1x1x1` identities.
- The fixed-geometry cutoff gate passed: image-4 minus image-2 is `0.830367 eV` at `50/200 Ry` and `0.829871 eV` at `100/400 Ry`; the `0.000497 eV` difference is below the predeclared `0.02 eV` tolerance.
- The completed Gamma-versus-`2x2x1` stage failed: the image-4 minus image-2 contrast changes from `+0.830367 eV` to `-0.615659 eV`, a `-1.446026 eV` change with an ordering reversal. Quarantine Gamma-only barriers as final references.
- The `4x4x1` versus `5x5x1` contrast differs by `0.010251 eV` and passes. Cutoff-at-`4x4x1` differs by only `0.000938 eV`, accepting `50/200 Ry` and `4x4x1`. Nonspin `30879759/61` and collinear-Cr `30879762/64` completed; the collinear fixed-geometry contrast is `-0.692803 eV`, `0.212056 eV` away from the nonspin reference, so spin remains a separate physical-model branch. Explicit-SOC jobs `30879766/68` were cancelled per user instruction and must not be continued or resubmitted until the user re-enables SOC work.
- Prepare endpoint/saddle SCF sensitivity pairs before launching the 12-site prospective DFT validation batch.
- Test cutoff and k-point convergence on barrier differences with a proposed `0.02 eV` tolerance; treat collinear spin as a separate physical-model variant and defer explicit SOC.
- Do not alter settings inside an active NEB restart chain.
- Revise the SI's global `100/400 Ry + 4x4x1 + SOC` statement only after the sensitivity matrix determines the defensible protocol and rerun scope.

## Ready But Not Submitted

The machine-readable path-queue audit now has `12/12` paths assigned: nine
terminal local MACE assets and three active corrected-PBC jobs, with zero
uncovered rows. This is coverage of the preprocessing workflow, not scientific
acceptance.

- MACE jobs `30852128` (`1-4`), `30852143` (`1-7`), and `30852144` (`1-8`)
  are terminal and fully synchronized. Their runtime, checkpoint, input,
  candidate-manifest, output-image, and complete-history hashes pass.
- No branch passes the QE handoff gate: `1-4` has a `5.178 eV` internal energy
  collapse; `1-7` has a `2.643 A` maximum Cr step; `1-8` has a `3.521 A`
  maximum Cr step and a `9.229 eV` energy collapse. Archive them as model/path
  diagnostics and do not generate QE NEB inputs from them.

- `1-2`: QE input ready, but MLFF relaxation is large; inspect before long QE.
- `1-3`: QE input ready, large MLFF relaxation; useful as model-disagreement evidence and possible DFT refinement.
- `1-5`: QE input ready but MLFF internal force is very high; keep in difficult queue.
- `1-6`: original MACE-derived QE branch is quarantined; continue only through either repaired-image MACE or direct historical QE preflight.
- `1-7`: original MACE-derived QE branch is quarantined; continue only through direct historical QE preflight `30791508`.

## 81-Atom Classification

- PBC-aware continuous classification places all five `2D_421_Diffusion_traj` paths in the deep-penetration region. Penetration scores for `neb_1...5` are `0.691`, `0.736`, `0.718`, `0.740`, and `0.741`; gap scores are `0.350`, `0.210`, `0.300`, `0.216`, and `0.235`.
- Treat these as one broad family with multiple continuous submechanisms, not five independent categorical truths. Use path length, endpoint displacement, coordination trajectory, layer-plane clearance, and energy profile for later clustering.
- Per-image historical energy profiles are available under `data_processed/neb_energy_profiles/*81_neb*.svg`. Their barriers remain quarantined until endpoint and path-force gates pass.

## Path Repair Gate

- Use `repair_neb_images.py` only as an auditable preconditioner, not as final evidence.
- When the offending pair includes the migrant and both endpoints pass, prefer PBC-aware `migrant-only` repulsion so the host and endpoints remain immutable. Use IDPP or smoothing only as separately named alternatives, never as silent repair.
- `1-2` cannot be repaired by moving intermediate images because its final endpoint has a `1.425 A` Cr-Sb contact; redefine or replace that endpoint.
- Jobs `30850728`-`30850731` are quarantined because their inputs lacked a periodic cell; never use their apparent convergence or barriers.
- Corrected periodic MACE-MP-0 jobs `30850830`-`30850833` are terminal
  diagnostics. All four force-converged but collapsed to zero barrier proxies
  and failed geometry, energy-collapse, or provenance gates; do not use them as
  QE initializers or speedup evidence.
- Monitor paired QE point-relax jobs `30850842`-`30850847`. Compare ionic steps and cumulative SCF iterations only after both members converge below `0.05 eV/A` and pass the same-basin gate.
- Continue only `clean_max_seconds_restartable` or clean maximum-ionic-step QE relax rows through `continue_qe_relax.py`; it starts from the latest printed geometry, records that hash separately from the last force-evaluated geometry, explicitly writes `nstep=200` by default, and accumulates work over the full lineage. The current exact-pair continuations are `30851800/03/07/06` for 1-6 and `30851802/01/08/09` for 1-7. Do not poll them before the registered three-hour window and do not auto-restart scheduler-truncated, SCF-failed, or incomplete outputs. A force-refinement branch is accepted only when its latest geometry has aligned evaluated forces and maximum atom force `<=0.05 eV/A`.
- Keep `1-5` diagnostic: its `4.200 A` maximum Cr step remains a path-continuity failure even after the close-contact projection. Reconstruct or densify it in a separate named branch before any QE NEB.
- Monitor clearance-graph MACE job `30850997`. Its 13-image initializer passes `1.806 A` clearance and `1.830 A` maximum-step gates, but similarity to historical `1-5` is only `0.535`; classify it as a separate submechanism unless final-path clustering proves equivalence.
- Reject repairs that create minimum distances near the hard threshold or produce large MLFF energy collapses during the second MACE pass.
- Monitor primary trust-region MACE diagnostic `30853021` no more frequently
  than the three-hour cadence. It uses seed 42, `k_Cr=4.0 eV/A^2`,
  `k_host=0.5 eV/A^2`, and the same `1-4` input/model/seed as unrestrained job
  `30852128`. Job `30852994`, seed 52, is an implementation replicate. Require
  complete dual-potential history and classify either only as a DFT-preconditioner
  candidate; never report tether energy as a barrier.
- The current historical 1-4 trust run may only return a curvature-donor
  diagnostic because its source endpoints are unverified. Never hand its final
  images directly to production QE. After images 1, 4, and 5 pass independent
  repeat relaxations, reconstruct separate endpoint-bound `1->4` and `4->5`
  candidates, then rerun MACE on each new candidate.
- Use `materialize_endpoint_remap_candidates.py` with exactly two
  `--acceptance SEGMENT_ID=endpoint_pair_acceptance.json` arguments. Do not
  manually edit the null placeholders or construct paths from unverified
  structures; the materializer verifies the repeat-final structure hashes and
  shared endpoint calculator before producing the MACE queue.
- Full Rockfish CPU smoke `30853484` validates that both planned segment
  acceptances can drive real candidate materialization (`1 passed in 4.45 s`).
  It uses synthetic accepted fixtures and therefore closes the software gate,
  not the physical endpoint gate.
- Only if a newly rerun, endpoint-bound comparator returns
  `trust_region_candidate_ready_for_dft_ab_design` and N24 passes, create the matched pair with
  `prepare_qe_neb_handoff_pair.py --mlff-initialization-role
  trust_region_mlff_preconditioned --mlff-preconditioner-acceptance ...`.
  Require schema-1.1 validation before Slurm submission. After both QE lineages
  converge, use `compare_qe_neb_initialization_ab.py`; never calculate speedup
  from running prefixes or paths that finish in different mechanisms.

## Victor

- Located Victor V-ST `neb.out` files are quarantined.
- Next useful action is to inspect their corresponding `neb.in`/structures and regenerate clean MLFF-preconditioned QE inputs under `/scratch16/pclancy3/yi`, not use their current barriers.

## Endpoint Graph

- `endpoint_selection_v3` is canonical: raw pipeline metadata proves Victor and Joshitha are namespaces within the same `Sb2Te3 4x2x1` search. It contains 191 namespaced sites, 12052 same-system pairs, and 24 representatives; all 5696 cross-namespace pairs now have an explicit compatibility basis.
- Use `data_processed/site_stability_proposer/v1_mace_screening/` only as a MACE-energy screening ranking. Its grouped geometry metrics pass, but the DFT-label gate fails (`0/20` accepted independent minima), so it cannot yet assert endpoint stability or automatically submit QE.
- The older 12052-pair output is superseded because it did not guarantee source-key uniqueness or system isolation.
- Before the representative queue feeds MLFF, each selected site must be reconstructed as a full periodic structure and pass standalone endpoint relaxation.
- Prospective site-proposer DFT jobs `30854592`-`30854619` and transformed jobs
  `30861875`-`30861915` are terminal after reaching `max_seconds` during their
  first SCF. Each has zero ionic steps and only `15-20` completed electronic
  iterations. Do not calculate yield or speedup, and do not bulk-restart them.
  First evaluate performance-only job `30920087` (2 ranks/2 pools, 180 GB,
  24 h) after `2026-09-15T10:48Z`; only a successful SCF may authorize a
  matched continuation design that preserves all cumulative work.
  These outputs are a holdout for v1 and must not be added to its training or
  cross-validation data. Use them first for paired local-minimum yield and basin
  deduplication, not as a same-structure speedup experiment.
- CPU scaling diagnostic `30864875` failed by OOM after one SCF iteration with
  4 ranks/4 pools and 180 GB. Reject that resource profile and never use its
  energy as label, endpoint, or training evidence. The 12 scientific jobs remain
  healthy and unchanged.

- `81_neb_4`: MACE-derived preflight `30791512` quarantined for QE path-length overflow; direct-historical preflight `30791514` submitted.

## Victor actionable queue

- Use `data_processed/victor_inventory/victor_actionable_neb_queue.csv` before submitting Victor jobs.
- Prioritize `standardize_and_preflight` rows with path-specific images and more than endpoint-only images.
- Treat 2-image Victor NEB-like runs as endpoint/preflight evidence only; regenerate intermediate images before barrier benchmarking.

- `victor_01_neb_v2`: standardized 2-endpoint Victor path into 5-image QE preflight `30791542`; clean stop but no complete table, needs SCF/restart triage.
- `victor_23_neb`: standardized input prepared and ready, not submitted.
