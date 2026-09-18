# Branch Manifest And Logging Standard

Every MigrationBench NEB branch must leave two records:

- a machine-readable event in `/Users/alina/Project/26MIgrationBench/state/workflow_events.jsonl`
- a branch manifest next to the generated input or output on Rockfish and in the synced local snapshot

The schema is `/Users/alina/Project/26MIgrationBench/configs/branch_manifest_schema.yaml`.

## What Counts As A Branch

A branch is any distinct proposal for a migration path:

- original historical images
- repaired images from physics heuristics
- MACE/MLFF-relaxed NEB images
- QE preflight
- QE restart continuation
- accepted final QE reference

Two branches are different if they differ in source images, repair method, model, seed, QE input, walltime/max_seconds, restart parent, or decision state.

## Required Lineage

Each branch must state:

- `path_id`: the scientific migration path, for example `1-7`, `81_neb_4`, or `victor_01_neb_v2`
- `branch_id`: the exact computational branch, including method and seed
- `parent_branch_id`: the branch it came from
- `source_images`: the image file that generated this branch
- `source_run_dir`: the Rockfish directory, if any
- checksums for local synchronized input files whenever available
- `runtime_provenance.json` and its SHA-256 for every new computational branch;
  historical branches must explicitly record why this artifact is unavailable

For QE restart segments, the manifest must also record:

- parent QE run directory
- restart mode, normally `restart`
- copied restart artifacts
- Slurm walltime
- QE `max_seconds`, which must be slightly below walltime

## Repair Logging

When intermediate images are adjusted by physics heuristics, record:

- repair mode: `smooth-host`, `smooth-all`, `idpp`, or another named mode
- thresholds: minimum pair-distance floor and any displacement limits
- whether endpoints were preserved
- before/after geometry metrics
- why the repair was attempted
- whether the repaired branch is allowed to go to MACE or QE

The repair itself is not evidence for the manuscript. It is only an auditable proposal-generation step.

## MACE/MLFF Logging

Each MACE/MLFF NEB branch must record:

- model path and model role: foundation, fine-tuned, scratch, or diagnostic
- seed
- optimizer, force threshold, maximum steps, number of images
- initial and final energy profile
- per-image relaxation energy delta
- maximum internal-image final force
- approximate MLFF barrier

Large energy lowering is not automatically good. It is acceptable only when it is a small path refinement that preserves the same migration channel. If internal images drop by many eV relative to the starting path, the branch is treated as basin collapse unless QE preflight proves otherwise.

## QE Logging

Each QE `neb.in` must have a companion manifest explaining:

- whether the images came from historical DFT, repaired geometry, MACE, or Victor input/output
- the engine template used for `pw.x` sections
- number of images
- Slurm job name and job id after submission
- Slurm walltime and QE `max_seconds`
- restart mode
- parsed path length, inter-image distance, activation table, convergence state, and overflow flags

If QE stops because `max_seconds` was reached and restart files exist, the next branch must use the previous run directory as parent. If Slurm kills the job before QE exits cleanly, the branch is `timed_out_dirty` and must be triaged before resubmission.

Every new MACE and QE wrapper calls `capture_runtime_provenance.py` before the
scientific executable. It records imported and distribution package versions,
module-file hashes, Slurm/module/Conda identity, scientific binary hashes, and
declared input hashes. A final-reference dataset row must bind this record by
URI and SHA-256.

## Decision Rule

Every branch ends in exactly one decision:

- `accept_for_next_stage`: can be used to generate the next branch
- `continue_with_caution`: can proceed, but the caveat must be carried downstream
- `restart_from_parent`: continue the same QE chain
- `ready_not_submitted`: input is ready but intentionally queued
- `quarantine_do_not_restart`: keep as diagnostic evidence but do not continue
- `accepted_final_reference`: allowed to feed final figures, tables, and reviewer response text

Only `accepted_final_reference` values may become final manuscript numbers. Everything else must remain labeled as preflight, diagnostic, quarantined, or pending.

## Current Example: 1-6 Repaired MACE Branch

The `1-6_mace_idpp_repair` branch passed geometry checks, but its follow-up MACE branch dropped internal images by `17-19 eV`. That is now logged as `quarantine_for_qe_initialization`, because it likely moved the images into another basin instead of refining the same migration path. :codex-annotation{index="1"}
