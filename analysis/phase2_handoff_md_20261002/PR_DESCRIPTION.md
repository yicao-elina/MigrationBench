# Phase 2: short-range handoff MD stability analysis

## Summary

- Stacks on `codex/phase1-static-force-comparison` (Phase 1 PR #6).
- Adds an isolated, additive handoff-MD campaign using exact source snapshots at steps `461800` and `462000`; neighboring steps `460000` and `462200` are retained for provenance.
- Evaluates Foundation, Scratch, FT-600K, and FT-MultiT for 2 ps at 600 K with 1 fs integration and 10-step metric saving.
- Preserves initial positions, momenta, cell, PBC, species, model hashes, scheduler logs, retry evidence, manifests, and Plot Atlas figure sources.

## Results

- Canonical retry2 coverage: 8/8 runs completed and produced full 202-row metric tables.
- 4 runs were `runaway_or_nonfinite` (Foundation/Scratch at both snapshots).
- FT-600K and FT-MultiT were stable at step 461800 and bounded but structurally shifted at step 462000.
- The result is a short-range transfer/stability gate, not a diffusivity estimate or DFT validation.

## Validation and provenance

- Exact source trajectory SHA256 and per-model hashes are recorded in `phase2_manifest.json`.
- Initial and retry failures remain retained under `remote_artifacts/logs/`; no existing continuation task was cancelled or modified.
- Updated time-series figure uses the registered `nature_steered_dynamics` palette and `timecourse` primitive with the canonical Atlas-compatible renderer provenance recorded in the SVG.
- `py_compile`, manifest parsing, and `git diff --check` pass.

## Stack

Base branch: `codex/phase1-static-force-comparison`
Head branch: `codex/phase2-handoff-md`

This PR is intended to merge after Phase 1 PR #6. Phase 3 can be stacked on this branch after review.
