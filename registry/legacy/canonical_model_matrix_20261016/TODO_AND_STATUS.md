# Canonical model matrix: TODO and status

## Completed in this planning/submission pass

- Audited the 12-model historical registry and separated model labels from
  actual data/foundation/architecture lineage.
- Confirmed that the Multi-T grouped three-seed training has compiled models
  and completed test logs, but still needs formal acceptance and canonical
  grouped IDs.
- Confirmed that the 600 K grouped OMAT-FT three-seed models exist and have
  per-seed test metrics, but seed123 must be registered separately from the
  historical random-split model.
- Verified that the 600 K source universe is contained in the Multi-T source
  universe. Direct concatenation is prohibited because it duplicates source
  trajectories and can place one trajectory in multiple splits.
- Frozen `from_scratch_all_data_v1` as the deduplicated 2-layer Multi-T source
  universe, with trajectory/group-level splits and explicit hashes.
- Added the strict-comparison practice and pre-submission gates.
- Submitted canonical scratch Lmax=2 seeds 123/234/345 as Rockfish job
  `32074582` (`32074582_[0-2%3]`) after the environment bootstrap repair.
  It remains pending; no result is scientifically accepted yet.
- Submitted additive scratch-600K grouped seeds 123/234/345 as Rockfish job
  `32075474` (`32075474_[0-2%3]`) using the independently audited grouped
  release. It remains pending; the provisional upstream-manifest boundary is
  retained.
- Repaired and resubmitted the six 5% FT jobs as `32075144_[0-5%6]` after
  preserving the failed `SINGULARITY_BIND` bootstrap evidence from job
  `32069341`.
- Submitted the registered/completed-model gate as job `32075596`; it checks
  exact model hashes, MACE loadability, finite parameters, and a finite
  Cr/Sb/Te periodic probe before any static screen use.
- Prepared but did not submit the canonical 639-candidate screen. Its source
  code, candidate-manifest hash, 64-chunk plan, and held status are recorded
  in `SCREEN_SOURCE_MANIFEST.json`.
- Validated the DFT-ready local reconstruction builder with one historical
  candidate as tooling evidence only. The output is explicitly
  `TOOLING_VALIDATION_ONLY` and no QE job was submitted.

## Required before publication use

1. Monitor jobs `32074582`, `32075474`, `32075144`, and `32075596` with
   `squeue`, `sacct`, logs, checkpoints, finite status, and the acceptance
   matrix.
2. For each seed, verify terminal training completion, compiled-model existence,
   loadability, finite parameters, Cr/Sb/Te element support, and model hash.
3. Record the resolved MACE/PyTorch/ASE environment and exact job/array IDs.
4. Build a canonical registry append with IDs
   `from_scratch_all_data_v1_seed123/234/345`; do not overwrite historical IDs.
5. Add formal grouped IDs and model hashes for
   `ft_multit_grouped_A2_seed123/234/345` and
   `ft_600k_grouped_A2_seed123/234/345` after their independent load/audit
   checks.
6. Audit all source-group overlaps across every canonical split and ensure no
   trajectory crosses train/valid/test.
7. Re-run the 639-candidate static MLIP screen on the accepted canonical
   matrix. Require 639 x number-of-model rows, one finite result per pair, and
   unchanged candidate/source-stage IDs.
8. Recompute stage medians, q95/max-force tails, paired force disagreement,
   and strict within-family seed dispersion. Exclude all historical
   heterogeneous models from those variance estimates.
9. Update the model registry, campaign manifest, claim-evidence matrix, and
   figure manifest with hashes and exact rebuild commands.
10. Re-render the four-panel paper figure and retain the historical comparison
    as a separately labelled diagnostic; panel b must not use vertical grey
    separators.
11. Only after the canonical MLIP screen is accepted, decide whether a reduced
    approximately 120-atom DFT fragment is worth submitting. Keep it labelled
    as a reconstruction/local-fragment test, not exact 2050-atom parity.

## Explicitly not accepted

- Job `32055099_[0-1]`: it points to the unresolved legacy `data/` release and
  remains provisional even if it later completes.
- The old three-column “From Scratch” group as a strict seed experiment.
- Any direct union of the prepared 600 K and Multi-T extxyz files.
- Any foundation checkpoint described as a random-seed replica.
- Any DFT correctness or causal claim based only on MLIP disagreement.
