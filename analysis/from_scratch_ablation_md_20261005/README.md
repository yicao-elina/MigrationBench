# From-scratch ablation MD campaign (2026-10-05)

This directory contains the reproducible runner and audit logic for the
additive Rockfish campaign
`/scratch16/pclancy3/yi/mace_from_scratch_ablation_MD_20261005`.

The formal 20 ps screening matrix includes only model artifacts whose current
training log, configuration, compiled artifact, and data/ provenance agree.
`ablation_correlation_3` seed123 is retained in the audit as
`PROVENANCE_CONFLICT` because its directory contains both a legacy
data_600K/foundation run and a later data/ scratch run under overlapping names;
it is not submitted as a formal scratch model.

Every run records the model/input SHA256, training seed, independent MD seed,
2050-atom/PBC/species/cell checks, energy/temperature/force/geometry metrics,
trajectory, checkpoint, and an atomic status manifest. The 20 ps outputs are
screening evidence only. A failure model is unstable and receives no Dxy.
The hop count is explicitly a candidate displacement count until site mapping
is validated; HFACF is unavailable unless heat flux is recorded.

`campaign_manifest.json` and `PROTOCOL_AND_COMPARISON.md` define the accepted
protocol and the boundary between same-protocol evidence and legacy/reference
results. `unified_postprocess_md.py` reuses the repository stress-based
heat-flux convention when stress is recorded or recomputed from the exact
model, and emits explicit unavailable/gate states otherwise.

`comparison_inventory.csv` is a merge-ready status table. It labels the old
continuation metrics as `REFERENCE_PROTOCOL_DIFFERENT`; it does not treat them
as replicas of the new fresh-start ablation runs.

`plot_unified_comparison.py` is the rerunnable Plot Atlas entry point for the
temperature/pressure, MSD, VACF, HFACF and running-κ comparison figure. The
current `figures_unified_atlas_20ps/` output is explicitly screening-only. The
selected 1 ns branches terminated at `instability_threshold`; their additive
postprocess products, manifests, model registry, claim/evidence matrix, gap
report and blocked diagnostic Plot Atlas figure are in
`terminal_failure_bundle_20261006/` and
`figures_terminal_failure_atlas_20261006/`. No production transport figure is
generated from these incomplete trajectories.

`analyze_production_uncertainty.py` is the post-completion audit entry point:
it removes a declared 100 ps equilibration window, calculates eight-block
thermodynamic and Cr-MSD/Dxy statistics, and reports family-level replica
mean/SD/SEM with explicit `n=1_no_replica_uncertainty` states.
When post-processing outputs are relocated away from the campaign root, pass
the original `runs/` directory with `--runs-root`; recorded absolute paths are
used first and the fallback is layout-independent.

For the present terminal-failure campaign, the same audit is retained under
`uncertainty_blocked_20261006/`; it reports zero production-accepted runs and
therefore no block or replica estimate.

If the 3-day scheduler segment ends before 1,000,000 steps, use
`continue_ablation_md.py` through `submit_1ns_continuation.slurm`. It resumes
from the validated checkpoint into a new `rNN` directory, preserves absolute
steps and model/checkpoint hashes, reconstructs the initial Langevin RNG stream
for the first continuation, and serializes RNG state for later segments.
Scientific failures do not auto-continue; wall-time checkpoints can chain only
after the segment passes the finite-state and identity gates. The current
terminal failures are scientific instability, not wall-time interruptions, so
no continuation segment was launched.
