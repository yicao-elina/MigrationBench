# From-scratch ablation MD gap report (terminal 1 ns campaign)

Status: BLOCKED for formal transport comparison; all three selected 1 ns branches terminated by the scientific instability threshold.

## Verified gates

- All runs used the same 2050-atom periodic input, the same MD seed, 1 fs timestep, 600 K target, 0.01 fs^-1 friction, and the recorded model/input/species hashes.
- All three runs have finite postprocessed energy/force/temperature/pressure fields and reproducible trajectory hashes.
- None reached the requested 1,000,000 absolute steps; therefore no run passes the complete-step gate.
- Each branch is `FAILURE_MODE` and `EXCLUDE_FROM_PUBLICATION` for formal Dxy, hop-rate, replica uncertainty, or transport claims.

## Scientific consequences

- The longer seed-123 lmax-2 branch provides a diagnostic 200–500 ps Dxy fit and candidate hop count only; its later 500–900 ps fit overlaps a time window beyond the recorded terminal time and is not accepted.
- Stress-based heat-flux and running-kappa products are retained as repository diagnostics. They are not converged HFACF/transport results because the trajectory is unstable and incomplete.
- No block/replica uncertainty estimate is scientifically valid for this three-branch set because all selected replicas are incomplete.
- No automatic continuation was performed: the terminal condition is scientific instability, not a wall-time interruption.

## Required before formal publication

1. Resolve the force/geometry instability with a validated model/protocol change or a user-approved replacement branch.
2. Re-run all approved replicas to the complete absolute-step target with checkpoint, identity, finite-state, geometry, and continuity gates passing.
3. Recompute block/replica uncertainty, mapped hop events, MSD/Dxy/VACF and, when a physical heat-flux observable is available, HFACF/running integral.
