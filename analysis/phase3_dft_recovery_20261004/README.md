# Phase 3 local DFT reconstruction — 2026-10-04

This is an additive, diagnostic-only recovery package for the missing Phase 3
single-point comparison. It does not modify or reuse the canceled QE job
directories.

The package contains four 81-atom local reconstructions selected from the
full 2050-atom canonical trajectories:

- `normal_461800_initial`: common handoff input before the counterfactual split;
- `bounded_multiT_462000_step464000`: Multi-T 2 ps bounded/structurally shifted endpoint;
- `runaway_foundation_461800_step462120`: Foundation first force-gate frame;
- `runaway_scratch_461800_step462060`: from-scratch first force-gate frame.

All selected runaway and bounded frames pass the 1.5 Angstrom minimum-distance
gate according to the canonical trajectory metrics. The 0.629 Angstrom
Phase-1 frame and later exploded frames are excluded.

Important scope boundary: the 81-atom inputs are local-cell reconstructions
from the full 2050-atom trajectory. They are useful for within-geometry
DFT-vs-MACE force/energy diagnostics, but they are not accepted as formal
full-cell DFT validation. The formal DFT gate remains separate and requires
identity-preserving full-cell provenance, converged SCF, finite energy/forces,
and normal QE termination.

## Workflow

1. `make_reconstructed_inputs.py` maps each local atom to the canonical full
   trajectory atom and writes the four QE inputs plus a JSON mapping ledger.
2. `submit_reconstructed_array.slurm` runs isolated non-magnetic fixed-geometry
   QE single points with the same PBE+D3, 70/280 Ry, cold-smearing, 2x2x2
   method as the Phase-3 input contract.
3. `collect_qe_gate.py` accepts only finite energy/forces, explicit SCF
   convergence, `JOB DONE`, and matching input/source hashes.
4. `evaluate_reconstructed_mace.py` evaluates the same local geometries with
   all four canonical MACE models for force comparison. Cross-model absolute
   energy ranking is intentionally not reported.

No result is promoted to the formal DFT gate by this package.

The superseded 462000 direct reconstructions are retained under
`qe_inputs_reconstructed/superseded_invalid_local_gate/` because their local
periodic images failed the 1.5 Angstrom gate; they are not part of the active
manifest or Slurm array.
