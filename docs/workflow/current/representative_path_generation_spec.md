# Representative NEB Path Generation Specification

## North Star

Construct a small, traceable set of migration mechanisms that covers the predefined migration space, then validate one complete path through endpoint relaxation, nonlinear initialization, MACE NEB, and converged DFT NEB before scaling the workflow.

The endpoints need not be equal in energy. They must represent reproducible local basins. For a segment with endpoint energies `E_A`, `E_B` and saddle energy `E_TS`:

```text
Delta E = E_B - E_A
E_a_forward = E_TS - E_A
E_a_reverse = E_TS - E_B
E_a_forward - E_a_reverse = Delta E
```

Equal-energy endpoints are useful for symmetric mechanisms but are not an acceptance requirement.

## Migration Space

A candidate belongs to one explicit system and one endpoint graph:

```text
Omega = (system_id, host_structure_hash, migrant_identity, candidate_basins, allowed basin pairs)
```

Candidate basins may come from independently relaxed structures, low-energy internal NEB images, collaborator site searches, or symmetry-generated equivalents. Sites from different cells, compositions, charge states, or unresolved source namespaces cannot be paired.

Symmetry-equivalent structures are grouped only after periodic atom matching. The durable basin ID is based on system ID, composition, cell, matched coordinates, and relaxation protocol. A label such as `1-7` is an alias, not an identity.

## Endpoint Evidence

An internal NEB image is a basin candidate when it is lower than both neighboring images by at least the configured tolerance. It becomes an accepted basin only after standalone fixed-cell DFT relaxation passes all of these gates:

- QE and BFGS end cleanly.
- Final maximum atomic force is at most `0.05 eV/A`.
- A repeat relaxation changes the structure by at most `0.10 A` and lowers energy by at most `0.02 eV`.
- Geometry has no configured close contact.
- Input, output, every ionic step, energy, atomic forces, and hashes are stored.

Frozen NEB endpoints do not provide evidence of local stability. A low energy by itself also does not prove a minimum; a structure may still have a large force or relax into another basin.

## Curve Representation

For images `R_i`, define cumulative configuration-space arc length using periodic minimum-image displacement after atom matching:

```text
l_0 = 0
l_i = l_(i-1) + RMS_atoms(MIC(R_i - R_(i-1)))
s_i = l_i / l_N
```

Every trajectory is resampled on a common `s in [0,1]` grid before comparison. The path descriptor stores:

- periodic aligned geometry and discrete Frechet distance;
- local migrant environment descriptors and coordination topology;
- migrant and host displacement fields, including direction cosines;
- normalized energy profile and transition-state position;
- bond-graph changes, path length, curvature, and tortuosity.

The trajectory distance is:

```text
D_traj = 0.30 D_geometry
       + 0.25 D_local
       + 0.20 D_displacement
       + 0.15 D_energy
       + 0.10 D_topology

S_traj = exp(-D_traj)
```

Every component is normalized from a declared reference population. Missing energy does not become zero; geometry-only candidates use an explicitly versioned reduced distance.

## Nonlinear Initialization

Candidate initial paths are generated in this order:

1. Reuse a matching subpath from a historical DFT trajectory when its endpoints map to the accepted basins.
2. Splice compatible historical subpaths at a shared accepted basin and reparameterize by arc length.
3. Warp the nearest representative path to new endpoints using periodic host alignment and migrant-local coordinates.
4. Use IDPP only when no reusable curved path exists.
5. Use Cartesian linear interpolation as a diagnostic baseline, not the preferred production initializer.

Each initializer produces a branch manifest. Cartesian averaging between different mechanisms is forbidden because it can create close contacts or erase a physically distinct route.

The executable geometry layer is
`scripts/migrationbench/generate_nonlinear_neb_candidates.py`. It implements a
MIC baseline, clearance-aware sinusoidal migrant arcs, and historical-curvature
warp, with endpoint PBC equivalence, dual migrant/all-atom duplicate gates, and
geometry-only extxyz output. Rockfish Slurm smoke `30850304` validates its
portable output with ASE; production remains gated on accepted endpoint minima.

## Representative Selection And Coverage

Path quality is scored before expensive DFT:

```text
Q = 0.30 Q_endpoint
  + 0.20 Q_geometry
  + 0.25 Q_coverage_gain
  + 0.15 Q_energy
  - 0.10 Q_redundancy
```

Representative paths are selected by weighted facility location with a farthest-point tie breaker. For candidate importance `p_i` and selected set `M`:

```text
Coverage(M) = sum_i p_i max_(j in M) exp(-D_ij) / sum_i p_i
```

The initial target is `Coverage >= 0.90`. Coverage is always reported with its candidate population, descriptor version, weights, and unresolved regions; a percentage without those fields is invalid.

The first executable historical-trajectory selector is
`scripts/migrationbench/select_representative_trajectories.py`, configured by
`configs/historical_trajectory_portfolio.json`. It uses 21-point arc-length
resampling, direction-joint matching, median within-system component scales,
and strict cross-system isolation. Over the current 12-lineage diagnostic
population, five of seven 61-atom paths reach `0.900583` coverage and four of
five 81-atom paths reach `0.900223`. Three selected 61-atom rows are invalid
close-contact coverage witnesses and must be repaired or replaced, not sent
directly to production. Restricting centers to currently valid 61-atom
geometries caps coverage at `0.735641`, so valid coverage remains incomplete.
See `docs/representative_trajectory_portfolio.md`.

## Pilot Path State Machine

The first complete pilot is selected only after the current endpoint relaxations finish. Current provisional candidates are the lower-energy image-2 basins found in `1-6` and `1-7` plus a validated partner basin. The states are:

```text
candidate_image
  -> endpoint_relax_running
  -> accepted_basin | rejected_basin
  -> endpoint_pair_defined
  -> nonlinear_path_generated
  -> mace_neb_running
  -> mace_path_accepted | mace_path_quarantined
  -> dft_neb_running/restarting
  -> dft_neb_converged
  -> mechanism_clustered
  -> dataset_and_manuscript_eligible
```

DFT acceptance requires an interior saddle, path force at most `0.03 eV/A`, a final check at `0.02 eV/A`, barrier drift at most `0.02 eV`, no overflow, and independently validated endpoint basins.

## Iteration-Level Provenance

For endpoint relaxations, every BFGS ionic step is stored as extxyz with energy and atomic forces. For MACE and DFT NEB, every optimizer iteration stores every image structure, image energy, atomic forces, NEB/tangent force, reaction coordinate, spring settings, optimizer state, and convergence flags. Raw logs remain immutable and are linked by SHA-256.

No benchmark path, endpoint, transition state, or post-outcome DFT image may enter training for the model evaluated on that mechanism. Splits are grouped by `system_id + endpoint_basin_ids + mechanism_cluster_id`, not by individual image, which prevents adjacent-image and same-mechanism leakage.

## Current Evidence

The topology audit over 14 real paths found 9 internal low-energy basin candidates and 23 path segments. Current `1-6` and `1-7` r3 paths each contain a lower-energy image 2 and endpoint-dominated segments, so their reported `0.820400 eV` and `0.557933 eV` values are diagnostic snapshots, not accepted single-barrier references.

The canonical v3 site graph contains 191 namespaced sites, 12052 same-system candidate pairs, and 24 representatives. Raw pipeline metadata, one common host/template, a shared reference cell, and `245/245` matching `Sb32Te48Cr1` input compositions establish that Victor and Joshitha are assignment namespaces within one physical system. The 5696 cross-namespace pairs are therefore permitted only under explicit `system_id=sb2te3_4x2x1`. The unnamespaced original graph remains superseded; v2 records the earlier fail-closed isolation before compatibility was proved.
