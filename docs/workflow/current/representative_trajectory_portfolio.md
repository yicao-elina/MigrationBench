# Representative Trajectory Portfolio

## Scope

This is a diagnostic selection over 12 distinct historical lineages: seven 61-atom paths and five 81-atom paths. Current `1-6_qe_r3` and `1-7_qe_r3` are refinements of existing lineages and are deliberately not double-counted as new mechanisms. No selected path is promoted to an accepted DFT reference by this analysis.

## Executable Distance

Each path is parameterized by normalized cumulative configuration-space arc length and resampled at 21 points. The implemented distance is:

```text
D_traj = 0.30 D_geometry_Frechet
       + 0.25 D_local_environment
       + 0.20 D_displacement_field
       + 0.15 D_energy_shape
       + 0.10 D_bond_topology
```

- `D_geometry_Frechet` compares element-pair-stratified, sorted periodic pair-distance spectra.
- `D_local_environment` compares sorted Cr-host distance signatures by element.
- `D_displacement_field` compares element-stratified atomic displacement magnitudes.
- `D_energy_shape` compares min-max-normalized energy profiles; it cannot compare absolute energies from incompatible calculators.
- `D_bond_topology` compares element-resolved coordination counts over declared radii.

Each component is normalized by the median nonzero distance in its explicit system. Forward and reversed path alignments are both evaluated, and one orientation is chosen jointly by the minimum weighted total. Cross-system distance is undefined and fails closed. Pair-distance descriptors are invariant to rigid translation, rigid rotation, and within-element permutation; PBC uses MIC where a cell is available.

## Coverage Result

Selection uses uniform candidate importance and greedy facility location:

```text
Coverage(M) = mean_i max_(j in M) exp(-D_traj(i,j))
```

| System | Candidate lineages | Representatives to reach 0.90 | Ordered representatives | Coverage |
|---|---:|---:|---|---:|
| `Sb2Te3Cr_61` | 7 | 5 | `1-7`, `1-5`, `1-8`, `1-3`, `1-2` | 0.900583 |
| `Sb2Te3Cr_81` | 5 | 4 | `81_neb_2`, `81_neb_5`, `81_neb_3`, `81_neb_1` | 0.900223 |

For the 61-atom set, `1-2`, `1-3`, and `1-5` have minimum separations below `1.8 A`. They are coverage witnesses for poorly represented regions, not runnable production initializations. Their downstream action is repair or replacement by a valid trajectory in the same descriptor region. The other selected rows still require independently accepted endpoint basins and converged QE NEB.

If representative centers are restricted to the four currently geometry-valid 61-atom paths (`1-4`, `1-6`, `1-7`, `1-8`), the maximum attainable coverage is only `0.735641`. The nearest valid analogs are `1-6` for invalid `1-2` and `1-3`, and `1-4` for invalid `1-5`, but their similarities are only `0.4251`, `0.3949`, and `0.3295`. Therefore the 61-atom inventory does **not** meet the 0.90 target with runnable geometries; these three regions require new or successfully repaired paths.

## Reproducibility

The implementation is `scripts/migrationbench/select_representative_trajectories.py`; configuration is `configs/historical_trajectory_portfolio.json`. Local and Rockfish CPU Slurm outputs have byte-identical pairwise-distance, representative-selection, and coverage-gap CSV files. Final Slurm smoke job `30850696` completed in 16 seconds with seed 42 and passed candidate-count, system-isolation, monotonic-coverage, target-coverage, valid-center ceiling, and representative-action-label gates. Earlier jobs `30850673` and `30850687` validated preceding numerical/output stages; duplicate submission `30850674` was cancelled before running after the first submit response was lost at the client timeout boundary.

Artifacts are under:

- `data_processed/representative_trajectory_portfolio/`
- `cluster/mb_trajrep3_sm_s42_30850696/representative_trajectory_portfolio/`

This portfolio is a versioned prior over the currently observed historical universe. It does not prove global coverage of all physically possible migration mechanisms. Coverage must be recomputed whenever accepted basins, symmetry-generated paths, collaborator paths, or new MLFF/QE mechanisms enter the candidate population.
