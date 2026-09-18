# Coverage-Gap Path Repairs

## Corrected Decision

The first `coverage_gaps_v1` export and jobs `30850728`-`30850731` are
quarantined. The historical XYZ files did not contain a periodic cell, so those
inputs and MACE outputs have `pbc="F F F"`. Slurm completion does not rescue
scientifically invalid boundary conditions. The immutable correction record is
`state/invalid_mlff_repair_ab_s46.json`.

The trajectory portfolio identified `1-2`, `1-3`, and `1-5` as necessary coverage witnesses with minimum separations below the `1.8 A` geometry gate. The repair is deliberately restricted to periodic unwrapping plus displacement of the Cr migrant away from its closest host atom. Host atoms and both endpoints are immutable.

| Path | Before min distance (A) | After min distance (A) | Endpoint gate | Action |
|---|---:|---:|---|---|
| `1-2` | 1.410172 | 1.425069 | fail: final endpoint 1.425069 | replace/redefine endpoint; no MACE or QE handoff |
| `1-3` | 1.376673 | 1.800000 | pass | periodic direct/repaired MACE and QE-relax A/B |
| `1-5` | 1.408495 | 1.800000 | pass | diagnostic A/B; path discontinuity remains |

The corrected `coverage_gaps_pbc_v3` records attach the exact QE cell before
minimum-image unwrapping. For `1-3`, only Cr is moved in images 2 and 3. For
`1-5`, only Cr is moved in image 2. No host atom or endpoint is changed. These
are deterministic initializer transformations, not physical relaxation labels.
Rockfish CPU Slurm smoke `30850824` passed the periodic-cell, atom-order,
endpoint, and per-image distance gates. The superseded resampling smoke
`30850820` was cancelled; it changed image count and therefore was not a clean
A/B control. Full-suite Rockfish rerun `30851071` passed `58/58` tests,
including metadata isolation, clean-timeout continuation, cumulative lineage,
MLFF periodic-output rejection, and paired/single-job configuration hash gates,
and reproduced
the canonical `1-3` and `1-5` repaired extxyz files byte-for-byte.

## MACE A/B

Four replacement CPU jobs use the same explicit MACE-MP-0 medium checkpoint,
seed 47, five images, 200 FIRE steps, `fmax=0.10 eV/A`, spring constant `0.10`,
8 CPUs, 48 GB, and six-hour walltime:

| Path | Direct job | Repaired job |
|---|---|---|
| `1-3` | `30850830` (`mb_mrep13p_dir_s47`) | `30850831` (`mb_mrep13p_fix_s47`) |
| `1-5` | `30850832` (`mb_mrep15p_dir_s47`) | `30850833` (`mb_mrep15p_fix_s47`) |

They are monitored only by the three-hour heartbeat. Comparison must include
optimizer steps, internal true force, internal NEB force, barrier proxy,
per-image energy changes, and final path similarity. A speedup claim requires
both variants to converge to the same final mechanism. Because endpoints are
not independently DFT accepted, none of these diagnostic runs may be handed to
QE NEB production.

`scripts/migrationbench/compare_mlff_repair_ab.py` is the fail-closed comparator. It verifies repair-manifest binding and identical model/protocol identity, measures the final path with the same historical `D_traj` scales, requires final-path similarity of at least `0.80` under `configs/mlff_repair_ab_comparison_v1.json` before treating the mechanisms as equal, and emits no speedup when either run is unconverged or the mechanism changes. The comparison policy is separate from the immutable submission configuration.

Configuration and per-image repair ledgers are under
`configs/mlff_repair_ab_coverage_gaps_pbc_s47.json` and
`data_processed/path_repairs/coverage_gaps_pbc_v3/`.

## Paired DFT Relaxation

Six 24-hour CPU QE relaxations compare the exact direct and transformed points:
`1-3` images 2 and 3, and `1-5` image 2. Jobs `30850842`-`30850847` use two MPI
ranks, 160 GB, and `max_seconds=84600`. The primary efficiency measures are
ionic steps and cumulative SCF iterations. Walltime is secondary. No speedup is
reported unless both members converge below `0.05 eV/A` and the final Cr/local
host geometry establishes the same basin. A different basin is a mechanism
change, not an acceleration result.

The six submitted inputs contain the correct coordinates. Their direct-branch
manifests inherited a nonzero paired-transform field in metadata only; the
identity interpretation and generator correction are recorded in
`configs/qe_geometry_transform_relax_ab_s47.json`. This has no calculation
impact and the transformed manifests remain authoritative for displacement.
