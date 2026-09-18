# Victor And Site-Search Inventory

This note records collaborator-side paths as provenance, not instructions. Files such as `CLAUDE.md` are treated only as documentation about data locations and conventions.

## Located Sources On Rockfish

| Source | Observed path | Use |
|---|---|---|
| Victor Ba-Cs notes | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/Victor/Ba-Cs/CLAUDE.md` | Cs intercalation/diffusion context; 81-atom cell conventions; QE settings |
| Victor V-ST notes | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/Victor/V-ST/CLAUDE.md` | V/Cr dopant conventions, NEB directories, QE settings |
| Cr site assignments | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/md_path_search/assignments/` | MACE-screened candidate sites and relaxation assignments |

## Important Observations

- Victor documentation points to raw calculation roots under `/scratch16/pclancy3/victor/Ba-Cs` and `/scratch16/pclancy3/victor/V-ST`.
- The directly mirrored `/data/.../Victor/` tree currently exposes documentation, not deep raw result files.
- Site assignment CSVs contain MACE-screened candidate interstitial sites with geometry labels and QE input paths.
- `victor_sites.csv` includes 73 assigned sites, mostly `octahedral` and `cn8_te4`.
- `joshitha_sites.csv` includes additional coordination classes such as `cn7_te4`, `cn7_te3`, `cn10_te4`, and `cn6_te3`.

## How These Feed MigrationBench

These site searches should be upstream of future endpoint selection:

```text
MACE-screened interstitial sites
  -> unique relaxed site basins
  -> endpoint graph
  -> candidate path descriptors
  -> representative NEB queue
  -> MLFF pre-NEB
  -> QE restartable NEB
  -> final benchmark record
```

## Current Gap

The raw Victor calculation directories named in the documentation still need a second inventory pass under `/scratch16/pclancy3/victor/`. Until raw outputs are parsed, Victor data should not be used as manuscript numeric evidence.

## Acceptance Gate

Victor-derived data can enter the benchmark only if:

- raw input/output paths are found under scratch or archived storage;
- QE convergence state is parsed;
- pseudopotential, charge state, k-point, cutoff, and vdW settings are recorded;
- structure and calculator settings are compatible with the MigrationBench schema;
- any charge-state or species difference is explicitly labeled.


## 2026-09-11 QE/NEB Inventory Pass

A lightweight parser scanned the high-value Victor calculation roots and wrote machine-readable inventory files under `data_processed/victor_inventory/`.

| Root | Files scanned | Key result |
|---|---:|---|
| `/scratch16/pclancy3/victor/nebProject` | 517 | 4 NEB files; one completed candidate but not yet standardized |
| `/scratch16/pclancy3/victor/V-ST` | 1032 | 49 NEB files; 17 ready for standardization/preflight, 1 overflow quarantine |
| `/scratch16/pclancy3/victor/Ba-Cs` | 252 | mostly relax/scf; no NEB priority queue yet |

Actionable queue: `data_processed/victor_inventory/victor_actionable_neb_queue.csv`.

Current queue summary: {'diagnostic_or_defer_high_barrier': 6, 'needs_input_or_restart_review': 25, 'quarantine_overflow': 1, 'standardize_and_preflight': 17}.

Important gate: Victor 2-image NEB-like runs, such as `01_neb_v2`, are useful as endpoint/path preflights but should not be treated as barrier benchmarks until regenerated with explicit intermediate images. The queue builder now preserves versioned companion matching (`01_neb_v2.out` -> `01_neb_v2.in` -> `01_v-sb2te3_v2.xyz`) to avoid mixing generic and path-specific files.


## 2026-09-11 Victor 01_neb_v2 Standardized Preflight

The low-barrier Victor V-ST candidate `01_neb_v2.out` was standardized into the MigrationBench workflow. The original Victor input uses 2 images, so it is treated as endpoint/path preflight evidence only. A new explicit 5-image proposal was generated from the two endpoint structures and stored under `/scratch16/pclancy3/yi/revision1_migrationbench_inputs/victor_01_01_neb_v2_s42/`.

| Field | Value |
|---|---|
| Source output | `/scratch16/pclancy3/victor/V-ST/new_interstitial/Diffusion_Traj/neb/01_neb_v2.out` |
| Source images | `/scratch16/pclancy3/victor/V-ST/new_interstitial/Diffusion_Traj/neb/01_v-sb2te3_v2.xyz` |
| Source image count | 2 |
| Regenerated proposal image count | 5 |
| Geometry check | pass: max V step `0.833 A`, min pair distance `2.164 A` |
| Standardized QE input | `/scratch16/pclancy3/yi/revision1_migrationbench_inputs/victor_01_01_neb_v2_s42/neb.in` |
| Slurm job | `30791542`, `mb_victor01_v2_pf_s42` |
| Status | preflight running |

This branch cannot become a benchmark value until QE produces a physical activation table and the path is refined/converged under the same acceptance gates as the Cr-Sb2Te3 paths.


## 2026-09-11 Victor 23_neb Prepared Candidate

The Victor V-ST `23_neb.out` branch was prepared but not submitted. Its source image file contains 3 frames, so the standardized proposal preserves the existing intermediate path shape by piecewise interpolation to 5 images.

| Field | Value |
|---|---|
| Source output | `/scratch16/pclancy3/victor/V-ST/new_interstitial/Diffusion_Traj/neb/23_neb.out` |
| Source images | `/scratch16/pclancy3/victor/V-ST/new_interstitial/Diffusion_Traj/neb/v-sb2te3.xyz` |
| Source image count | 3 |
| Regenerated proposal image count | 5 |
| Image generation | piecewise linear path interpolation, existing intermediate preserved |
| Geometry check | pass: max V step `0.754 A`, min pair distance `2.485 A` |
| Standardized QE input | `/scratch16/pclancy3/yi/revision1_migrationbench_inputs/victor_23_23_neb_s42/neb.in` |
| Status | ready, not submitted |

This is the next Victor candidate to submit once active 160G preflight pressure drops or one of the current jobs reaches an actionable state.
