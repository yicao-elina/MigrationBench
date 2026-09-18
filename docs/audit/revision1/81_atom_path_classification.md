# 81-Atom Path Classification Audit

Date: 2026-09-09

## Dataset And Grain

Dataset: `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/2.2-layer-diffusion/2D_421_Diffusion_traj/neb_{1..5}`.

Grain: one NEB path. Each path has 10 images and 81 atoms (`Cr1 Sb32 Te48`). The migrating atom is the single Cr atom at zero-based index 80.

Local synchronized evidence:

- `data_processed/cluster/81_atom_2D_421_Diffusion_traj/neb_*/sb2te3.xyz`
- `data_processed/cluster/81_atom_2D_421_Diffusion_traj/neb_*/neb.out`
- `data_processed/cluster/81_atom_2D_421_Diffusion_traj/path_geometry_classification.csv`
- `data_processed/cluster/81_atom_2D_421_Diffusion_traj/path_geometry_classification.json`

## Checks Performed

Script: `scripts/migrationbench/classify_neb_path_geometry.py`.

For each path, the script records:

- Cr path length and endpoint displacement.
- Minimum and mean Cr-host distance.
- Cr coordination count within 3.2 A for each image.
- Separation from host z-planes and z-density overlap.
- Continuous `penetration_score` and `gap_score`.
- A derived class label with confidence.
- Historical QE activation barrier and last complete NEB iteration max image error.

## Findings

| path | class | penetration score | gap score | mean Cr coordination | min Cr-host distance (A) | candidate barrier (eV) | final max image error (eV/A) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `neb_1` | deep penetration | 0.691 | 0.350 | 6.0 | 2.454 | 1.034728 | 0.355916 |
| `neb_2` | deep penetration | 0.736 | 0.210 | 5.5 | 2.383 | 1.561262 | 0.635720 |
| `neb_3` | deep penetration | 0.718 | 0.300 | 6.0 | 2.378 | 1.691053 | 0.231831 |
| `neb_4` | deep penetration | 0.740 | 0.216 | 5.6 | 2.373 | 2.063858 | 0.526984 |
| `neb_5` | deep penetration | 0.741 | 0.235 | 5.0 | 2.333 | 1.414868 | 0.319104 |

Interpretation: `neb_2`, `neb_3`, and `neb_4` match the remembered deep-penetration class. `neb_1` and `neb_5` are also better described as deep penetration than in-gap diffusion because the Cr atom remains closely coordinated to host atoms and overlaps host z-planes throughout the path.

## Data-Quality Risk

The path labels are now auditable, but the historical DFT barriers remain manuscript-quarantined. All five jobs report `JOB DONE`, but the last complete NEB iteration force errors are 0.23-0.64 eV/A, far above the production gate of 0.03 eV/A. These barriers are therefore useful as candidate ordering and path-discovery evidence, not final reference values.

## Remediation

Run MACE preconditioning from each historical `sb2te3.xyz`, then generate QE `neb.in` from the MACE-relaxed images with a companion config/manifest. The QE refinement should run on CPU resources under `/scratch16/pclancy3/yi/revision1_migrationbench_runs` and must pass the parser gate before entering manuscript tables.
