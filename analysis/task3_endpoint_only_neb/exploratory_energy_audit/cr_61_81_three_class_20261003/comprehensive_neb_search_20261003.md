# Comprehensive NEB search for additional Cr–Sb2Te3 paths

Date: 2026-10-03  
Scope: the historical `3.neb`-equivalent data, the local MigrationBench snapshots, the xforce source tree, and nearby preflight/proposal outputs.

## Bottom line

Yes: there is an additional 81-atom Cr path that is a strong candidate for the unified `in_gap_diffusion` class:

`2.2-layer-dislocation/2.cr-in-gap/sb2te3.xyz`

The migrant Cr remains in the inter-layer gap while translating laterally. Across the ten saved images its z coordinate is 23.645–23.880 Å. The adjacent Te planes are approximately 22.25–22.38 Å and 24.90–25.05 Å, so the trajectory stays near the middle of that gap rather than crossing into a quintuple layer. The endpoint displacement is 2.487 Å and the path length is 2.505 Å. This is geometrically distinct from the existing `2D_421_Diffusion_traj/neb_1..5` family.

It is not yet a publishable DFT barrier: the saved `neb.out` stops during iteration 22, has no `JOB DONE` or convergence marker, and the last reported forward activation energy is 3.820331 eV. It should therefore be added as an exploratory 81-atom `in_gap_diffusion` path for Foundation-model testing, while its DFT value remains quarantined until the NEB is rerun or completed.

## Sources searched

1. The local historical NEB snapshot under `data_raw/historical_neb/layer_81` and `legacy_61`.
2. The published 81-atom path-classification inventory and the current complete Cr 61-path audit worktree.
3. The xforce source tree:
   `/Users/alina/Project/xforce-paper/packages/xforce/data/Cr-Sb2Te3-NEB`
4. The nearby `26MIgrationBench` preflight/proposal directories, including the two Victor V paths.
5. Recursive inventories of `neb.in`, `neb.dat`, `neb.out`, `sb2te3.xyz`, and `sb2te3.path` under `/Users/alina/Project`, `/Users/alina/Documents/Codex`, and `/Users/alina/Documents/ChatGPT`; no literal directory named `3.neb` or `quintuple-layer` was found locally, so the xforce tree is the historical source-of-truth equivalent.

The xforce Cr–Sb2Te3 tree contains 50 `neb.in`, 43 `neb.out`, 39 `neb.dat`, 39 saved `sb2te3.xyz`, and 40 `sb2te3.path` files. Most are alternate restarts, no-correction branches, or generated proposals rather than unique 81-atom Cr paths.

## Unified three-class inventory

### 61-atom Cr system

The canonical DFT-D3 set has seven requested routes 1-2 through 1-8. The evidence-supported mapping is:

- `gap_to_QL`: 1-2, 1-3, 1-4, 1-5, 1-7
- `in_gap_diffusion`: 1-6, 1-8; 1-8 is lower confidence because the path is disordered and force-convergence evidence is weak
- `within_QL`: none identified in the 61-atom set

The xforce tree also contains two alternate ten-image 1-5 branches under `1.vdw_corr_DFT_D3/1-5/`; these are branch/restart variants, not new route IDs.

### 81-atom Cr system

The existing `2.2-layer-diffusion/2D_421_Diffusion_traj/neb_1..5` set contains five distinct paths. The continuous geometry descriptors classify all five as deep-penetration trajectories rather than clean gap-only diffusion. For the paper's three-class language, they are best reported as `within_QL (deep-penetration realization)` with an explicit geometry qualifier, not as pure in-gap diffusion.

The dislocation family adds four relevant records:

| record | geometric reading | three-class handling | NEB state |
|---|---|---|---|
| `2.cr-in-gap` | Cr stays in one gap plane and translates laterally | `in_gap_diffusion` candidate | interrupted at iteration 22; no completion marker |
| `b-path2/2.cr-in-gap` | endpoint displacement only 0.165 Å; path dives toward a host plane | invalid/duplicate, exclude | reaches iteration 102; no completion marker |
| `migration` | Cr moves from a gap toward a host-layer region | deep penetration / possible gap-to-QL transition | input and path only; no `neb.out` |
| `c-after_aimd/2.cr-in-gap` | 81-atom input exists, but no saved trajectory | incomplete, do not classify | only one NEB iteration in `neb.out` |

Thus the comprehensive 81-atom search finds one additional, physically meaningful `in_gap_diffusion` candidate: `2.2-layer-dislocation/2.cr-in-gap`. It does not find a second completed or converged 81-atom gap→gap NEB.

## Excluded but relevant families

- `2.2-layer-diffusion/1.Gap-Center`: the root trajectories are 82-atom systems, not the requested 81-atom single-Cr system.
- `1.Gap-Center/441/Cr3` and `Cr5`: these are 163/165-atom multi-Cr in-gap pathways. They are valuable mechanistic context for cooperative or multi-defect diffusion, but cannot be merged into the 81-Cr Foundation benchmark.
- `E3_NEB_GENERATION/neb/ID_1_TypeA` and `TypeB`: 82-atom proposals, not 81-atom Cr1 routes.
- `victor_01_01_neb_v2` and `victor_23_23_neb`: 81-atom V proposals, not Cr. Their generic geometry audit labels them as deep penetration, and they are preflight proposals rather than completed production NEBs.
- `2.2-layer-diffusion/neb-0-1` and the automated path-search directories: they are proposal/restart machinery associated with the existing diffusion campaign; they do not add a new saved, independently identified 81-Cr gap→gap route.

## Interpretation for the next Foundation-model benchmark

The correct update is not to force all 81 paths into a symmetric 3-by-3 table. The evidence supports:

1. 61-atom Cr: five `gap_to_QL`, two `in_gap_diffusion`, and no confirmed `within_QL`.
2. 81-atom Cr: five existing deep-penetration/within-QL candidates plus one new exploratory `in_gap_diffusion` candidate (`2.cr-in-gap`).
3. A separate 81-atom `migration` input can be retained as a gap-to-QL/deep-penetration proposal, but it must not be counted as a completed DFT path.

The Foundation-model calculation can be run on all finite saved 81-atom trajectories, including the new `2.cr-in-gap` candidate, but every DFT-derived reference value must remain labelled exploratory until SCF and NEB convergence are independently demonstrated. The machine-readable inventory is in `comprehensive_neb_inventory_20261003.csv`; the geometry output for the three dislocation candidates is in `dislocation_81_geometry_classification.csv` and `.json`.

The earlier statement “gap→QL: 3 / in-gap: 3 / within-QL: 3” was limited to the Cu/Li/Na endpoint-only comparison subset; it should not be reused as the Cr 61/81 route count. :codex-annotation{index="1"}
