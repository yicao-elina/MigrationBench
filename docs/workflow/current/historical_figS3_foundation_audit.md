# Historical Fig. S3 Foundation Self-NEB Audit

## Identity

- Historical Rockfish directory: `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806-NEB/0-foundation-omat`
- Historical Slurm job: `10199010`
- Model: `MACE-matpes-pbe-omat-ft.model`
- Model SHA-256: `e618ad582b84239905b9c3b77ce6e9ce111b0ecd1533223a1a6aac7a696b8aa0`
- Images: 12 total, including 10 movable images
- Composition: `Cr2Sb32Te48` (82 atoms)
- Original optimizer: ASE climbing-image NEB with BFGS
- Historical stop: iteration 295, projected NEB `fmax=0.049927 eV/A`

This is not the main-text 1-4 path. Path 1-4 has composition `CrSb24Te36`
(61 atoms), one Cr atom, a different cell, and five images. The original paper
therefore compared results from different physical systems/path definitions
without saying so.

## Historical Calculator Defect

The historical runner assigned MACE calculators only to movable images. The
two endpoint structures retained stored DFT single-point calculators. In the
last trajectory band, endpoint energies are about `-179665 eV`, while internal
MACE energies are about `-309 eV`. ASE NEB uses image energies in tangent/path
construction, so the issue affects the optimization itself; it is not merely a
plotting or serialization error.

The final XYZ has an additional serialization defect. One shared calculator
was assigned to all images immediately before writing, causing all 12 stored
energies to equal `-309.013281 eV`. The original runner consequently printed a
zero forward barrier. A later plotting script changed the definition to
`max(E)-min(E)`, but the stored constant energies cannot support any nonzero
value.

## Independent Uniform-Checkpoint Rescoring

Rockfish Slurm job `30851756` rescored the final 12 geometries using the exact
historical Foundation checkpoint for every image and reconstructed the ASE NEB
residual. It reproduced the historical projected residual to numerical
precision (`0.049927283 eV/A`), confirming the geometry/iteration alignment.

The uniformly rescored profile gives:

| Quantity | Value |
|---|---:|
| Forward endpoint-referenced barrier | `0.000000 eV` |
| Reverse endpoint-referenced barrier | `0.379052 eV` |
| Whole-path max-min energy span | `1.037407 eV` |
| Endpoint energy difference | `-0.379052 eV` |
| Highest-energy image | image 1, frozen endpoint |
| Lowest-energy image | image 11, movable image |

None of these values is `0.41 eV`. More importantly, uniform final-band
rescoring cannot repair a path whose optimization used inconsistent endpoint
and internal-image energy providers. The historical result is therefore an
invalid diagnostic, not an MLFF migration-barrier benchmark.

## Manuscript Decision

Remove the quantitative `0.41 eV`, "exceptional", and "chemical accuracy"
claims. Replace Fig. S3 with a fully rerun, losslessly logged same-system
self-NEB comparison, or retain only a clearly labeled failure-analysis panel.
Do not compare this 82-atom Cr2 path numerically with the 61-atom Cr1 1-4 path.
