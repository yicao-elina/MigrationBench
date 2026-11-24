# Current collaborator DFT/NEB audit

Audit date: 2026-09-29. Raw files remain on Rockfish. Values below are
observations extracted from output text; they are not accepted scientific
results until the stated gates pass.

## V@Sb2Te3 NEB candidate summary

- Candidate trajectory records currently catalogued: **10**.
- Runtime-failed records: **3** (`neb2`, `neb3`, `neb4`; MPI signal 9).
- Completed-but-unvalidated records: **1** (`neb5`; `JOB DONE`, ~0.0059 eV,
  suspiciously collapsed path).
- Partial records with activation-energy lines but no explicit completion marker:
  **6** (`01`, `12`, `23`, `34`, `45`, `55_cr` in `neb_2`).
- Accepted publication-grade NEB barriers from this audit: **0**.

## First path-structure cross-check

`neb_2/01` is more informative than a filename-only candidate: its input
declares 10 images and 81 atoms, its XYZ contains an 81-atom structure, and
its `.dat` profile contains 10 rows with a reported forward maximum of about
1.288 eV. This upgrades it from “unstructured candidate” to a reproducible
partial trajectory record, but not to an accepted result: the output stops
during iteration 19, with per-image error values up to 1.305707 eV/A. The
reported 1.288 eV profile is therefore an intermediate failed-convergence
state, not a final barrier. `neb5` declares 3 images and 61 atoms and has `JOB DONE`, but its
near-zero barrier still requires the same identity/force audit.

The activation-energy lines are retained for forensic comparison and to guide
recovery, but are not used in manuscript tables or DFT-vs-MACE error metrics.

The 12 Cr/Mn/Mo endpoint outputs all contain `JOB DONE` and optimizer
convergence markers, but reported final `Total force` values range from
0.001797 to 0.039689 in QE-reported units. These are not `max force` values,
so conversion to eV/Angstrom is provisional. Four TMi4 outputs also lack an
extracted final energy. All 12 therefore remain `force_threshold_unverified`
until final-coordinate and force-vector records are parsed.

The first raw force-vector parse was initially misleading because it included
QE's `The non-local contrib. to forces` sub-block. The corrected parser uses
only the block from `Forces acting on atoms` up to that marker. Under a
diagnostic 0.03 eV/Angstrom threshold, TMi1--TMi3 pass for Cr, Mn, and Mo;
all three TMi4 endpoints fail. This is a parser-validated candidate result,
but final promotion still requires exact QE-unit conversion, final-energy
extraction for TMi4, structure/site identity, and a project-approved force
threshold.

The corresponding inputs use `ion_dynamics='bfgs'` and `cell_dynamics='bfgs'`
and contain no observed `if_pos` or explicit constraint tokens. Thus the large
raw force components cannot currently be explained away as frozen atoms. The
force parser must reconcile the final force block with the printed total-force
summary and cell-optimization semantics before endpoint promotion.

## New endpoint portfolio: Cr/Mn/Mo@Sb2Te3

The `ST-other/{cr_d3,mn_d3,mo_d3}` trees contain 12 named 61-atom D3 relax
outputs (four TMi sites for each dopant). All show BFGS/convergence and `JOB
DONE`; eight have directly extracted final energies. The current audit has not
yet established force-threshold acceptance in normalized eV/Angstrom or
extracted all final structures, so these are endpoint candidates, not accepted
reference endpoints. They are high-value for the generalization portfolio and
should be parsed before any MACE comparison.

Input-level identity is now verified for the nine force-pass candidates:
each is a periodic 61-atom `X1Sb24Te36` cell (`X=Cr,Mn,Mo`), with one dopant,
24 Sb atoms, 36 Te atoms, `ibrav=0`, and explicit `CELL_PARAMETERS`. This
removes the system-label ambiguity for the first processed MACE endpoint
release; final-coordinate identity is still a separate gate.

## Current scientific value

This portfolio is valuable as a V@Sb2Te3 generalization candidate and as a
path pool for a controlled MACE follow-up. It is not yet evidence that V@Sb2Te3
has a particular barrier. The next minimum evidence bundle for any accepted
path is: input/output hash, image count and atom identity, PBC/cell, method
metadata, explicit NEB convergence criterion, finite energies/forces, and
endpoint identity.

## Manuscript mapping

Until acceptance, these rows belong in an internal audit supplement only. If
one or more paths pass, they can populate the planned independent-validation
table and a supplementary DFT-vs-MACE barrier figure for A4/generalizability;
they should not replace the Cr@Sb2Te3 A1 reference unless the calculation
protocol and convergence evidence are method-matched.
