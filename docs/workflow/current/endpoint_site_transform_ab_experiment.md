# Endpoint Site Transform A/B Experiment

## Question

Does a deterministic, physics-motivated displacement of Cr toward a larger local void reduce the DFT work needed to relax a selected NEB image, relative to relaxing the identical image directly?

This experiment tests initialization efficiency. It does not train a site-stability model and does not establish a new stable endpoint until QE relaxation passes the force and basin gates.

## Paired Design

Each pair uses the same host geometry, QE template, pseudopotentials, cutoffs, k points, convergence thresholds, CPU resources, walltime, and seed. The only changed variable is the initial Cr coordinate.

The exact sources are the latest complete QE path iterations available when the experiment was created:

- `1-6`: `sb2te3.path108`
- `1-7`: `sb2te3.path112`

Images 2 and 5 were selected because the topology audit identifies them as internal or terminal basin candidates. Every source file, generated input, parameter set, and submitted job is SHA-256 recorded in the paired manifests under `data_processed/endpoint_relax_inputs/exact_iter_pairs/`.

## Mathematical Transform

For the original Cr coordinate `r0`, the transformed coordinate is the constrained minimizer

`r* = argmin_(||MIC(r-r0)|| <= delta_max) [lambda ||MIC(r-r0)||^2 + sum_j softplus((d_safe-d_j(r))/sigma)^2]`.

Here `d_j(r)` is the periodic minimum-image distance between Cr and host atom `j`. The first term prevents a large, untraceable displacement; the second smoothly penalizes short Cr-host contacts. Parameters are `d_safe=2.80 A`, `sigma=0.25 A`, `lambda=2`, and `delta_max=0.25 A`.

This is a local geometric prior, not a DFT energy model. A lower geometric objective is not assumed to imply a lower DFT energy.

## Acceptance And Comparison

A formal speedup is reported only if both pair members:

1. complete QE BFGS relaxation;
2. have final maximum atomic force at or below `0.05 eV/A`;
3. converge to the same basin, checked by periodic Cr displacement, host RMSD, and Cr local-distance signature;
4. remain comparable under identical electronic and resource settings.
5. carry valid input/job/`pw.x` runtime provenance for every segment included
   in cumulative ionic-step and SCF counts.

For a manuscript-level acceleration claim, the shared electronic settings must
also pass the independent QE calculator-identity gate. Until then, a matched
pair can diagnose optimizer behavior but cannot establish acceleration under the
final production Hamiltonian. If the accepted QE identity changes, both members
of every pair will be rerun from their recorded initial structures.

Primary cost metrics are ionic steps and total observed SCF iterations. Walltime is secondary because queue placement and node variability can bias it. A different final basin is reported as a mechanism change, never as acceleration.

Because the current parent segments predate runtime capture, they remain
diagnostic even if later continuations converge. A production speed comparison
must rerun both arms with complete provenance from segment one. Rockfish smoke
`30857352` validates this fail-closed rule.

## Current Prefix Result

Latest synchronized Rockfish snapshot: `2026-09-13T10:50Z`, after about 3.5
hours of CPU execution. All eight jobs are healthy and still running.

These values use the latest synchronized output and are not final speedups:

| Path/image | Shift (A) | Initial DFT energy penalty, transformed-direct (eV) | Ionic steps direct/transformed | SCF iterations direct/transformed | Current max force direct/transformed (eV/A) |
|---|---:|---:|---:|---:|---:|
| `1-6/2` | 0.250 | +0.184 | 22 / 27 | 470 / 504 | 0.084 / 0.096 |
| `1-6/5` | 0.143 | +0.177 | 28 / 29 | 528 / 544 | 1.441 / 0.072 |
| `1-7/2` | 0.250 | +0.179 | 28 / 29 | 520 / 523 | 0.078 / 0.077 |
| `1-7/5` | 0.132 | +0.166 | 30 / 28 | 511 / 529 | 1.883 / 1.469 |

The updated prefix no longer supports a general early-step advantage. The two
image-2 pairs are effectively tied in force and SCF work, while the transformed
`1-6/5` branch currently has a much smaller force than its direct control. That
single favorable force snapshot is not a speedup: the current geometries fail
the provisional same-basin threshold, and neither branch has converged. The
defensible result for every pair remains `pending_not_converged`.

The calculator-identity gate is also pending. Consequently, even a later
same-basin prefix advantage under the historical 50/200 Ry Gamma non-spin setup
will remain protocol-specific until the cutoff, k-point, spin, and SOC branches
in `docs/qe_methods_resolution_branches.md` are resolved.

The comparator now separates `diagnostic_current_geometry_equivalence` from
final `basin_equivalence`. All four final basin labels are currently `pending`.
The running-prefix geometry is currently different for `1-6/5` and `1-7/2`
and similar for the other two pairs, but none is a finalized basin switch. The paired input
calculator identities match in all four cases; N24 acceptance remains false.

The live parser reports the force from the last complete force block. If QE has
already printed the next geometry but is still writing its force block, that
trailing partial step is retained but cannot overwrite the last complete force.

## Interpretation And Next Version

The local-void transform successfully increases selected nearest-neighbor distances, but it raises the initial DFT energy for all four tested points. This shows that maximizing local free volume alone is not a sufficient stability coordinate in this bonded layered environment.

The next proposal model should retain the contact penalty as a hard safety filter, then rank candidates using a learned local descriptor and uncertainty term. SOAP or a frozen MACE embedding can provide the invariant representation; kNN, kernel regression, or a Gaussian process can estimate energy/force/stability without claiming that the current small dataset supports a new large neural network. That descriptor-ranked experiment must be a separately versioned A/B batch, not a relabeling of this geometric baseline.

## Artifacts

- Inputs and manifests: `data_processed/endpoint_relax_inputs/exact_iter_pairs/`
- Pair tables and all ionic-step curves: `data_processed/endpoint_relax_comparison/exact_iter/`
- Force plots: `data_processed/endpoint_relax_comparison/exact_iter/*/force_relaxation_curves.svg`
- Generator: `scripts/migrationbench/prepare_qe_path_iteration_relax_pairs.py`
- Comparator: `scripts/migrationbench/compare_endpoint_relaxation_speed.py`
- A formal speedup additionally requires an explicit N24 calculator-acceptance
  JSON whose identity matches both input-derived calculator hashes.
