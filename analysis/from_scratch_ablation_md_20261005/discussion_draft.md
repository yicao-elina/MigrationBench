# Discussion draft — from-scratch ablation MD

This draft is deliberately evidence-bounded. Statements under “confirmed” use
the complete 20 ps audit and the terminal 1 ns failure audit; no formal 1 ns
transport conclusion is reported because the selected trajectories did not pass
the complete-step gate.

## Confirmed from the 20 ps matrix

The formal matrix contains 14 auditable non-group-split from-scratch model
artifacts. One additional nominal entry, correlation=3/seed123, is excluded
because legacy foundation and later scratch records overlap under the same
source directory. The 14 formal tasks have no unexplained missing status:
three reach the requested 20,000 steps and eleven stop at a retained
scientific instability onset.

The stable screening outcomes are baseline/seed345 and lmax=2/seed123,234.
The lmax=2 family therefore provides the ablation contrast while baseline
seed345 provides a scratch-baseline control. This selection is not a claim
that the selected seeds are statistically representative of the full matrix;
the failed seeds remain part of the stability result and are not discarded.

The instability pattern is family-dependent in this short audit. Both
correlation=3 formal seeds fail within 2 ps, all three lmax=0 seeds fail
within 1.6 ps, and two of three baseline seeds fail before 3.4 ps. The
correlation=1 family includes one longer-lived failure at 11.6 ps but still
crosses the force threshold. These observations support a stability/robustness
statement only; they do not identify a unique physical failure mechanism.

All 14 trajectories have finite stress-derived pressure samples and can be
processed through the repository’s simplified stress-based heat-flux
convention. The resulting HFACF and running-κ traces are diagnostic only at
20 ps. The heat-flux expression is retained as an explicit approximation, not
silently treated as a rigorous many-body MLIP heat current.

## Terminal 1 ns failure audit

The three selected 1 ns branches all terminated with the recorded scientific
stop reason `instability_threshold`: baseline/seed345 at step 83,000 (83.0 ps),
lmax=2/seed234 at step 52,400 (52.4 ps), and lmax=2/seed123 at step 795,600
(795.6 ps). All three runs retained finite postprocessed fields and passed the
identity checks for 2,050 atoms, periodic boundary conditions, common input and
species hashes, timestep, target temperature and friction. None reached the
requested 1,000,000 absolute steps, so the campaign is `BLOCKED` for the
formal transport comparison.

The lmax=2/seed123 branch is the longest-lived and reaches a terminal maximum
temperature of 7,733.6 K, maximum force of 180.9 eV/A and minimum distance of
0.476 A. The other branches also show finite but terminally unstable force and
temperature excursions. These values are retained as failure-onset evidence,
not as equilibrium observables. The candidate displacement counts (23, 6 and
3) remain candidate-only because site mapping was not completed.

The postprocessor emitted stress-based heat-flux and running-kappa traces for
all three branches. They are explicitly repository diagnostics and not
converged HFACF/thermal-transport measurements. The lmax=2/seed123 branch has
a diagnostic 200–500 ps Dxy fit, but no Dxy estimate is promoted because the
trajectory is unstable and incomplete; its 500–900 ps fit also overlaps time
past the recorded terminal point. No block/replica uncertainty estimate is
scientifically valid for this incomplete set, and no automatic continuation was
performed because the stop was scientific instability rather than wall time.

The complete terminal inventory, hashes, claim/evidence matrix and gap report
are in `terminal_failure_bundle_20261006/`. The Plot Atlas figure in
`figures_terminal_failure_atlas_20261006/` is diagnostic-only and marked
`BLOCKED`.

## Pending replacement production analysis

Any replacement production comparison must report, for each selected run, the exact
balanced/production window, time sampling, Cr MSD$_{xy}$ and MSD$_{3D}$,
VACF, candidate/site-mapped hops, pressure and thermodynamic statistics,
HFACF, running κ, and block/replica uncertainty. A completed 1 ns trajectory
is not by itself evidence that Dxy or κ has converged.

The paper and old continuation traces remain context-only unless their input,
start step, model hash, stress/heat-flux definition and window match the new
protocol. They are drawn as `REFERENCE_PROTOCOL_DIFFERENT` in the comparison
figure and are not pooled into same-protocol replica statistics.

## Claim boundary for the manuscript

At present the defensible claim is: the from-scratch ablation models show
large differences in short-time numerical stability under the common fresh
start, while all three selected 1 ns branches terminate in scientific
instability before the target duration. The lmax=2/seed123 branch provides the
longest diagnostic record, but it does not establish a formal diffusion or
thermal-conductivity result. Relative Dxy, hop-rate or thermal-transport claims
remain blocked until a validated replacement campaign passes complete-step,
finite-state, geometry, continuity, window, block and replica gates.
