# Periodic Clearance-Graph Path Reconstruction

## Method

The historical `1-5` trajectory has a `4.200 A` maximum Cr step. Ordinary
linear densification inserts images along that segment but can pass directly
through a host atom. The clearance-graph method instead builds a layered graph
over resampled host configurations.

For each intermediate layer, Cr candidates are sampled on deterministic
spherical shells around the inherited reference position. Candidates with
periodic Cr-host distance below `1.8 A` are removed. Dynamic programming then
minimizes

\[
J = w_r\sum_i\|\mathbf r_i-\mathbf r_i^{ref}\|^2
  + w_s\sum_i\|\mathrm{MIC}(\mathbf r_i-\mathbf r_{i-1})\|^2
\]

subject to a hard `2.0 A` maximum adjacent Cr step. Endpoints and the resampled
host path are fixed. No energy or force labels are created.

## Rockfish Result

CPU Slurm job `30850985`, seed 48, completed all geometry gates:

| Metric | Historical `1-5` | Clearance graph |
|---|---:|---:|
| Images | 5 | 13 |
| Minimum pair distance (A) | 1.4085 | 1.8056 |
| Maximum Cr step (A) | 4.1997 | 1.8304 |
| Total Cr arc length (A) | 13.3708 | 15.7376 |
| Minimum host-host distance (A) | not limiting | 2.7144 |

The longest reference displacement of a selected Cr point is `2.0 A`, so this
is a reconstructed candidate mechanism rather than a small repair.

Geometry-only comparison in Slurm job `30850995` excludes the unavailable
energy-profile component and renormalizes the other four weights. The candidate
is nearest to historical `1-5` with similarity `0.5347`. Adding it as a valid
center raises geometry-only facility coverage from `0.7424` to `0.7726`, a
gain of `0.0302`. This supports retaining it as a distinct `1-5`-region
candidate, not substituting it silently for historical `1-5`.

MACE diagnostic job `30850997` tests whether the reconstructed path remains a
continuous mechanism after MLFF NEB. It cannot enter QE production until its
final mechanism is classified and both endpoints pass independent DFT repeat
relaxation under an N24-accepted calculator identity.
