# Sampling and representativeness

## Source structure

The 639 candidates contain 639 unique source snapshots from three source conditions and nine trajectory/seed branches:

| source | candidates |
|---|---:|
| `original` | 216 |
| `reproduce-1` | 216 |
| `reproduce-2` | 207 |

The run-key counts are 72, 72, 72, 72, 72, 72, 72, 64 and 71. Therefore these structures are not all sampled from one trajectory.

## Stage balance

| candidate group | window | n |
|---|---|---:|
| `stable_early_control` | early stable | 207 |
| `stable_pre_onset` | pre-onset | 216 |
| `precursor_near_onset` | precursor | 208 |
| `threshold_or_terminal` | threshold | 7 |
| `terminal_stable_target` | threshold window / 1000 ps target | 1 |

This is representative of the audited failure-attribution frame across the available trajectories and training seeds. It is intentionally enriched around onset and threshold behavior; it is not an unbiased equilibrium distribution or a time-weighted estimate of the trajectory population.

## Source table

`../analysis/candidate_source_coverage.csv` is the canonical source/run coverage table. The preflight tables under `../preflight_distribution/` retain the same counts before MLIP evaluation.
