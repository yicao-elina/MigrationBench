# NEB Path Topology Audit

All endpoint stability labels remain unverified because the inspected NEB endpoints are frozen. Internal minima are candidates for standalone endpoint relaxation, not accepted minima.

| Path | QE forward (eV) | Hidden basin candidates | Topology | Max error (eV/A) |
|---|---:|---:|---|---:|
| 1-2_historical | 2.790171 | 1 | hidden_basin_candidate;endpoint_dominated_segment | 3.212886 |
| 1-3_historical | 4.003362 | 0 | single_segment_candidate | 5.364093 |
| 1-4_historical | 0.33605 | 1 | hidden_basin_candidate;endpoint_dominated_segment | 0.482407 |
| 1-5_historical | 1.099583 | 1 | hidden_basin_candidate | 4.673162 |
| 1-6_historical | 1.026035 | 1 | hidden_basin_candidate;endpoint_dominated_segment | 0.584852 |
| 1-7_historical | 0.582586 | 1 | hidden_basin_candidate;endpoint_dominated_segment | 0.291742 |
| 1-8_historical | 1.75562 | 1 | hidden_basin_candidate | 0.327388 |
| 81_neb_1 | 1.034728 | 1 | hidden_basin_candidate;endpoint_dominated_segment | 0.355916 |
| 81_neb_2 | 1.561262 | 0 | single_segment_candidate | 0.63572 |
| 81_neb_3 | 1.691053 | 0 | single_segment_candidate | 0.231831 |
| 81_neb_4 | 2.063858 | 0 | single_segment_candidate | 0.526984 |
| 81_neb_5 | 1.414868 | 0 | single_segment_candidate | 0.319104 |
| 1-6_qe_r3 | 0.8204 | 1 | hidden_basin_candidate;endpoint_dominated_segment | 2.571356 |
| 1-7_qe_r3 | 0.557933 | 1 | hidden_basin_candidate;endpoint_dominated_segment | 1.583026 |

Segment barriers are diagnostic until endpoint relaxations and converged QE NEB calculations pass their gates.
