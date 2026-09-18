# QE NEB Barrier Atlas

Generated: `2026-09-13T13:30:39.929782+00:00` from locally synced outputs.

> No row currently passes the unified final-reference gate. Values below are candidate energy spans, not manuscript-ready migration barriers. `endpoint_dominated` means the maximum is a frozen endpoint rather than an interior saddle image.

| Path | QE forward candidate (eV) | QE reverse candidate (eV) | Endpoint delta (eV) | Peak | Max movable error (eV/A) | Status |
|---|---:|---:|---:|---:|---:|---|
| `1-2_historical` | 2.790171 | 1.015330 | 1.774841 | 4 (interior_peak_candidate) | 3.212886 | `historical_candidate_quarantine` |
| `1-3_historical` | 4.003362 | 3.950126 | 0.053236 | 3 (interior_peak_candidate) | 5.364093 | `historical_candidate_quarantine` |
| `1-4_historical` | 0.336050 | 0.016829 | 0.319221 | 3 (interior_peak_candidate) | 0.482407 | `historical_candidate_quarantine` |
| `1-5_historical` | 1.099583 | 0.753468 | 0.346115 | 2 (interior_peak_candidate) | 4.673162 | `historical_candidate_quarantine` |
| `1-6_historical` | 1.026035 | 0.205627 | 0.820408 | 3 (interior_peak_candidate) | 0.584852 | `historical_candidate_quarantine` |
| `1-7_historical` | 0.582586 | 0.024650 | 0.557936 | 3 (interior_peak_candidate) | 0.291742 | `historical_candidate_quarantine` |
| `1-8_historical` | 1.755620 | 0.989243 | 0.766396 | 2 (interior_peak_candidate) | 0.327388 | `historical_candidate_quarantine` |
| `81_neb_1` | 1.034728 | 1.030269 | 0.004460 | 5 (interior_peak_candidate) | 0.355916 | `historical_candidate_quarantine` |
| `81_neb_2` | 1.561262 | 2.111153 | -0.549890 | 6 (interior_peak_candidate) | 0.635720 | `historical_candidate_quarantine` |
| `81_neb_3` | 1.691053 | 1.668197 | 0.022856 | 6 (interior_peak_candidate) | 0.231831 | `historical_candidate_quarantine` |
| `81_neb_4` | 2.063858 | 1.543546 | 0.520312 | 6 (interior_peak_candidate) | 0.526984 | `historical_candidate_quarantine` |
| `81_neb_5` | 1.414868 | 1.065830 | 0.349037 | 7 (interior_peak_candidate) | 0.319104 | `historical_candidate_quarantine` |
| `1-6_qe_r3` | 0.820400 | -0.000000 | 0.820400 | 5 (endpoint_dominated) | 0.202478 | `running_unconverged_synced_snapshot` |
| `1-7_qe_r3` | 0.557933 | 0.000000 | 0.557933 | 5 (endpoint_dominated) | 0.234097 | `running_unconverged_synced_snapshot` |

## Interpretation

- `accepted_converged` requires the parser's force and barrier-drift gates; there are currently zero accepted rows.
- `historical_candidate_quarantine` preserves old numerical evidence but does not promote it to a final reference.
- Current r3 rows are the latest locally synced snapshots, not a fresh scheduler poll.
- The table reports QE's interpolated forward/reverse activation energies from the last complete iteration.
- The JSON/CSV also retain discrete `max(E)-E_initial` and `max(E)-E_final` values; those actual image energies are used to draw the curves.
