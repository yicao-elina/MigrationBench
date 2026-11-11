# R02 independent review feedback ledger

All repairs below were additive and preserved raw runs, endpoint manifests, retry tables, and historical nonconverged attempts.

| Round | Reviewer | Finding | Repair / disposition |
|---|---|---|---|
| R01 | Data | Pre-execution/partial campaign; no release decision. | Continued campaign and retained pre-execution report. |
| R02 | Visual | Composite/standalone typography below intended manuscript-scale floor; reference pointer not explicit. | Increased source typography, recorded 180 mm composite and 100 mm standalone placement widths, and bound the archived authoritative reference with SHA-256. |
| R03 | Visual | Cu/Li/Na row labels too close to first-column negative y ticks. | Shifted row annotations from axes x=-0.29 to x=-0.36; fresh review measured approximately 5.97 pt clearance at 180 mm. |
| R04 | Data | Raw index stale, retry roles not propagated, scheduler maps not valid TSV, derived package unsynchronized, empty failure registry lacked schema. | Re-indexed 125 attempts, propagated selected manifest roles/retry fields, normalized five scheduler maps to TSV, synchronized and input-bound derived artifacts, and emitted a header-only zero-row failure registry. |
| R05 | Data | R02A retry reason absent from selected rows/attempt ledger. | Added retry-table lineage fallback in collector; all R02A/B/C retry rows now carry `retry_reason` and `retry_round`; re-collected and re-bound hashes. |
| R06 | User/data scope | User requested the NEB reference curve in addition to the MLIP self-relaxed curves. | Added a versioned 120-point QE/DFT reference layer and frozen R03 data/visual addendum. The layer is explicitly labelled SCF-warning diagnostic and fixed-path, not same-geometry DFT ground truth. |
| R07 | Visual repair | Independent visual review found the Cr gap→QL reference trace crossed the composite panel letter; this was a layout defect, not a data defect. | Moved panel letters outside the axes, archived the failed review/output round, and re-rendered all 39 R03 files. |
| R08 | User legend scope | User requested that the QE row show only `DFT-NEB` and that `converged / included` be removed. | Simplified the legend to the three model families plus `DFT-NEB`; retained the full SCF-warning/fixed-path boundary in caption, manifest, and review gates. |
| R09 | User visual standard | User requested the final NEB styling to match the supplied reference: bold x-axis title, thicker panel borders, and thicker DFT-NEB line; then standardized the x-axis title to `Reaction Coordinate`. | Applied the weighted style to all 39 outputs, recorded the reusable convention in `NEB_FIGURE_STYLE_STANDARD.md`, and archived one materialization-failure round where incomplete class-summary hydration produced `n=?`/`n=0`; the final rerender restored the byte-verified `n=5/2` counts. |

Final acceptance requires both independent reports in `data_review/` and `visual_review/` to begin with `PASS`, plus a release manifest with `release_ready=true`.
