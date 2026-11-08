# Campaign status: 10-model interlayer sliding

Updated: 2026-10-25 UTC

## Final progress

The complete canonical matrix is collected:

| scope | expected | accepted | status |
|---|---:|---:|---|
| models | 10 | 10 | PASS |
| systems | 2 (`bulk`, `cr_in_gap`) | 2 | PASS |
| fixed-path rows | 20 | 20 | PASS — finite diagnostic |
| self-relaxed rows | 20 | 20 | PASS — endpoint-only MLIP NEB converged |
| matrix rows | 40 | 40 | PASS |
| image profiles | 400 | 400 | PASS |
| QE reference rows | 20 | 20 | PASS — reference only, SCF warning retained |

The final data review and the independent figure-render/style review both
returned `PASS`. The fixed-path figure set contains standalone bulk and
Cr-in-gap panels in PDF/SVG/PNG plus the two-panel composite in PDF/SVG/PNG.

## Submission history and boundary

- Model smoke/preflight: job `32465646`, passed for MP-0b3, MH-0 with the
  explicit `omat_pbe` head, and MatPES-PBE OMAT-FT.
- Fixed-path array: job `32465652`, 20/20 accepted rows.
- The first self-relaxed array (`32465653`) hit a runner staging error because
  the iteration-history helper was an empty staging placeholder. Its 20
  `FAILURE_MODE` attempts are retained in `results/tables/ATTEMPT_LEDGER.tsv`
  and are excluded from the primary matrix.
- Repaired self-relaxed arrays: jobs `32465712` and `32465724`, 20/20 accepted
  repair rows after the helper was restored and checked.

The failed first attempts are operational staging failures, not scientific
failures and not silently merged into the accepted rows.

## Scientific interpretation

`FIXED_PATH_SINGLE_POINT` and `SELF_RELAXED_ENDPOINT_ONLY_NEB` remain separate
observables. Fixed-path barriers are finite MLIP energies evaluated on the
retained QE b-path2 geometries; the QE curve is shown only as a retained
`COMPLETED_BUT_SCF_WARNING` reference. Self-relaxed values are endpoint-only
MLIP PES path proxies and must not be described as converged DFT MEP barriers.

## Review artifacts

- `manifests/POST_RUN_DATA_REVIEW.json`: data/provenance review, `PASS`
- `manifests/POST_RUN_VISUAL_REVIEW.json`: render/style review, `PASS`
- `manifests/COLLECTION_MANIFEST.json`: collection counts and table hashes
- `results/tables/SLIDING_MATRIX.tsv`: compact 10 × 2 matrix with both methods
- `results/tables/ATTEMPT_LEDGER.tsv`: accepted and preserved failed attempts
- `results/figures/fixed_path_interlayer_sliding_10model_2system.png`: composite

