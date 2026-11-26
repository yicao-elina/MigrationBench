# Revision 1 NEB/DFT evidence audit — 2026-11-26

**Scope:** read-only; only the user's Scratch16 tree `/scratch16/pclancy3/yi` and the canonical local MigrationBench checkout at commit `cf6c2b1649ba993d079d0eadf8478775a81736d2`. No other-owner directories were searched. No scientific files or SQLite databases were changed.

## Key correction: legacy Table S5 outputs are present and hashed

The earlier 2026-11-25 crosswalk correctly said six legacy 61-atom `neb.out` files were not in the canonical Git checkout, but it predates the R2 SQL import and scoped Scratch16 search. All six bytes are present under the personal Scratch16 path `revision1_migrationbench_pipeline/data_raw/historical_neb/legacy_61/{1-2,1-3,1-5,1-6,1-7,1-8}/neb.out`; fresh SHA-256 checks on 2026-11-26 match the R4 SQLite artifact rows. The R4 snapshot links each of those six raw-path artifacts to its forward and reverse Table S5 result (12 `PARTIAL` result-artifact links). The original audit had no expected historical SHA, so the present-day identity is known while exact producer/release identity remains partial. This does not itself prove that the current file generated the submitted table values.

The five 81-atom outputs C1–C5 and the 61-atom G1 output are tracked in Git; their local SHA-256 checks match the R4 DB. `TABLE_S5_RAW_OUTPUT_RECHECK.tsv` lists all 12 distinct output files, table cells, current SHA/size, existing status, and remaining gap.

## Evidence families and boundaries

- **QE Table S5:** 24 forward/reverse cells link to the submitted TeX, processed CSV and path registry; raw artifacts are addressable. G1 and legacy G2/Q1/Q2/Q3/M1 retain `UNCONVERGED` status. L1 is labelled `CONVERGED` in the legacy source, but result provenance is still partial. C1–C5 carry SCF warnings; the existing audit reports force residuals of 0.38–3.2 eV/Å against a 0.03 eV/Å gate. The submitted TeX omits the per-path status column. Reproduced rounding is not a convergence result.
- **Figure 4:** the existing crosswalk has 30 panel/group rows (six panels × five groups) and hash-addressable profile/figure assets. It lacks the complete per-curve → checkpoint/path → run/attempt/job → raw output chain. Fig. 4 is tied to R05 while the submitted Table S3 reproduces R02.
- **Figure S2:** registry/reproduction evidence exists, but the caption is marked `FULL_CAPTION_MISMATCH`: the caption describes legacy G1 / the superseded 0.16 eV baseline; the registered plotted panel is the canonical in-gap grouped comparison. Keep this issue open for author review.
- **Fixed/self/MIC-linear matrix:** Scratch16 campaign `neb_fixed_vs_self_matrix_20261019` has 57 fixed-path records, 57 endpoint-only IDPP self-relaxed records, and 57 endpoint-only MIC-linear sensitivity records; jobs are 32212739, 32212740, 32216474. R01 release QA passes its identity/hash/coverage checks. Outcomes are 51 fixed `FINITE_DIAGNOSTIC` + 6 failures; IDPP 49 converged + 3 failures + 5 nonconverged; MIC-linear 44 converged + 3 failures + 10 nonconverged. These are distinct observables; the canonical QE reference remains `COMPLETED_BUT_SCF_WARNING`. The separate six-trajectory/144-record main A1 target is unresolved and excluded from R01. The campaign has not yet been imported as a complete run-level matrix in the central lineage SQL.
- **QE restarts:** 20261018 and 20261019 manifests each describe five 81-atom path snapshots, last complete iterations 53–61, and distinct patched-input/restart SHA values. The manifests state that the exact original QE checkpoint is unavailable. The 20261022 repair manifest links parent jobs 32211491–32211495 and records reuse of saved `OUT` state. Keep each source, partial checkpoint, patched input, restart attempt, and output separate; do not infer DFT acceptance from job completion.

## Highest-priority next work

1. Add exact QE input/settings, parent job/attempt, per-image forces, SCF markers, convergence outcome, and processing/parser hash to each Table S5 cell/run chain. Preserve the current outputs and statuses.
2. Reconcile current Table S5 raw bytes with their original producer manifests/checksums if available; keep `PARTIAL` until the expected historical identity is evidenced.
3. Join Fig. 4 R05 profile rows to each curve's exact model/checkpoint hash, path hash, attempt, job, and raw artifact. Keep the submitted R02 Table S3 release separate until the author chooses the paper authority.
4. Import the 171 fixed/IDPP/MIC-linear campaign records as campaign-level evidence with explicit observable/initializer fields. Do not use it to fill the unresolved six-trajectory A1 figure without exact path identities.
5. Preserve the Figure S2 caption/asset mismatch and ask the author which quantity/description is intended. Do not edit manuscript science text automatically.

## SQL

`queries.sql` has read-only QL25–QL26 lookups for current Table S5 raw-output links and status counts against the R4 snapshot. Snapshot path:

`/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_lineage_overlays/20261125_r4/migrationbench_r1_lineage_overlay_20261125_r4.sqlite`
