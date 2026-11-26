# Local MigrationBench workspace reconciliation — 2026-11-26

## Scope and disposition

This read-only comparison covers the separate local legacy MigrationBench workspace, canonical MigrationBench main at `296ef562a5bc28151a824d9201c383766cfa116f`, and only the authorized Scratch16 user tree `/scratch16/pclancy3/yi`. It does not inspect other users' `/data` or `/scratch4` contents. No files were moved, deleted, committed, or pushed, and no large raw payloads were hashed.

The local tree is a standalone stale `master` checkout at `ffa168c76f133a7a4d817bcd851bc8a8e6cfeccd`, with no configured remote or merge base to its stale `github/main` ref. Its committed tree has 1,987 files versus 15,571 in canonical main. It contains 20 modified tracked paths, 12,719 untracked regular files totaling 17,170,974,430 logical bytes, seven untracked symlinks, and 15 ignored status entries. These are inventory counts, not unique physical storage or quota savings. It is not a clean branch suitable for direct merge or retirement.

## Six priority evidence groups

| Group | Local workspace | Canonical repo / Scratch16 comparison | Disposition |
|---|---:|---|---|
| `analysis/canonical_model_matrix_20261016/` | 70 files; 1,136,004 B | Scratch16 direct campaign root: 147 files; 134,068,443 B. Four common relative paths (two same-size, two different-size); 66 local-only files (1,119,837 B), 143 remote-only files (134,052,856 B). The two compared submit scripts with different sizes also have different SHA-256. | Local-only metadata/scripts are PR candidates after a content review. Preserve the audit status: 21 registered models, 9 gate-pass, 12 held; the 639-candidate screen remains `HELD_PENDING_MODEL_ACCEPTANCE`, not a complete or accepted matrix. Do not move model/run payloads based on this comparison. |
| `analysis/cr_interlayer_sliding_grouped_20261025/` | 374 files; 11,948,126 B | Canonical repo subtree: 344 files; 6,346,660 B. All 344 canonical relative paths exist locally; 253 are same-size and 91 differ in size. Local-only: 30 files; 5,584,729 B. Direct Scratch16 campaign roots have substantial raw data whose local copies match path/size subsets, but size is not byte identity. Selected compact files both match and differ by SHA. | Preserve repo and archive snapshots as distinct versions until each mismatch is reconciled. R01 embedded raw copy differs from its direct-root counterpart. DFT R05 remains `COMPLETED_BUT_SCF_WARNING`; fixed-path diagnostics and MLIP NEB proxies do not become a validated DFT MEP. |
| `analysis/task3_endpoint_only_neb/` | 3,354 files plus one symlink; 4,874,023,221 B | Canonical repo subtree: 607 files plus one symlink; 24,731,326 B. 603 common paths (563 same-size, 40 different-size); 2,752 local-only files (4,849,298,727 B); three remote-only files (10,053 B). Several local campaign trees map to much larger Scratch16 campaign roots; one mic-linear trajectory differs substantially in size. | Candidate mapping to the campaign archive, not proof of duplicate content. Validate compact manifests and exact payload hashes only through an approved, bounded workflow before consolidation. Preserve preflight status: zero of ten DFT gate successes; no formal MACE NEB result established. |
| `analysis/shap_full_campaign_20261101/` | 78 files; 19,180,954 B | Canonical repo subtree: 89 files; 16,932,552 B. 60 common paths (39 same-size, 21 different-size); 18 local-only (5,370,037 B); 29 remote-only (2,645,163 B). README and `analysis_manifest.json` match. All 12 local per-frame CSV hashes match the manifest pins; all 12 Scratch16 CSV hashes differ from those pins. Registry, quality-audit, and figure manifests also differ. | Hold as a manifest contradiction. Byte identity is unresolved between the expected source export and Scratch16 counterpart; semantic row equivalence has not been established. Local and Scratch16 figures each match their own figure manifest; three PDFs match across trees, while PNG/SVG and manifest hashes differ. Do not select an authority or reinterpret the 12-family × seed analysis. |
| `paper_evidence/` | 7 files; 18,158 B | Scratch16 canonical repo path: seven files, same sizes; all seven SHA-256 values match. | Exact duplicate at the checked file level. The canonical copy is sufficient for navigation; this does not authorize deleting the containing local workspace while other groups remain unmapped. |
| X-FORCE deletion receipt | 1 file; 759 B | Exact Scratch16 archive receipt at `/scratch16/pclancy3/yi/X-FORCE_data_archive/revision1_dualx_legacy_worktree_evidence_20261125/XFORCE_OLD_RESULTS_DELETION_RECEIPT_20261125.json`; SHA-256 matches: `5e320cd2514206bd64217c94ddd86c9aa1e9126d30a5de4a0e213087c7069709`. | Keep with X-FORCE archive evidence; do not duplicate into MigrationBench. It records 375 deleted files / 1,988,089,369 logical bytes and four archived unique payloads. Its temporary pre-delete pointer is not durable. |

Counts use regular-file path inventory, with symlinks stated separately. “Common” means matching relative path, not necessarily matching content. Hash comparison was limited to compact manifests, scripts, and derived evidence; no large raw payload was hashed.

## Campaign path map and cross-reference risks

### Cr interlayer sliding and Lmax1

- The R01 local raw tree contains 90/90 path-and-size matches against its direct Scratch16 root. The root has 228 files / 177,178,375 B. The repository-embedded R01 collector TSV differs by SHA from the direct-root collector, so both are separate snapshots until resolved.
- R02 has 15/15 local path-and-size matches to its direct root (325 files / 843,645,865 B); R03 retry has 11/11 (69 files / 1,891,618,781 B); Scratch Lmax1 G4 R02 has 102/102 (226 files / 137,569,273 B). These are path/size comparisons, not content proofs. The R02 and R03 collector TSVs do match the respective direct-root SHA values.
- The local R05 profile and PDF agree with their local manifest. The manifest's expected PDF SHA is `77c07914…f6f6a8`; the present same-name campaign PDF is 77,385 B and SHA `437790f1a21fa77e42fe8fdbe1b99feca55a45a6be66f70afca204a2c5abaa8c`. The expected artifact was referenced through a missing author-workstation download location and was not found at the checked Scratch16 campaign/repository candidates. Relative links and rebuild scripts that point to workstation or `/tmp` paths will need repair before relocation.
- The unresolved PDF choice changes figure provenance. Preserve the current profile/PDF and both identities; author direction is needed only if recovery/reconstruction leaves competing scientifically plausible outputs.

### Endpoint-only NEB

- The local IDPP tree (825 files / 2,235,091,022 B) maps by relative path and size to the Scratch16 `runs/idpp` subtree, which has 60 additional files. The mic-linear tree has 821 same-size matches and one trajectory with a size difference (14,155,776 B local versus 68,001,016 B remote). The fixed and smoke subtrees are exact at checked path/size counts; derived and release trees have partial overlap.
- The seed-matrix local tree (15 files / 79,365 B) has only seven common names with `/scratch16/pclancy3/yi/mace_neb_cr_seed_matrix_20261016` (987 files / 2,110,610,236 B). Retry2 (18 files / 194,246 B) sits alongside a much larger Scratch16 retry root (1,436 files / 4,483,892,046 B).
- README, endpoint provenance JSON, and remote SHA ledger matched the canonical copies. These compact checks do not resolve every local-only run path. Some input references point to collaborator-owned source locations outside the three scanned user roots; public maps should retain a redacted ownership note rather than embed collaborator-home paths.

### Model weights and matrix metadata

Four Task 2 weight files are local-only candidates, totaling 113,147,003 B: FT-600K (13,075,227 B), FT_Multi_T (12,984,205 B), Foundation (79,471,284 B), and Scratch_Multi_T (7,616,287 B). Their manifest references exist, but exact checkpoint identity and Scratch16 destination mapping have not been proven in this reconciliation. Keep weights out of Git; map manifests, model role, training provenance, and consuming campaigns before any archive consolidation. Their presence does not change existing health-gate or scientific-result status.

The canonical-model-matrix audit describes 21 registered models (9 gate-pass, 12 held) and a 639-candidate screen still pending model acceptance. The submitted Figure 5 15×8 slice and MB15 model-status snapshots remain separate grains; they do not fill the complete MigrationBench model-by-task matrix. Do not advertise this local package as a complete evaluated matrix.

## What needs an author decision, and what does not

### Decisions that can change scientific authority or retention

1. If the R05 PDF cannot be recovered, choose between the manifest-pinned expected source, the present same-name candidate, or a reconstructed export after reviewing its recipe and source data.
2. Decide retention only after a verified file-to-run map and archive check are available. No raw trajectories, models, or checkpoints should be removed because of path/size overlap alone.
3. If moving a collaborator-owned source becomes necessary, the relevant owner must direct its archival location and retention.

### Routine work that can proceed without a scientific choice

- Reconcile the 12 SHAP CSVs with newline-normalized and row-level comparisons; update an additive source/version map while keeping the manifest mismatch explicit.
- Build a per-file map for the remaining local-only paths, repair relative links and scripts, and add small-file integrity records.
- Map the four model weights to exact manifest/checkpoint records and campaign consumers; do not add model payloads to Git.
- Map endpoint-only NEB compact evidence to Scratch16 campaign trees, preserving separate snapshots where sizes or hashes disagree.
- Prepare exact archive/retention manifests before proposing any cleanup. This work can identify candidates but does not itself authorize deletion.

## Safe-retirement conclusion

The local workspace is **not safe to retire yet**. `paper_evidence/` and several compact files are exact duplicates, but the workspace also contains large local-only endpoint-NEB payloads, conflicting SHAP exports, different sliding/NEB snapshots, and unmapped model weights. A later retirement review should require (1) a complete source-path inventory including ignored and symlinked content, (2) explicit destination for every unique item, (3) content-level verification for each payload that is proposed as duplicate, (4) repaired durable path references, and (5) a separate, explicit retention decision for deletion candidates. This report meets none of the deletion gates by itself.
