# Manuscript source authority audit — R3

**Captured:** 2026-11-26 UTC
**Scope:** GitHub MigrationBench submitted Revision 1 source and the preserved GitHub PR #10 candidate snapshot. The live Overleaf server tip was not captured in this audit.

## Identified source versions

| Version | Status | Git identity | Manuscript files |
|---|---|---|---|
| Submitted Revision 1 | `SUBMITTED_RELEASE_CANONICAL` | source sync commit `334260cdda3869d55999081cd610b34bfba53257`; release record identifies these files as the submitted source synced from the 2026-11-09 Overleaf export | Main `paper/revision1/sn-article-final.tex`, SI `paper/revision1/sn-article-SI.tex`, source ledger `paper/revision1/data_processed/manuscript_source_ledger.csv` |
| PR #10 manuscript candidate | `CLOSED_UNMERGED_CANDIDATE` | head `51cb06f71732e46cd10eb92cee1f0c5f8f4833c9`; PR #10 closed with no merge commit | Byte-preserved copy under `paper/revision1_candidates/pr10_overleaf_candidate_20261123/source/paper/revision1/` |
| Current live Overleaf tip | `LIVE_OVERLEAF_NOT_VERIFIED` | no commit or source export pinned | Not captured; no hash or equality claim is made |

The candidate README calls the PR #10 files comparison-only and says they were not compiled as a standalone tree. A closed, unmerged candidate commit is not evidence of the current Overleaf server source.

## Full PR #10 integration reconciliation

PR #10 changed 47 repository paths. Although GitHub reports it closed with no merge commit, its head `51cb06f71732e46cd10eb92cee1f0c5f8f4833c9` is an ancestor of current main `19708d74d14103832e4f86b51b8b5d8875175cc3`. Comparing each PR-head blob with current main finds:

| Reconciliation | Files | Meaning |
|---|---:|---|
| `SAME_PATH_IDENTICAL` | 44 | PR #10 file bytes remain at the same path on current main. |
| `PRESERVED_CANDIDATE_IDENTICAL` | 3 | Main text, SI and source ledger changed at their submitted paths; exact PR #10 versions remain at the candidate snapshot paths. |
| Unresolved | 0 | No PR #10 changed file is missing from both current and preserved candidate paths. |

The SQL table `pr_file_reconciliation` records source and current blob IDs, byte counts, SHA-256 values, paths and status; query it with QCE09. Thus the 639-candidate experiment data, scripts and other PR #10 artifacts already have a current-main path and do not need another merge. The three manuscript files remain distinct versions. This establishes repository integration only, not scientific validation or manuscript authority.

PR #11 is also closed without a merge commit, but head `476ca161ac6235965f8062b83484d580c846fed7` is an ancestor of current main. Its 639×12 package is present under `analysis/multimodel_639_screen_20261014/`; this does not make those 12 rows interchangeable seeds or elevate them into submitted manuscript claims.

The PR #10 campaign capsule records 639 frames from nine same-seed MD runs and 1,917 successful static evaluations over three registered checkpoints. Eight MD runs failed and one completed 1 ns; the diagnostic manifest describes descriptor distance as novelty diagnostics, not calibrated uncertainty or causality. Rockfish read-only SHA checks confirmed all three checkpoint files against the hashes in the model registry. The 2,050-atom QE attempt remained resource-gated; the 120-atom local-shell input is a reconstruction marked prepared-only/submitted-pending. These data support a measured screening/failure-attribution package, not a DFT-validated mechanism or demonstrated active-learning gain.

For portability, the MLIP result manifest's model-registry reference was changed from a local macOS absolute path to the tracked repository-relative file `analysis/from_scratch_failure_attribution_20261011/mlip_candidate_campaign_20261013/models_rockfish.tsv`; the file hash remains unchanged. Raw model checkpoints and trajectories stay external and are not implied to be stored in Git.

## Exact file identities

| Source | File | Bytes | SHA-256 |
|---|---|---:|---|
| Submitted | `sn-article-final.tex` | 66,938 | `a2252edcf786a4ce575eb6afaa018b1b8aad5b8266874934b3da5fd95566416f` |
| Submitted | `sn-article-SI.tex` | 46,875 | `2e65eb40ca03c0b90332ae64998447f453e33acdfeabfe51be8dcc66b8226ab5` |
| Submitted | `manuscript_source_ledger.csv` | 6,296 | `ae698692a5226fb60a05e5033ebb74be32d7f81dd3bfbbfb04ec6a0aa3c1d256` |
| PR #10 candidate | `sn-article-final.tex` | 66,938 | `20b39b6cee13207bc612d54d29e98ec785e11c3ee713413c42118a00c3a748cf` |
| PR #10 candidate | `sn-article-SI.tex` | 46,875 | `c0ee7f9fc82cdfba22cb7e6b98ff03eda4d08fa037f0fff67d4ceeb715d77680` |
| PR #10 candidate | `manuscript_source_ledger.csv` | 6,946 | `1146541ae767df1d31c145f2c506f98ce1825e94d77c1865ca8b6650efc210dc` |

The submitted PDF release record pins `paper/revision1` as the submitted source. The PR #10 source snapshot has a substantially changed main-text and SI SHA; these are distinct manuscript versions even though their paths share the same logical roles.

## Material differences found

- The submitted main text has seven figures and the candidate has six, with different figure roles and labels. The submitted abstract describes selected process-dependent migration gains, calibration errors and limits on mechanistic inference; the candidate abstract asserts equilibrium reproduction and reframes specialist failures and long-range mechanics.
- The submitted SI contains five figures and ten tables; the candidate SI contains eight figures and four tables, adding transport, failure-attribution, DFT NEB profile, self-NEB, latent-robustness and SHAP-CV material. The candidate was not compiled standalone in the checked source snapshot, and its cited outputs were not revalidated by this comparison.
- The submitted source ledger has 20 rows; the candidate ledger has 22. The row-level delta is queryable in R3: eight candidate-only IDs, six submitted-only IDs, and four altered common IDs.
- In particular, the submitted ledger marks `NEB-SELF-FOUNDATION` `EXCLUDE_FROM_PUBLICATION` because it is a different system/path with calculator and serialization defects. The candidate marks it `AUDITED-LEGACY` and cites the historical figure PDF. Candidate main-text lines 472–475 report a 0.41 eV self-optimized barrier and speculate about unphysical separation/catastrophic forgetting. That text is not a validation of the underlying record.
- Three other shared ledger rows have source-detail name-spacing differences for SHAP model identifiers. Preserve those exact strings; do not infer that the versions are equivalent from the typography alone.

The submitted and candidate manuscript files remain separate. No scientific wording, figure, or status was promoted from PR #10 to the submitted source by this audit.

## Source-of-record decision

The submitted release remains the pinned historical publication source. The current live Overleaf tip is still not captured, so this audit does not identify a new future manuscript authority. Keep the three-file PR #10 manuscript snapshot separate from the submitted release pending that source capture and the scientific/evidence reconciliation. The other 44 PR #10 files have already been reconciled to current main.
