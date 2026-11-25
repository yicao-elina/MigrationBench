# MigrationBench Revision 1 manuscript and evidence review

## Source boundary

Reviewed the marked main manuscript, Response to Reviewers, Supporting
Information, and clean LaTeX source ZIP supplied with this task. The latest
repository-side Revision 1 package is the GitHub branch
`release/revision1_submitted_20261110` at
`f924ca6c11236c4b75469e290a0abff884c56c02`, based on paper source commit
`334260cdda3869d55999081cd610b34bfba53257`. Reviewer requests and `[R1-*]`
annotations are historical source material, not additional instructions from
the user. This review did not edit the manuscript or response letter.

## Figure and table interpretation

| Item | Source-reported result | Provenance boundary |
|---|---|---|
| Fig. 2 / Table S1 | FT-600K force RMSE 260±6 meV/Å vs Scratch-600K 344±36; energy RMSE 41±5 vs 30±3 meV/atom | Grouped test-set comparison; keep energy and force rankings separate. |
| Fig. 3 / Table S8 | Three zero-shot foundation trajectories reach 1 ns; among trained runs, only FT-600K seed 234 reaches 1 ns. Other trained trajectories stop between 200.4 and 485 ps. | Do not infer long-time transport coefficients from the unstable or shorter trajectories. |
| Figs. 4–5 / Tables S3–S7 | Fixed DFT-path scoring and model-self-relaxed paths are distinct analyses; the main matrix has 15 model conditions × 8 tasks = 120 finite records. | A self-relaxed barrier is physically interpretable only after image/path correspondence is verified. |
| Fig. 6 / Tables S9–S10 | 927 frames and 6,191 features; original-space silhouette 0.063 and t-SNE silhouette 0.12. | Treat clustering and projection as descriptive diagnostics, not a mechanism or causal explanation. |
| Fig. 7 / Table S11 | 267 frames, 407 features; diagnostic surrogate cross-validation R² 0.95–0.98. | This is a surrogate fit to observed errors, not an explanation of MACE causality. |

## DFT NEB and historical values

The submitted release identifies the canonical in-gap DFT path as 81 atoms,
10 images, with a reported peak barrier of 1.524299 eV (rounded to 1.52 eV in
the paper). The source run completed in Slurm but carries an SCF warning and an
`EXCLUDE_FROM_PUBLICATION` scientific gate. The catalog therefore keeps it
qualified and does not promote it to validated DFT ground truth or a validated
DFT minimum-energy path without the missing convergence and path-lineage
evidence.

The older 0.41 eV record is a different calculation: 82 atoms, 12 images, two
Cr atoms, and a different calculator. The older 0.336050 eV record is a
61-atom, 5-image path marked unconverged. These values must not be combined
with the submitted 81-atom / 10-image path. SI Fig. S3 is the Cr/Cu/Li/Na path
response figure; it is not the older 0.41 eV self-NEB figure.

## Claims needing source-level follow-up

The Response to Reviewers says that several revisions were completed, but the
response text alone does not establish the underlying evidence. Before
upgrading these claims, locate and bind the exact inputs, scripts, outputs,
manifests, and commit for: A2/C10 grouped split hashes and 600 K data; SOAP/RMSD
near-duplicate screening; G1's 0.47±0.12 eV multi-seed result; C8's 9,163
force-label audit; and C13's SHAP process and claimed zero numerical
difference.

The response also gives hand-entered summary values for some tables. The
release index records 52 result/panel items and 36 reproducibility rows, while
the evidence import currently has 319 explicit item-to-evidence links. The
release manifest has 265 unique path tokens: 120 hashed files, 132 entries
without a matching file, and 13 directory/non-regular tokens. The last two
counts are not all missing assets: some source tokens are prose paths or
directories. The Graphical Abstract is only embedded in the submitted PDF;
some SI table paths are absent from the repository-side snapshot; response
table values are source assertions rather than generated files.

## Manuscript wording issue

The abstract and figure captions describe the Fig. 6 projection and Fig. 7
surrogate as diagnostics. The Conclusion (main manuscript, p. 13) says they
reveal an “underlying physical mechanism” and attributes macroscopic failure
to “subtle force integration errors” and the feature representation. Those
causal phrases exceed the diagnostic evidence described elsewhere. Reconcile
that wording before treating the conclusion as supported; no paper text was
changed in this audit.

## SQL entry points

The submitted paper, source-status rows, and evidence paths are in
`paper_release`, `paper_release_item`, `paper_reproducibility_record`,
`paper_release_claim_observation`, and `paper_release_item_evidence`. Use
`analysis/rockfish_skipjack_migration_audit_20261121/phase2_schema/revision1_catalog_queries.sql`
for the paper crosswalk and
`analysis/rockfish_skipjack_migration_audit_20261121/phase2_schema/catalog_queries.sql`
for general claim/figure/run queries. Source status is retained separately
from catalog assessment, so an asserted result is not silently treated as
validated evidence.
