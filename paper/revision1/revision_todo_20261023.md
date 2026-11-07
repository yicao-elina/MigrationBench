# MigrationBench revision TODO and closure record

Date: 2026-10-23  
Scope: `paper/revision1/` only  
Read-only annotation source: `/Users/alina/Downloads/sn-article-final copy.pdf`  
Independent reader review: `/tmp/migrationbench_independent_reader_review_20261023.md`

This record maps every annotation object in the annotated copy to a reader-facing
revision or to an explicit residual item. The annotation PDF contains 17 objects
(visual highlights/underlines and free-text comments). Highlight objects do not
store selected text in their PDF metadata, so their page and rectangle are recorded
alongside the visible paragraph or section they mark. Line numbers below refer to
the source state at the time this record was created; section names and labels are
the stable locator.

Status legend: **Done** = implemented and checked in the target source/PDF;
**Accepted residual** = not silently hidden, with a reason and a bounded follow-up;
**Open** = still requires an in-scope edit before submission.

## Annotation-by-annotation trace

| ID | PDF locator and selected context | Problem to solve | Planned and actual change location | Status |
|---|---|---|---|---|
| A01 | p. 2, highlight, rect `(239.4,528.6,358.0,538.6)`; wording around the process/task definition | The central term should not imply that every task is non-equilibrium; distinguish task-relevant or process-conditioned evaluation from a blanket “disruptive/non-equilibrium” label. | Main `sn-article-final.tex`, Introduction and `Task-Conditioned Evaluation`; the paper now defines a task-conditioned distribution and two linked axes: configuration distribution and measurement operator. | **Done** |
| A02 | p. 2, free-text rect `(391.9,527.2,584.9,542.8)`; “disrupt aware process / task-relevant” comment | Define the target question in reader language and avoid making non-equilibrium a prerequisite for relevance. | Main Introduction, Methodology, and Figure 1 caption; use “task-conditioned”, “process-conditioned”, and explicit observables rather than “disrupt-aware” as an undefined category. | **Done** |
| A03 | p. 3, highlight, rect `(100.1,75.6,472.3,97.6)`; defensive contrast in the Introduction | Avoid a defensive “should be X rather than Y” construction; state the scientific problem positively. | Main Introduction was reorganized as broad gap → prior answers → focused missing puzzle → present benchmark. The comparison is framed as an empirical question about specialist, zero-shot, and adapted routes. | **Done** |
| A04 | p. 3, free-text rect `(97.5,41.4,527.8,56.9)`; direct request for task levels and task metrics | Introduce task difficulty levels before discussing model conclusions, and tie each level to a measurable quantity. | Main Introduction and Figure 1 now move from near-equilibrium pointwise error to fixed-path migration, self-relaxed NEB/MD stability, collective motion, and larger-distortion probes; Methodology names the associated operators and observables. | **Done** |
| A05 | p. 3, highlight, rect `(426.2,447.6,436.2,457.6)`; highlighted hyphenated wording | Remove unnecessary argumentative hyphens and awkward compound phrasing. | Main Introduction, Methodology, captions, and revised Figure 1 were edited for plain prose. Technical compounds that are scientifically conventional (for example, `fixed-path`, `zero-shot`, and `out-of-distribution`) remain only where they carry a defined technical meaning. | **Done** |
| A06 | p. 3, free-text rect `(472.5,448.2,542.7,463.7)`; “这种 hyphen 都要避免出现” | Apply the style rule consistently rather than changing only one sentence. | Main reader-facing prose was scanned after restructuring; unnecessary hyphenated rhetorical phrases were removed. Defined technical labels are retained for precision and are explained on first use. | **Done** |
| A07 | p. 3, free-text rect `(98.8,605.0,464.7,643.2)`; related-work paragraph should connect organically | Related work must serve the argument rather than look like a response inserted for a reviewer. | Main Introduction related-work paragraph now summarizes broad equilibrium benchmarks and process-focused studies, then states the remaining fixed-system, fixed-protocol comparison gap before introducing MigrationBench. | **Done** |
| A08 | p. 3, free-text rect `(304.2,530.7,517.5,591.5)`; broad gap → prior answers → focused gap → this work | Establish a continuous narrative from the general problem through prior evidence to the specific question answered here. | Main Introduction, especially the paragraphs preceding the task-conditioned questions; no reviewer-only transition remains in the main text. Nature-style direct framing is used without claiming that prior work was inadequate. | **Done** |
| A09 | p. 3, highlight, rect `(120.9,531.6,344.8,541.6)`; selected citation-heavy related-work sentence | Do not make readers decode a list of numbered citations; explain what each prior line of work tested and found. | Main Introduction condenses citations into sentence-level summaries: generic equilibrium benchmarks, targeted diffusion adaptation, process/barrier evaluation, and sparse high-energy correction. | **Done** |
| A10 | p. 3, highlight, rect `(100.1,567.6,471.1,577.6)`; selected related-work/gap transition | Make the final gap explicit: prior work establishes useful ingredients, while the matched task-conditioned comparison remains open. | Main Introduction ends the related-work sequence with the focused question: when a foundation prior helps, when target data are needed, and where neither route is reliable on the same system and protocol. | **Done** |
| A11 | p. 6, highlight, rect `(124.6,399.6,495.2,421.8)`; reviewer/process material in the main text | Remove revision history, reviewer tags, and audit narration from the reader-facing manuscript. | Main `sn-article-final.tex`: reviewer-only formalization input was removed; the main text now presents the finished scientific design and evidence. Reviewer-specific explanations remain only in `response_to_reviewers.tex`. | **Done** |
| A12 | p. 6, free-text rect `(295.8,420.6,563.6,458.8)`; explicit GitHub/commit/reviewer clarification comment | No GitHub commit or reviewer clarification should interrupt the main scientific story. | Main Code Availability now says the scripts will be made available with the article and points readers to the SI for provenance/metadata; commit identifiers and reviewer clarifications remain response-only. The data-availability link is retained as a normal release statement. | **Done** |
| A13 | p. 6, highlight, rect `(124.6,243.6,497.2,313.6)`; detailed methods/data block | Move parameter-level and data-processing detail to SI while keeping the main method intelligible. | Main Methodology is concise and active; SI contains benchmark definitions, path classes, training protocol, split controls, descriptor diagnostics, and detailed tables/figures. | **Done** |
| A14 | p. 6, free-text rect `(292.9,288.7,548.1,326.9)`; concise Nature-style method and tables for parameters/data | The main method should explain design choices, not reproduce a lab notebook. | Main `Data and Models` and `Task-Conditioned Evaluation` give the design logic; SI tables and subsections carry the numerical protocol and dataset features. | **Done** |
| A15 | p. 6, free-text rect `(301.5,358.9,527.9,397.1)`; active “we” voice | Describe what the authors did in a direct, reader-facing voice instead of “are documented” or other detached language. | Main Methodology was rewritten with “We generated”, “We used”, “We trained”, and “We evaluated”. Conventional passive wording that describes a recorded protocol may remain in isolated SI captions, but the key actions and design decisions are author-led. | **Done** |
| A16 | p. 6, free-text rect `(7.9,348.1,124.4,420.3)`; C8 details should stay in response and be consolidated in SI | Keep exporter/audit/reviewer resolution out of the main text; consolidate reproducibility logic into one compact data/model pipeline. | Main retains only the scientific data statement: AIMD frames carry real DFT energy/force labels and NEB images remain evaluation data. SI `MACE Model Training Protocols` consolidates the pipeline and metadata; response C8 carries the exporter and audit details. | **Done** |
| A17 | p. 6, highlight, rect `(124.6,315.6,495.2,349.6)`; SI pipeline/metadata block | Provide a coherent SI location for method/data-processing detail instead of scattering reviewer fixes through the paper. | SI `Task-Conditioned Benchmark Definitions`, `MACE Model Training Protocols`, split tables, path registry, and diagnostic appendices now form the detailed reproducibility path. | **Done** |

## Independent reader review closure

The independent reader review was performed after the source restructuring and was
used as a separate check rather than as an edit. The following table records how its
priority items were handled in the target folder.

| Review item | Closure evidence | Status |
|---|---|---|
| P0: stale PDFs and reviewer-only material | Main, SI, and response PDFs were rebuilt from the current `revision1` source state. Main/SI no longer contain reviewer tags or commit explanations; response retains them by design. | **Done** |
| P1: Foundation included incorrectly in the near-equilibrium RMSE claim | Main now says “trained variants” for Table S1 and evaluates zero-shot Foundation checkpoints separately where the matched pointwise table is unavailable. | **Done** |
| P1: non-converged path values called physical barriers | R01–R03 corrected Q1–Q3/M1. R05–R07 now qualify G1/G2 as `UNCONVERGED` fixed-image references in the main/SI text and Fig. S15; round-5 independent review confirmed the closure. | **PASS — independent review round 5** |
| P1: undefined “predefined stability gate” | The exact gate claim was removed from the reader-facing main narrative. The current wording reports late-trajectory/self-relaxed instability under the tested protocol and does not present a physical transition time. | **Done** |
| P1: fine-tuning/SHAP language too causal | Main now describes fine-tuning as conditional and task-specific; representation and SHAP analyses are explicitly descriptive and not a validated acquisition rule. | **Done** |
| P1: one-dimensional ladder obscures two different axes | Main Methodology and Figure 1 explicitly separate configuration distribution from measurement operator, with MD and self-relaxed NEB described as model-in-the-loop/trajectory observables. | **Done** |
| P1: reproducibility metadata and exact historical numerical reproduction | The SI contains the protocol, splits, path classes, diagnostics, and data/model scope. The response explicitly distinguishes workflow shape/alignment reproducibility from unresolved exact regeneration of locked historical SHAP numbers. Immutable model/checkpoint hashes and a single external release manifest are not invented where the current evidence does not provide them. | **Accepted residual** |
| P1: reviewer directory names and “Audit ID” leak into SI source/assets | The execution pass copied the two reviewer-generated evidence assets used by the SI into neutral reader-facing names under `revision1/figures/` and `revision1/tables/`, changed the SI references, renamed the path-registry field to `Path ID`, changed the visible `projection audit` wording to `projection check`, and localized the grouped-reference caption so it no longer says “audited G1 images”. The source-only figure registry and other archival metadata retain internal fields outside the reader-facing manuscript. | **PASS — independent review round 5; source-only residual accepted** |
| P1: response does not clearly distinguish release identities | Response C8/C11/C13 now labels lineage as artifact-specific and explicitly states that no single commit is canonical for every artifact; the three submission PDFs are built from one `revision1` source state. | **Done** |
| P2: stale Figure 2/4 assets and internal wording | Main Figure 2 now uses the revised MD diagnostic; Figure 4 uses the task-conditioned fixed-path error figure; Figure 1 was regenerated with “shared evaluation criteria” and “diagnose coverage gaps”. | **Done** |
| P2: nonfatal build warnings | Final logs were searched for undefined references, multiply-defined citations, missing files, fatal errors, and emergency stops. Remaining warnings are layout/font/template warnings and are not evidence of a broken build. | **Done** |

## Subagent execution gate

The following gate is required by the revision workflow: an execution subagent must
inspect and, where necessary, edit the candidate revision; a different review
subagent must then inspect the resulting source and PDFs without editing them. The
parent agent does not count its own edits as independent execution or approval.

| Execution item | Scope | Executor status | Independent review status |
|---|---|---|---|
| E01 | Replace or verify the stale Figure 4 asset and make its labels, caption, and in-text references agree on G1, Q1--Q3, and fixed-path energy-assignment errors. | **Executed: verified current asset and source references; no old labels found; added neutral `fig_fixed_path_errors` canonical alias for the main Fig. 3 role; corrected the main fixed-path panel reference** | **PASS — independent review round 5** |
| E02 | Recheck the main-text narrative and all updated reader-facing figures against the annotation TODO and the independent-reader findings; remove any remaining reviewer/process language from main/SI. | **Executed: Fig. S6 and response table/figure terminology revised with local assets** | **PASS — independent review round 5** |
| E03 | Rebuild the main, SI, and response PDFs from the post-execution source state and record any remaining warnings or residual source-only provenance paths. | **Executed: rebuilt after R12 source-bound figure-reference alignment; final diagnostics recorded below** | **PASS — independent review round 5** |

## Execution report for this pass

This section records execution work only. It is not an independent review or
approval; the independent reviewer must re-read the resulting source and PDFs.

- **E01 / fixed-path error asset:** Verified the legacy `figures/fig4_task_conditioned_neb_errors.pdf` and
  its PNG companion. Extracted figure text contains G1, Q1, Q2, Q3,
  Foundation, FT-600K, Scratch-5%, and fixed-path error wording; it contains no
  `Deep Penetration (OOD)` or `In-gap (In-distribution)` label. Main caption,
  `fig:neb_errors`, and all main-text references use the same G1/Q1--Q3
  terminology. The figure was visually inspected at readable scale. Added
  `figures/fig_fixed_path_errors.pdf` (and PNG companion) as the canonical
  neutral alias for the asset typeset as main Fig. 3; the legacy filename is
  retained as a provenance copy.
- **E02 / reader-facing text and assets:** Updated SI asset references to
  `figures/grouped_retraining_overlap.pdf`,
  `figures/seed_resolved_path_errors.pdf`, and
  `tables/seed_resolved_path_errors.tex`; changed `Audit ID` to `Path ID`,
  removed the visible `projection audit` wording, localized the grouped-space
  caption, and changed the Q1--Q3 table header/caption from `QE barrier` to
  `QE reference` with an explicit non-converged image-set qualification.
  R01--R03 then changed Table S6's caption to conditional activation-energy
  language, added the QE convergence status column while preserving all
  numerical entries, and qualified Q1--Q3/M1 as image-set references/spans,
  not physical barriers.
  Figure 1, Figure 2, and Figure 4 were visually checked; Figure 1 uses
  reader-facing evaluation criteria and conditional language, Figure 2 uses
  the current MD diagnostic, and Figure 4 uses the current fixed-path error
  data.
- **E03 / builds:** From `paper/revision1/`, `sn-article-final.tex`,
  `sn-article-SI.tex`, and `response_to_reviewers.tex` all compiled to PDF.
  Final outputs are 22, 32, and 23 pages, respectively. The three logs contain
  no undefined references, multiply-defined citations, missing-file errors,
  fatal errors, or emergency stops.
- **R05--R07 / status-qualified reader-facing terminology:** G1/G2 status values
  were not changed. Main text, SI path definitions, Table S6/S8-adjacent prose,
  foundation profiles, and fixed-path captions now describe G1/G2 as QE
  image-set references, forward-minus-first-image energy differences, or
  fixed-image reference errors; only rows marked `CONVERGED` are eligible for
  activation-energy wording.
- **R06 / SI Fig. S15:** Regenerated `figures/figS_neb_operators.pdf` from
  `scripts/plot_neb_operators.py`. The fixed-path panel now uses neutral
  energy-difference/reference-profile labels, while the self-relaxed stability
  panel and its numerical data are preserved.
- **R08 / response alignment:** Updated only response prose and manuscript-change
  descriptions needed to match the status-qualified manuscript/SI terminology;
  reviewer quotations and historical context remain in the response by design.
- **R09 / SI Fig. S6:** Added the local plotting source
  `scripts/plot_seed_resolved_path_errors.py` and regenerated
  `figures/seed_resolved_path_errors.pdf`. Both axes now use fixed-path
  reference-error wording; numerical values, seed error bars, and G1/Q1--Q3
  classes are unchanged.
- **R10 / response tables and figure:** Added local copies
  `tables/a2_grouped_audit_table.tex` and `tables/a3_strict_path_errors.tex`,
  updated the response to use them, and switched its seed-resolved figure to
  the local neutral asset. Captions identify QE image-set references and state
  that UNCONVERGED values are not physical migration barriers.
- **R11 / rebuild and scan:** Rebuilt all three PDFs and scanned the resulting
  source/PDF text for the requested barrier, old-label, and process-marker
  classes; the only remaining historical terms are confined to reviewer
  quotations or source-only archival material.
- **R12 / main figure reference:** Replaced the main-text `Fig. 3d` shorthand
  with `Fig.~\ref{fig:neb_errors}a`, which is bound to the existing fixed-path
  panel. The figure, data, caption, and response-specific historical discussion
  were otherwise left unchanged.

## Historical rework queue after independent review failures (resolved)

| ID | Independent finding | Required executor action | Status |
|---|---|---|---|
| R01 | SI Table S6 caption still calls all forward/reverse values “activation energies”. | Make the caption conditional: converged paths may report activation energies; non-converged Q1–Q3/M1 values are image-set energy references/spans, not physical barriers. | **PASS — independent review round 5** |
| R02 | SI Table S8 and `tables/scratch5_path_resolved_descriptive.tex` still use “QE barrier”. | Rename the reader-facing column to `QE reference` or `QE image-set span` and state the non-convergence qualification in the table caption and nearby text. | **PASS — independent review round 5** |
| R03 | `tables/dft_neb_barriers.tex` retains `E_act` under Q1–Q3. | Use a neutral table header/field name that does not imply activation energy for non-converged rows; preserve the numerical values and status column. | **PASS — independent review round 5** |
| R04 | Fixed-path asset filename `fig4_...` is typeset as Fig. 3 while Fig. 4 is sliding. | Do not silently renumber the paper; either rename the asset/script to its actual canonical role or record a clear source-only alias so response/main references cannot misidentify it. | **PASS — independent review round 5** |

All three PDFs were rebuilt after R01–R04. The historical rework queue was
subsequently rechecked by independent review round 5; the final status is recorded
in each row below and in the final closure section.

## Rework queue after independent review round 2

| ID | Independent finding | Required executor action | Status |
|---|---|---|---|
| R05 | G1/G2 are `UNCONVERGED` in `tables/dft_neb_barriers.tex` and the underlying path registry, but main/SI still use “barrier” or “reference barrier” for G1. | Unless a current, verifiable convergence record can be located and the status table updated consistently, rewrite reader-facing G1/G2 language as QE image-set reference, forward energy difference, or fixed-image reference error; do not silently change status. | **PASS — independent review round 5** |
| R06 | SI Fig. S15 source/PDF uses `Fixed-path barrier (eV)` and “DFT reference barrier” despite the current status. | Regenerate or edit the figure and caption to use neutral fixed-path/reference-profile terminology consistent with the source status; preserve the numerical comparison and separate self-relaxed stability panel. | **PASS — independent review round 5** |
| R07 | Main and SI foundation/fixed-path paragraphs still use “predicted barriers”/“DFT reference barrier” for G1. | Align main, SI, captions, and table-adjacent prose with the chosen status-qualified terminology; keep the scientific comparison and uncertainty values unchanged. | **PASS — independent review round 5** |
| R08 | Response locations and claims must remain matched after the G1 wording change. | Update response-only location descriptions if needed, while retaining reviewer context only in the response; rebuild all three PDFs and run a full reader-facing marker/terminology scan. | **PASS — independent review round 5** |

All three PDFs were rebuilt after R05–R08. The historical rework queue was
subsequently rechecked by independent review round 5; the final status is recorded
in each row below and in the final closure section.

## Rework queue after independent review round 3

| ID | Independent finding | Required executor action | Status |
|---|---|---|---|
| R09 | SI Fig. S6 still displays `Within-gap |barrier error| (eV)` and `Absolute barrier error (eV)` for G1/Q1–Q3, all currently marked `UNCONVERGED`. | Regenerate `figures/seed_resolved_path_errors.pdf` and its local plotting source with neutral labels such as `Within-gap fixed-path reference error (eV)` and `Absolute fixed-path reference error (eV)`; preserve data, error bars, and task classes. | **PASS — independent review round 5** |
| R10 | Response p.5 and p.12 captions still say `QE barrier`/`fixed-path barrier errors` for G1/G2. | Use revision1-local table/caption copies or a local response-source override so response captions say `QE image-set reference`/`fixed-path reference error` and state that non-converged values are not physical barriers. Do not modify source files outside `paper/revision1/`. | **PASS — independent review round 5** |
| R11 | The final scan must cover the two newly identified residual classes. | Rebuild main, SI, and response; scan PDF text for `barrier error`, `QE barrier`, `DFT reference barrier`, `predicted barriers`, old Figure 4 labels, reviewer/commit/audit markers; visually inspect SI Fig. S6 and response p.5/p.12. | **PASS — independent review round 5** |

All three PDFs were rebuilt after R09–R11. The historical rework queue was
subsequently rechecked by independent review round 5; the final status is recorded
in each row below and in the final closure section.

## Rework queue after independent review round 4

| ID | Independent finding | Required executor action | Status |
|---|---|---|---|
| R12 | Main `sn-article-final.tex:306` and PDF p.8 still cite `Fig. 3d`, but the current fixed-path Fig. 3 has only panels (a) and (b). | Replace the stale citation with a source-bound reference to the existing fixed-path panel (for example `Fig.~\ref{fig:neb_errors}a` or the figure as a whole), rebuild all three PDFs, and verify no invalid panel reference remains. | **PASS — independent review round 5** |

All three PDFs were rebuilt after R12. A new independent reviewer,
different from the executor, must recheck R12, E01–E03, A01–A17, and the full
reader-facing PDF/source marker scan before any submission recommendation.

## Completion audit after independent review round 5

The parent agent performed a fresh, read-only completion audit against the
current evidence registry and the latest reader-facing source. This audit is a
new execution/review gate; the round-5 PASS does not close the items below.

| ID | Independent finding | Required executor action | Status |
|---|---|---|---|
| F01 | Main Figure 2 still uses the short, legacy `fig_md_msd_kappa_revised.pdf` and its caption/text claim Green--Kubo and thermal-conductivity quantities, while the latest seed-aware MD evidence is the 1~ns B--G benchmark with no same-campaign heat-flux record and no defensible $\kappa$ claim. | Localized `fig_md_seed_aware_1ns.pdf` and the seed-aware result/uncertainty summaries; replaced the main and SI Figure 2 assets/captions; rewrote the adjacent Results and SI transport text; removed the old table's $\kappa$ columns; aligned C9 response prose around seeds, $D_{xy}$, pressure availability, VACF, failure branches, and the no-HFACF boundary. | **PASS — final independent review** |
| F02 | Main Figure 5 still calls the legacy `paper/Fig/NeurIPS2025-AI4MAT-figure6-phate.pdf`; its logged silhouette panel is inconsistent with the current revision1 descriptor audit (raw Euclidean 0.33, z-scored Euclidean $-0.02$, $t$-SNE $0.40\pm0.03$, PHATE $0.47\pm0.00$). The current SI figure also visibly contains the internal label `Old Fig. 5d (logged)`. | Added `scripts/plot_representation_summary.py`, generated local reader-facing main/SI summary figures from the current CSV files, removed the internal label, and aligned main/SI/response captions and text with raw/cosine/z-scored/PCA and descriptive projection values. The text explicitly states that projection values are qualitative and non-causal. | **PASS — final independent review** |
| F03 | Main Figure 4 still depends on the 2025 legacy sliding asset outside `revision1/`, with no current revision1-local provenance alias or evidence-level caption record. | Localized the approved asset as `figures/fig_sliding_revised.pdf` and generated the reader-facing neutral derivative `figures/fig_sliding_neutralized.pdf`; updated the main includegraphics/caption and sliding discussion to fixed-path relative-energy scope, and recorded the source, derivative hash, and absence of convergence/seed evidence in `figure_registry/figures/FIG-171/manifest.yaml` and `revision1_local_provenance.md`. | **PASS — final independent review** |

### Executor report for F01–F03 (not final acceptance)

- Rebuilt sn-article-final.pdf, sn-article-SI.pdf, and response_to_reviewers.pdf into this directory using the project-root compilation entry point.
- Main/SI PDFs were checked for the revised seed-aware MD figure, the clean representation summary, the localized sliding figure, and the absence of the removed legacy representation label.
- Build logs for the revision1 outputs contain no fatal error, emergency stop, missing-file, undefined-reference, undefined-citation, or multiply-defined-label diagnostic after the final rebuild.
- Remaining evidence boundaries are explicit: no same-campaign heat-flux record and therefore no HFACF/$\kappa$ claim; projection values are descriptive rather than causal or acquisition rules; and the localized sliding asset has no convergence or training-seed evidence and is not presented as a validated barrier.
- This section records execution only; the final independent review later closed F01–F03 as PASS.

## Follow-up queue after independent completion review

The independent completion review at
`/tmp/migrationbench_independent_completion_review_20261023.md` was read-only
and returned **FAIL**. The following items are the executor's required
follow-up; they were subsequently verified and closed by a different reviewer.

| ID | Independent finding | Required executor action | Status |
|---|---|---|---|
| R13 | Response C9 still uses the old “block uncertainties” and “matched-protocol multi-seed closure unavailable” wording, and locates the seed-aware figure as Supplementary Fig. S11 although it is Fig. S12. | Align the C9 overview, disposition, manuscript-change, and location text with the seed-aware evidence: training-seed sample SD for FT families, representative-model spread where applicable, and the explicit no-heat-flux/no-HFACF/no-$\kappa$ boundary; change the figure location to Fig. S12. | **PASS — final independent review** |
| R14 | Response A6/C11 text calls the representation results “Fig. 6”, but current Fig. 6 is SHAP and representation diagnostics are Fig. 5. | Correct the response-only figure number and rebuild; preserve reviewer context but make every location pointer resolve to the current manuscript. | **PASS — final independent review** |
| R15 | Response overview calls collective sliding “Fig. 5”, while the current main manuscript uses Fig. 4. | Correct the response overview/location pointer to Fig. 4 and rebuild. | **PASS — final independent review** |
| R16 | Localized Figure 4 still visibly embeds “NEB Migration Barrier: DFT vs. MACE Models” and “DFT (Ground Truth)”, which overstates a fixed-path relative-energy diagnostic lacking convergence and training-seed evidence. | Regenerate or edit the local sliding figure with neutral labels such as “Fixed-path relative-energy profile” and “DFT reference”; change the main caption from “barrier estimates” to “peak relative-energy values”; retain the explicit evidence boundary. | **PASS — final independent review** |
| R17 | Main text contains the imprecise phrase “the barrier, transport, and stability benchmarks” in the representation-analysis transition. | Replace it with status-accurate wording such as “fixed-path migration, transport, and stability benchmarks”, without expanding the claim. | **PASS — final independent review** |
| R18 | The TODO's historical round-5 PASS appears before the F01–F03 follow-up is independently closed and could be misread as the current submission verdict. | Keep the historical PASS as provenance but explicitly mark it superseded by the current F01–F03/R13–R17 gate; only the final follow-up reviewer may create the current PASS closure. | **PASS — final independent review** |

### Executor report for R13–R18 (not final acceptance)

- Updated `response_to_reviewers.tex` C9 overview, disposition, manuscript-change, and location text to match the localized seed-aware MD evidence and Supplementary Fig.~S12; preserved the reviewer quote and historical reviewer context.
- Corrected response-only navigation so collective sliding points to main Fig.~4 and representation diagnostics point to main Fig.~5, while the SHAP result remains main Fig.~6.
- Added `scripts/neutralize_sliding_figure.py` and generated `figures/fig_sliding_neutralized.pdf`; the derivative overlays neutral labels on the baked-in source artwork while preserving curves, markers, axes, and panel structure. Updated the FIG-171 hash/provenance record.
- Updated the main Figure~4 caption and representation transition wording without adding convergence, seed, barrier, HFACF, $\kappa$, or transport-ranking claims.
- This report did not constitute acceptance; the final independent review later closed R13–R18 and F01–F03 as PASS.

## Follow-up queue after independent completion review 2

The independent completion review 2 at
`/tmp/migrationbench_independent_completion_review2_20261023.md` was read-only
and returned **FAIL**. It found no new scientific-content failure in F01–F03 or
R13–R18, but identified two response/build-quality items that must be executed
and independently rechecked; both were later closed by the final review.

| ID | Independent finding | Required executor action | Status |
|---|---|---|---|
| R19 | Response source lines 334 and 361 still describe the current fixed-path result as “main-text Fig. 3d”, although the current main Fig. 3 has only panels (a) and (b). The historical reviewer quotation may retain its original Fig. 3d wording, but the response's own prose and table caption must be source-bound. | Changed only the current-description prose/caption to `main-text Fig.~3a`, preserved the verbatim reviewer quotation, rebuilt all three PDFs, and scanned the response source/PDF for stale current-location references. | **PASS — final independent review** |
| R20 | The response build retains a non-fatal overfull hbox warning around source lines 714–716 from a long reviewer-response identifier. | Inserted a TeX line-break opportunity within the commit identifier without changing scientific content or reviewer quotations; rebuilt all three PDFs and recorded the remaining warning state in the executor report below. | **PASS — final independent review** |

### Executor report for R19–R20 (not final acceptance)

- Rebuilt `sn-article-final.pdf`, `sn-article-SI.pdf`, and
  `response_to_reviewers.pdf` with the explicit output directory set to this
  `revision1/` folder. All three builds completed successfully; the final PDFs
  are newer than their corresponding source files.
- R19: the response's current prose and table caption now point to
  `main-text Fig.~3a`. A response-wide source scan finds `Fig.~3d` only in the
  A1 section heading and the verbatim historical reviewer quotation; no current
  response description or location pointer retains that stale panel reference.
- R20: the 27.0297pt overfull box previously reported at response source lines
  714--716 is absent after adding a TeX line-break opportunity inside the
  identifier. The response log still contains an unrelated 19.9438pt overfull
  box in `tables/a2_grouped_audit_table.tex` lines 5--19; the main log retains
  a 65.76123pt bibliography-URL overfull box at lines 790--798, and the SI log
  retains a 16.0pt table overfull box at lines 449--454. These are recorded
  residual layout warnings and were not changed because they are outside the
  requested R20 identifier fix.
- Final log scans found no fatal error, emergency stop, missing input, undefined
  reference/citation, or multiply-defined label. Main/SI PDF marker scans remain
  clean for reviewer, GitHub, commit, response-letter, audit/process, stale
  `Fig. 3d`, `barrier error`, `QE barrier`, and `Old Fig. 5d` markers.
- This section records execution only; the final independent review later closed
  R19–R20, F01–F03, and R13–R18 as PASS.

## Final independent review closure

The fifth independent review was read-only and did not modify the repository:
`/tmp/migrationbench_independent_final_review5_20261023.md`. Its verdict was
**PASS — recommend submission**. It independently verified R05–R12, E01–E03,
and A01–A17; confirmed the current main PDF uses `Fig. 3a` while response-only
historical `Fig. 3d` discussion remains in the response; confirmed no current
reader-facing hits for the forbidden barrier/old-label/process-marker classes;
and accepted the external `analysis/...` asset dependency as a package residual,
not a reader-facing failure.

This paragraph records the historical round-5 state only. It was superseded by
the follow-up queues above and is retained as provenance; it was not the current
submission approval.

## Current final independent review closure

The fresh read-only review at
`/tmp/migrationbench_independent_completion_final_20261023.md` returned
**PASS — recommend submission**. It independently verified A01–A17, R01–R12,
F01–F03, and R13–R20; confirmed current source/PDF freshness, clean fatal and
missing-graphic/reference scans, and visual consistency of main Figures 2, 4,
and 5 with SI Figures S12 and S16. It confirmed that the current main/SI PDFs
remain reader-facing, while reviewer context is confined to the response.

Accordingly, F01–F03 and R13–R20 are closed as **PASS — final independent
review**. The parent agent records the independent review's approval here; it
does not substitute for the journal's editorial decision.

### Execution residuals for independent review

The SI still includes some externally stored scientific figures and captions
under the existing `analysis/task_conditioned_distribution_atlas_20261018`
path, and the source registry contains archival provenance fields. These are
not visible reviewer markers in the current SI PDF, but the independent
reviewer should decide whether the final submission package should vendor or
rename every remaining external asset. The builds retain non-fatal template,
font, underfull/overfull box, and PDF-version warnings. No scientific claim was
expanded to close the unresolved immutable checkpoint-hash and exact historical
numerical-reproduction gap.

## Final release checks recorded

The following mechanical checks were completed for the current source/PDF state:

1. All three source builds were rerun after the final text/figure changes.
2. Each PDF is newer than its corresponding source and has no missing graphics.
3. Main/SI PDF marker scans are clean for reviewer, commit, GitHub, audit/process, stale `Fig. 3d`, barrier-error, QE-barrier, and legacy representation labels.
4. Main Figures 2, 4, and 5 and SI Figures S12 and S16 were visually checked.
5. The response remains reviewer-specific and its current figure/table locations match the final manuscript.

Accepted residuals remain explicitly bounded: no same-campaign heat-flux record
for HFACF or $\kappa$; representation projections are descriptive rather than
causal or acquisition rules; the collective-sliding asset lacks convergence and
training-seed validation and is not a validated barrier; and a small number of
non-fatal layout warnings remain without visible clipping.
