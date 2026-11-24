# Figure evidence group register (2026-10-08)

This is the human-readable layer over `figure_evidence_matrix.tsv`. The matrix has one row per logical figure output (PNG/SVG/PDF siblings are one row); this register gives the parent/child naming, article mapping, caption, result, discussion, and evidence-boundary language. `candidate` means a proposed article placement, not an accepted manuscript change.

## Naming hierarchy

Use the group ID as the parent, then a semantic child suffix: for example `MD-1NS.A` = evidence-aware benchmark, `MD-1NS.B` = final diffusion benchmark, `TASK3-NEB.61.1-4` = one Cr-61 path child, and `TASK3-NEB.81G.neb_1` = one Cr-81 grouped child. Export formats never receive separate figure numbers.

## Inventory coverage

The machine inventory contains **204 logical figure outputs** across **28 groups**. The exact per-output paths and SHA256 hashes are in `figure_inventory.tsv` and `figure_evidence_matrix.tsv`; this register deliberately keeps compiled manuscript PDFs separate from scientific figures.

## ARCHIVE-ENDPOINT — Archived endpoint-relaxation curves

- **Members (4 logical outputs):**
  - `data/published/endpoint_relax_comparison/exact_iter/1-6_iter108/force_relaxation_curves` (recent; partial)
  - `data/published/endpoint_relax_comparison/exact_iter/1-7_iter112/force_relaxation_curves` (recent; partial)
  - `data/published/qe_geometry_transform_relax_ab_s47/comparison/1-3/force_relaxation_curves` (recent; partial)
  - `data/published/qe_geometry_transform_relax_ab_s47/comparison/1-5/force_relaxation_curves` (recent; partial)
- **文章位置 / article target:** SI endpoint-relaxation evidence
- **证据链 / evidence chain:** Published endpoint manifests/log summaries -> archived force-relaxation curves; renderer not co-located.
- **Caption / 图注:** Force relaxation histories for exact endpoint/iteration or geometry-transform comparisons, with model/job/path identity.
- **Results / 结果描述:** These are convergence-history records and should be read with the endpoint gate, not as cross-model energy comparisons.
- **Discussion / 分析讨论:** Retain failed/unconverged curves as provenance. A smooth curve without identity and final gate evidence is incomplete.
- **当前状态 / status:** historical archive, partial chain
- **缺口 / gaps:** Recover plot command and raw log hashes; connect each curve to the endpoint table and manuscript claim.

## ARCHIVE-LEAKAGE — Training-leakage distance distribution

- **Members (1 logical outputs):**
  - `data/published/training_leakage_audit/checkpoint_finetuned_MACE_multihead0804_s42/leakage_distance_distributions` (recent; partial)
- **文章位置 / article target:** Methods/data provenance and reviewer evidence
- **证据链 / evidence chain:** `leakage_audit.json`, nearest-neighbor CSV, distance summary -> archived distribution plot; generator is outside the figure directory.
- **Caption / 图注:** Distance distributions for the audited checkpoint versus reference/training samples, with split scope and distance definition.
- **Results / 结果描述:** Supports a data-leakage/proximity audit, not downstream task accuracy.
- **Discussion / 分析讨论:** Must name the exact source scope and not generalize to unrelated four-model or Dual-X workflows.
- **当前状态 / status:** partial but data artifacts present
- **缺口 / gaps:** Link exact renderer and audit commit; extend source ledger if a manuscript claim uses this plot.

## ARCHIVE-NEB-BARRIER — Archived NEB barrier atlas derivatives

- **Members (15 logical outputs):**
  - `data/published/neb_barrier_atlas/1-2_historical` (recent; partial)
  - `data/published/neb_barrier_atlas/1-3_historical` (recent; partial)
  - `data/published/neb_barrier_atlas/1-4_historical` (recent; partial)
  - `data/published/neb_barrier_atlas/1-5_historical` (recent; partial)
  - `data/published/neb_barrier_atlas/1-6_historical` (recent; partial)
  - `data/published/neb_barrier_atlas/1-6_qe_r3` (recent; partial)
  - `data/published/neb_barrier_atlas/1-7_historical` (recent; partial)
  - `data/published/neb_barrier_atlas/1-7_qe_r3` (recent; partial)
  - `data/published/neb_barrier_atlas/1-8_historical` (recent; partial)
  - `data/published/neb_barrier_atlas/81_neb_1` (recent; partial)
  - `data/published/neb_barrier_atlas/81_neb_2` (recent; partial)
  - `data/published/neb_barrier_atlas/81_neb_3` (recent; partial)
  - `data/published/neb_barrier_atlas/81_neb_4` (recent; partial)
  - `data/published/neb_barrier_atlas/81_neb_5` (recent; partial)
  - `data/published/neb_barrier_atlas/all_qe_neb_profiles` (recent; partial)
- **文章位置 / article target:** SI DFT NEB barrier inventory / source ledger
- **证据链 / evidence chain:** `barrier_atlas.csv/json` -> archived PNG derivative; no co-located plot generator.
- **Caption / 图注:** Per-path barrier-atlas summary with path identity, forward barrier, convergence/status flags, and source provenance. Unconverged/diverged paths remain retained but are not final references.
- **Results / 结果描述:** These assets preserve historical path-level values and flags used by the revision audit.
- **Discussion / 分析讨论:** They are provenance records, not automatically publication-ready figures. A `JOB DONE` line or an optimizer value does not close SCF/force/NEB/path identity gates.
- **当前状态 / status:** historical archive, partial chain
- **缺口 / gaps:** Recover the renderer and raw-input lineage; associate each child with `figS_dft_neb_mep_profiles` only through the source ledger, not filename similarity.

## ARCHIVE-NEB-PROFILE — Archived NEB energy-profile derivatives

- **Members (18 logical outputs):**
  - `data/published/neb_energy_profiles/00_1-2__mb_mlff_cpu_1_2_s42_30791344` (recent; partial)
  - `data/published/neb_energy_profiles/02_1-3__mb_mlff_cpu_1_3_s42_30791345` (recent; partial)
  - `data/published/neb_energy_profiles/05_1-5__mb_mlff_cpu_1_5_s42_30791346` (recent; partial)
  - `data/published/neb_energy_profiles/06_1-6__mb_mlff_cpu_1_6_s42_30791347` (recent; partial)
  - `data/published/neb_energy_profiles/07_1-6__mlff_1-6_repair_idpp_s42` (recent; partial)
  - `data/published/neb_energy_profiles/08_1-6__mb_qe16_direct_pf_s42_30791500` (recent; partial)
  - `data/published/neb_energy_profiles/09_1-6__mb_qe16_direct_r2_s42_30791828` (recent; partial)
  - `data/published/neb_energy_profiles/10_1-6__mb_qe16_direct_r3_s42_30812496` (recent; partial)
  - `data/published/neb_energy_profiles/12_1-7__mb_qe17_direct_pf_s42_30791508` (recent; partial)
  - `data/published/neb_energy_profiles/13_1-7__mb_qe17_direct_r2_s42_30791829` (recent; partial)
  - `data/published/neb_energy_profiles/14_1-7__mb_qe17_direct_r3_s42_30812497` (recent; partial)
  - `data/published/neb_energy_profiles/19_81_neb_1__historical_2D_421_neb_1` (recent; partial)
  - `data/published/neb_energy_profiles/20_81_neb_2__historical_2D_421_neb_2` (recent; partial)
  - `data/published/neb_energy_profiles/21_81_neb_3__historical_2D_421_neb_3` (recent; partial)
  - `data/published/neb_energy_profiles/23_81_neb_4__historical_2D_421_neb_4` (recent; partial)
  - `data/published/neb_energy_profiles/24_81_neb_5__historical_2D_421_neb_5` (recent; partial)
  - `data/published/neb_energy_profiles/27_victor_neb2__victor_historical_V-ST_neb2` (recent; partial)
  - `data/published/neb_energy_profiles/28_victor_neb2_copy__victor_historical_V-ST_neb2_copy` (recent; partial)
- **文章位置 / article target:** SI DFT NEB profiles / fixed-path benchmark source evidence
- **证据链 / evidence chain:** `neb_energy_summary.csv/json` and `neb_energy_snapshot.md` -> archived profile PNG; plotting generator not co-located.
- **Caption / 图注:** Relative energy along the exact archived path/images, with path identity and source job retained. Profiles are qualified as candidate or exploratory where the original convergence evidence is incomplete.
- **Results / 结果描述:** The profiles support path-family inventory and fixed-geometry comparison, but do not by themselves support converged barrier ranking.
- **Discussion / 分析讨论:** Keep as source evidence; do not merge path families or infer equivalence from similar names.
- **当前状态 / status:** historical archive, partial chain
- **缺口 / gaps:** Recover the plotting script/config and verify exact path/image order, units, endpoints, and hashes against the source ledger.

## ARCHIVE-PRECOND — Archived MLFF preconditioner residual smoke

- **Members (1 logical outputs):**
  - `data/published/mlff_preconditioner_comparison/nonlinear_historical_warp_smoke/mlff_preconditioner_residual` (recent; partial)
- **文章位置 / article target:** SI NEB methods/audit only
- **证据链 / evidence chain:** Residual summary -> archived residual plot; source generator not co-located.
- **Caption / 图注:** Residual progression for the specified preconditioner smoke test, labeled as a method diagnostic.
- **Results / 结果描述:** Shows numerical preconditioning behavior, not a physical barrier.
- **Discussion / 分析讨论:** Keep outside main scientific narrative unless needed to explain the NEB implementation.
- **当前状态 / status:** partial archive
- **缺口 / gaps:** Recover exact command, input, and acceptance rule.

## ARCHIVE-SELFNEB — Historical self-consistent NEB profile

- **Members (1 logical outputs):**
  - `data/published/historical_selfneb_foundation_audit/foundation_self_neb_rescored_profile` (recent; partial)
- **文章位置 / article target:** SI NEB stability audit / Fig. S3 lineage
- **证据链 / evidence chain:** Historical self-NEB audit derivative -> archived profile image; source/renderer not co-located.
- **Caption / 图注:** Self-consistent MLFF-NEB profile, explicitly distinguished from fixed-path DFT scoring; any 0.41 eV value is model-reported on the model-optimized path.
- **Results / 结果描述:** The foundation model completed the archived self-consistent path while other models terminated before a converged band; this is a stability result, not fixed-path accuracy.
- **Discussion / 分析讨论:** Preserve the operator distinction emphasized in Reviewer A1 response.
- **当前状态 / status:** legacy audit derivative, partial chain
- **缺口 / gaps:** Link to the exact historical input/log and current SI Fig. S3 asset; do not blend with new fixed-path figures.

## ARCHIVE-TRAJ-PORTFOLIO — Archived representative-trajectory coverage/similarity plots

- **Members (4 logical outputs):**
  - `data/published/representative_trajectory_portfolio/Sb2Te3Cr_61_coverage` (recent; partial)
  - `data/published/representative_trajectory_portfolio/Sb2Te3Cr_61_similarity` (recent; partial)
  - `data/published/representative_trajectory_portfolio/Sb2Te3Cr_81_coverage` (recent; partial)
  - `data/published/representative_trajectory_portfolio/Sb2Te3Cr_81_similarity` (recent; partial)
- **文章位置 / article target:** Methods/SI trajectory coverage audit
- **证据链 / evidence chain:** Portfolio manifest plus coverage gaps, pairwise distances, and representative path tables -> archived image; generator not co-located.
- **Caption / 图注:** Coverage/similarity of the 61- and 81-atom trajectory portfolios, with the exact metric and sampled frame scope named.
- **Results / 结果描述:** These plots document data-selection/coverage decisions rather than model accuracy.
- **Discussion / 分析讨论:** Useful for provenance and sampling justification; not a substitute for a formal train/test distribution audit unless the split source is linked.
- **当前状态 / status:** historical archive, partial chain
- **缺口 / gaps:** Recover renderer and source-frame manifest; explicitly state whether the metric is geometry similarity or training-distribution coverage.

## COMPILED-DOCUMENT — Compiled manuscript/response PDFs

- **Members (4 logical outputs):**
  - `paper/response_to_reviewers` (recent; compiled document)
  - `paper/response_to_reviewers/revision_pack_2026-09-10/response_to_reviewers` (recent; compiled document)
  - `paper/sn-article-SI` (recent; compiled document)
  - `paper/sn-article-final` (recent; compiled document)
- **文章位置 / article target:** Whole document, not an independent figure
- **证据链 / evidence chain:** TeX sources -> compiler -> PDF.
- **Caption / 图注:** Not applicable at the logical-figure level.
- **Results / 结果描述:** Use only for visual placement verification; individual figures are audited separately.
- **Discussion / 分析讨论:** Excluded from figure count in the human audit despite being present in the inventory.
- **当前状态 / status:** compiled document
- **缺口 / gaps:** None as figures; verify included assets via TeX/source audit.

## ENDPOINT-MACE — Four-model endpoint relaxation robustness

- **Members (4 logical outputs):**
  - `analysis/mace_endpoint_four_model/outputs/fig1_endpoint_robustness` (recent; traceable)
  - `analysis/mace_endpoint_four_model/outputs/fig2_relative_energy_landscape` (recent; traceable)
  - `analysis/mace_endpoint_four_model/outputs/fig1_endpoint_robustness` (recent; traceable)
  - `analysis/mace_endpoint_four_model/outputs/fig2_relative_energy_landscape` (recent; traceable)
- **文章位置 / article target:** Results: endpoint robustness / local relaxation sensitivity
- **证据链 / evidence chain:** Raw model summary JSONL and relaxed extxyz archives -> `tidy_endpoint_relaxation_results.csv` plus six summary tables -> `analyze_and_plot.py` -> `fig1_endpoint_robustness` and `fig2_relative_energy_landscape` -> `discussion_draft.md`.
- **Caption / 图注:** A, convergence/effort and paired final-geometry robustness across 47 endpoints, three starting variants, and four models. B, model-internal relative-energy landscapes and endpoint-type dependence; energy zero points are not compared across models.
- **Results / 结果描述:** All models except three Scratch runs at `unique_151` satisfy the local relaxation gate; paired Cr displacements from FT-600K have medians of 0.225–0.284 Å, with only 3.5–15.6% within 0.10 Å.
- **Discussion / 分析讨论:** The figure supports basin-sensitivity screening and DFT-priority selection, not model-independent endpoint geometries or cross-model absolute-energy ranking.
- **当前状态 / status:** traceable current diagnostic
- **缺口 / gaps:** Confirm whether this remains in the journal revision or is an internal endpoint appendix; preserve three Scratch failures in any final table.

## FIG-1-DIFFUSION — Diffusion-aware evaluation / context–method–result composite

- **Members (2 logical outputs):**
  - `analysis/figure1_context_method_result_20261002/figures/figure1_diffusion_aware_evaluation` (recent; traceable)
  - `analysis/figure1_context_method_result_20261002/figures/figure1_diffusion_aware_evaluation` (recent; traceable)
- **文章位置 / article target:** Introduction/Results transition; candidate main Fig. 1
- **证据链 / evidence chain:** `figure1_nature.py` -> hard-coded audited summary values -> `figure1_diffusion_aware_evaluation.{pdf,svg,png}` -> `figure1_caption.md`.
- **Caption / 图注:** a, evaluation gap schematic; b, three-level diffusion-aware evaluation hierarchy; c, same-geometry MD geometry/force-integrity warning; d, exploratory fixed-geometry path sampling. The schematic is conceptual, and the quantitative panels are explicitly not model rankings or converged NEB barriers.
- **Results / 结果描述:** The source caption records a drop in minimum pair distance from 2.393 to 0.629 Å and a rise in model force spread from 1.29 to 4,565.24 eV Å⁻¹, plus Foundation/MPA0 median profile and barrier diagnostics over eight screened paths.
- **Discussion / 分析讨论:** This is a useful framing figure: pointwise agreement is necessary but does not close dynamical or path-level reliability. Panel c must remain a warning about a nonphysical collapsed frame, not a model-accuracy ranking.
- **当前状态 / status:** candidate, traceable to a renderer and caption but panel c/d values are embedded constants rather than linked input tables
- **缺口 / gaps:** Add a data manifest linking the hard-coded values to Phase-1 and processed-NEB CSVs; map the figure to the current revision1 TeX labels before calling it main Fig. 1.

## LEGACY-NEURIPS — Original NeurIPS-era manuscript assets

- **Members (17 logical outputs):**
  - `paper/Fig/NeurIPS2025-AI4MAT-Em-1013` (historical; partial)
  - `paper/Fig/NeurIPS2025-AI4MAT-Em_SI` (historical; partial)
  - `paper/Fig/NeurIPS2025-AI4MAT-Em_main` (historical; partial)
  - `paper/Fig/NeurIPS2025-AI4MAT-SHAP_SI` (historical; partial)
  - `paper/Fig/NeurIPS2025-AI4MAT-SHAP_main` (historical; partial)
  - `paper/Fig/NeurIPS2025-AI4MAT-SHAP_main2` (historical; partial)
  - `paper/Fig/NeurIPS2025-AI4MAT-SI1` (historical; partial)
  - `paper/Fig/NeurIPS2025-AI4MAT-SI2` (historical; partial)
  - `paper/Fig/NeurIPS2025-AI4MAT-fig1-schematic` (historical; partial)
  - `paper/Fig/NeurIPS2025-AI4MAT-fig2-v2` (historical; partial)
  - `paper/Fig/NeurIPS2025-AI4MAT-figure2` (historical; partial)
  - `paper/Fig/NeurIPS2025-AI4MAT-figure3` (historical; partial)
  - `paper/Fig/NeurIPS2025-AI4MAT-figure4` (historical; partial)
  - `paper/Fig/NeurIPS2025-AI4MAT-figure4-v2` (historical; partial)
  - `paper/Fig/NeurIPS2025-AI4MAT-figure5` (historical; partial)
  - `paper/Fig/NeurIPS2025-AI4MAT-figure6` (historical; partial)
  - `paper/Fig/NeurIPS2025-AI4MAT-figure6-phate` (historical; partial)
- **文章位置 / article target:** Original main/SI manuscript, retained for provenance only
- **证据链 / evidence chain:** Image-only historical assets referenced by old TeX captions; modern source chain is not uniformly present.
- **Caption / 图注:** Use the original manuscript captions only as historical text, and label any reused panel with the current source/data scope.
- **Results / 结果描述:** These figures document earlier benchmark framing and may contain superseded model names, protocols, or transport claims.
- **Discussion / 分析讨论:** Do not treat legacy captions as current evidence. The current revision must use the audited source ledger and corrected figures.
- **当前状态 / status:** legacy/image-first
- **缺口 / gaps:** For every reuse candidate, recover raw data, generator, protocol, and current hash; otherwise mark `RECONSTRUCTION` or `REPRODUCED_SCOPE_LIMITED`.

## LEGACY-PUBLISHED — Duplicate packaged publication assets

- **Members (5 logical outputs):**
  - `figures/publication/fig6_shap_revised` (recent; partial)
  - `figures/publication/figS_dft_neb_mep_profiles` (recent; partial)
  - `figures/publication/figS_latent_robustness` (recent; partial)
  - `figures/publication/figS_neb_pes_local_audit` (recent; partial)
  - `figures/publication/fig_md_msd_kappa_revised` (recent; partial)
- **文章位置 / article target:** Revision1 manuscript assets, duplicate location
- **证据链 / evidence chain:** Copied/packaged PDF assets without an independent generator.
- **Caption / 图注:** Use the caption attached to the canonical Revision1 asset, not a duplicate filename.
- **Results / 结果描述:** No independent result; duplicate must hash-match or be explained.
- **Discussion / 分析讨论:** Avoid counting duplicates as separate figures.
- **当前状态 / status:** duplicate/legacy
- **缺口 / gaps:** Hash-compare against canonical revision1 figures and record the source-of-truth path.

## LEGACY-SHAP — Legacy SHAP/descriptor diagnostics

- **Members (10 logical outputs):**
  - `paper/Fig/SHAP/comparison_error_distribution` (historical; partial)
  - `paper/Fig/SHAP/comparison_feature_stability` (historical; partial)
  - `paper/Fig/SHAP/comparison_prediction_scatter` (historical; partial)
  - `paper/Fig/SHAP/comparison_top15_difference` (historical; partial)
  - `paper/Fig/SHAP/comparison_top50_difference` (historical; partial)
  - `paper/Fig/SHAP/enhanced_error_feature_analysis` (historical; partial)
  - `work/rockfish_shap_candidates/2-concentration/soap_analysis_results1/feature_importance_by_doping` (historical; partial)
  - `work/rockfish_shap_candidates/2-concentration/soap_analysis_results1/importance_differences_heatmap` (historical; partial)
  - `work/rockfish_shap_candidates/2-concentration/soap_analysis_results1/soap_importance_heatmap_radial_angular` (historical; partial)
  - `work/rockfish_shap_candidates/2-concentration/soap_analysis_results1/soap_pairs_importance` (historical; partial)
- **文章位置 / article target:** Original or reviewer-facing interpretability evidence
- **证据链 / evidence chain:** Historical image assets and candidate SOAP/SHAP outputs; current canonical chain is `revision1/scripts/shap_analysis.py` + `shap_feature_labels.py` + `plot_shap_fig6.py` + compact inputs.
- **Caption / 图注:** Feature-importance, error-distribution, or model-comparison diagnostic with the surrogate target and validation scope explicitly named.
- **Results / 结果描述:** Only the revised cross-validated SHAP chain is article-ready; prior in-sample or image-only panels are superseded unless re-audited.
- **Discussion / 分析讨论:** Interpret as surrogate error-predictor sensitivity, not causal MACE internals or physical mechanism.
- **当前状态 / status:** mixed legacy/current candidates
- **缺口 / gaps:** One-to-one map old assets to the revised Fig. 6 and mark all unlinked images as historical.

## LEGACY-SI — Legacy supplementary image assets

- **Members (4 logical outputs):**
  - `paper/Fig/SI/1-scratch-mace_l1_0802_run-123_train_default_stage_two` (historical; partial)
  - `paper/Fig/SI/2-finetuned_MACE_multihead0804_run-123_train_default_stage_two` (historical; partial)
  - `paper/Fig/SI/3-finetuned_MACE_multihead0804_run-123_train_default_stage_two-Multi-T` (historical; partial)
  - `paper/Fig/SI/NeurIPS2025-AI4MAT-figure3-2` (historical; partial)
- **文章位置 / article target:** Original SI; historical reference
- **证据链 / evidence chain:** Image-only or historical supplementary assets; source linkage varies.
- **Caption / 图注:** Use original captions only after protocol, model identity, and source data are confirmed.
- **Results / 结果描述:** Not current evidence without re-audit.
- **Discussion / 分析讨论:** Keep separate from Revision1 audited SI figures.
- **当前状态 / status:** legacy/image-first
- **缺口 / gaps:** Recover source chain or mark unavailable.

## MD-1NS — MACE 1 ns benchmark family

- **Members (9 logical outputs):**
  - `analysis/mace_1ns_publication_figure_20261005/figures/mace_1ns_evidence_aware_benchmark` (recent; traceable)
  - `analysis/mace_1ns_publication_figure_20261005/figures/mace_1ns_final_diffusion_benchmark` (recent; traceable)
  - `analysis/mace_1ns_publication_figure_20261005/figures/mace_1ns_overleaf_A_to_G_template` (recent; traceable)
  - `analysis/mace_1ns_publication_figure_20261005/figures/mace_1ns_overleaf_B_to_G` (recent; traceable)
  - `analysis/mace_1ns_publication_figure_20261005/figures/mace_1ns_strict_six_axis_replica` (recent; traceable)
  - `analysis/mace_1ns_publication_figure_20261005/figures/old_fig_from_scratch_failure` (recent; traceable)
  - `analysis/mace_1ns_publication_figure_20261005/figures/old_fig_msd` (recent; traceable)
  - `analysis/mace_1ns_publication_figure_20261005/figures/old_fig_stability` (recent; traceable)
  - `analysis/mace_1ns_publication_figure_20261005/figures/old_fig_vacf` (recent; traceable)
- **文章位置 / article target:** Results: dynamic stability and diffusion; SI transport-method/results candidate; current main-text integration is pending
- **证据链 / evidence chain:** Remote four-model archive -> local `data/*/{audit,metrics,msd,timeseries,vacf}.(json,csv)` and derived `data/thermo/*` -> `plot_figure.py`, `plot_final_benchmark.py`, `plot_strict_replica.py`, `plot_overleaf_panels.py` plus recomputation helpers -> three manifests -> 9 logical render variants.
- **Caption / 图注:** Parent caption: six-model-axis or evidence-aware benchmark at 600 K for 2,050 atoms, with temperature, pressure, stability/failure, MSD, VACF and explicitly qualified transport axes. The final benchmark removes HFACF and κ because no heat-flux time series is archived; the strict replica retains those slots only as `NOT AVAILABLE`.
- **Results / 结果描述:** FT-600K and FT-MultiT are stable for the archived diffusion analysis; Foundation-OMAT crosses a thermal threshold at 8.8 ps and from-scratch at 462.2 ps. The final package reports the stable models for diffusion and retains failures as evidence.
- **Discussion / 分析讨论:** Use the final diffusion benchmark for diffusion/stability claims. Do not claim thermal conductivity from the strict replica. Training seeds and MD velocity seeds remain separate, and identical velocity seeds are not independent replicas.
- **当前状态 / status:** recent, strongest current evidence package; scientific status is model-specific, not uniformly successful
- **缺口 / gaps:** Integrate the accepted variant into revision1 text/captions; preserve the explicit absence of HFACF/κ and verify the remote-to-local source hashes if the archive changes.

## MD-1NS-LEGACY — Legacy continuation-MD panels

- **Members (4 logical outputs):**
  - `md_continuation_20260930/analysis_publication_20261003/figures/fig_from_scratch_failure` (recent; partial)
  - `md_continuation_20260930/analysis_publication_20261003/figures/fig_msd` (recent; partial)
  - `md_continuation_20260930/analysis_publication_20261003/figures/fig_stability` (recent; partial)
  - `md_continuation_20260930/analysis_publication_20261003/figures/fig_vacf` (recent; partial)
- **文章位置 / article target:** SI transport/audit appendix only until lineage is reconciled with MD-1NS
- **证据链 / evidence chain:** `md_continuation_20260930/analysis_publication_20261003/make_publication_artifacts.py` and local audit JSON -> four figure stems (`fig_stability`, `fig_msd`, `fig_vacf`, `fig_from_scratch_failure`).
- **Caption / 图注:** Stability, MSD, VACF, and from-scratch failure panels from the continuation-MD audit; each caption must name the exact trajectory segment and whether the value is a diagnostic or an accepted transport estimate.
- **Results / 结果描述:** The outputs preserve stable/failed branches and show why a failed from-scratch branch cannot be pooled into a formal 1 ns transport estimate.
- **Discussion / 分析讨论:** These are useful audit/legacy panels but should not compete with MD-1NS final figures. They need a direct source-trajectory hash and exact window definition before publication reuse.
- **当前状态 / status:** recent but legacy/duplicate family
- **缺口 / gaps:** Record exact raw trajectory paths, derived-window inputs, and a superseded-by relationship to MD-1NS.

## MD-ABLATION-20PS — From-scratch ablation MD screening and transport diagnostic

- **Members (3 logical outputs):**
  - `analysis/from_scratch_ablation_md_20261005/figures_plot_atlas/screening_plot_atlas_msd_dxy` (recent; traceable)
  - `analysis/from_scratch_ablation_md_20261005/figures_unified_atlas_20ps/md_unified_comparison` (recent; traceable)
  - `analysis/from_scratch_ablation_md_20261005/remote_20ps_audit/figures/screening_msd_dxy` (recent; traceable)
- **文章位置 / article target:** Results/SI: training-family ablation, short-trajectory stability, and transport-screen boundary
- **证据链 / evidence chain:** Isolated worktree `/Users/alina/.codex/worktrees/bf6e/Wiley-MigrationBench/analysis/from_scratch_ablation_md_20261005` -> remote registry/status/checkpoint/trajectory artifacts -> screening/MSD/Dxy tables -> Plot Atlas-compatible screening and unified comparison renderers -> `figures_plot_atlas`, `figures_unified_atlas_20ps`, and remote audit figures.
- **Caption / 图注:** Parent caption: short-screen trajectory completeness, stability/failure state, MSD/Dxy screening, and model-family comparison. Each panel names its duration, accepted/failed branch, atom/PBC identity, and whether it is a transport screen or a formal transport estimate.
- **Results / 结果描述:** The worktree contains an independent from-scratch ablation family with explicit screening figures and failed/partial branches retained. A rendered screening figure is evidence of the analysis chain, not proof of a complete 1 ns or converged transport result.
- **Discussion / 分析讨论:** Use this family to separate training-family effects from numerical/runtime failures. Do not pool incomplete or unstable trajectories into formal MSD/Dxy/HFACF/κ claims; retain training seed and MD velocity seed as separate provenance fields.
- **当前状态 / status:** recent, traceable worktree package; branch-isolated and partly screening-only
- **缺口 / gaps:** Link each render to the exact status/checkpoint/trajectory manifest and designate the worktree branch as source-of-truth or superseded relative to current-checkout duplicates; no automatic promotion to the manuscript.

## MD-CONV — MD time-window convergence

- **Members (4 logical outputs):**
  - `analysis/mace_md_time_convergence_20261001/from-scratch/time_convergence` (recent; partial)
  - `analysis/mace_md_time_convergence_20261001/multi-T/time_convergence` (recent; partial)
  - `analysis/mace_md_time_convergence_20261001/from-scratch/time_convergence` (recent; partial)
  - `analysis/mace_md_time_convergence_20261001/multi-T/time_convergence` (recent; partial)
- **文章位置 / article target:** SI transport-method validation
- **证据链 / evidence chain:** `blocks_50ps.csv`, `cumulative_windows.csv`, `time_convergence.json` -> PNG outputs; the plotting script is not co-located in the figure directory.
- **Caption / 图注:** Time-window convergence of block statistics for the from-scratch and multi-temperature trajectories, showing the selected analysis window and whether the estimator has stabilized.
- **Results / 结果描述:** The figure is a convergence diagnostic, not a transport result by itself; it should justify the window used by the accepted MD package.
- **Discussion / 分析讨论:** Keep this as a method-support panel. A convergence-looking trace does not repair a trajectory that fails the physical stability gate.
- **当前状态 / status:** partial evidence chain
- **缺口 / gaps:** Recover or identify the exact plotting script and commit the command/configuration used to produce the PNGs.

## P1-FORCE — Phase 1 same-geometry static force comparison

- **Members (9 logical outputs):**
  - `analysis/phase1_static_force_comparison_20261002/figures/fig1_force_disagreement` (recent; traceable)
  - `analysis/phase1_static_force_comparison_20261002/figures/fig2_key_local_force` (recent; traceable)
  - `analysis/phase1_static_force_comparison_20261002/figures/fig3_geometry_step_diagnostics` (recent; traceable)
  - `analysis/phase1_static_force_comparison_20261002/figures/fig1_force_disagreement` (recent; traceable)
  - `analysis/phase1_static_force_comparison_20261002/figures/fig2_key_local_force` (recent; traceable)
  - `analysis/phase1_static_force_comparison_20261002/figures/fig3_geometry_step_diagnostics` (recent; traceable)
  - `analysis/phase1_static_force_comparison_20261002/figures/fig1_force_disagreement` (recent; traceable)
  - `analysis/phase1_static_force_comparison_20261002/figures/fig2_key_local_force` (recent; traceable)
  - `analysis/phase1_static_force_comparison_20261002/figures/fig3_geometry_step_diagnostics` (recent; traceable)
- **文章位置 / article target:** Results: model disagreement and trajectory-integrity boundary; candidate Fig. 1c/diagnostic SI
- **证据链 / evidence chain:** Common extxyz snapshots and `snapshot_manifest.json` -> NPZ model evaluations -> `analysis/{model_step_summary,pairwise_force_comparison,atom_force_disagreement,geometry_diagnostics}.csv` -> `scripts/plot_phase1.py` -> 3 export variants -> `Results.md`.
- **Caption / 图注:** a, pairwise force disagreement and leave-one-out force RMSE across the four models; b, Cr-local force response and maximum atom spread; c, minimum distance and local-shell diagnostics across the shared snapshots.
- **Results / 结果描述:** The minimum pair distance reaches 0.629 Å and maximum force spread reaches 4,565.24 eV Å⁻¹ at step 462200. The collapsed frame is excluded from formal DFT/diffusion/barrier statistics.
- **Discussion / 分析讨论:** The figure establishes a model-disagreement/trajectory-integrity warning. It does not identify the causal integrator/thermostat mechanism and does not rank model accuracy.
- **当前状态 / status:** recent, fully traceable analysis package
- **缺口 / gaps:** No article include/caption is yet wired to these outputs; the paper-repro duplicate must be reconciled with the current checkout.

## P2-HANDOFF — Phase 2 short-handoff stability and gate family

- **Members (12 logical outputs):**
  - `analysis/phase2_handoff_md_20261002/audit_20261003/figures/phase2_handoff_timeseries` (recent; traceable)
  - `analysis/phase2_handoff_md_20261002/audit_20261003/figures/phase2_stability_gate` (recent; traceable)
  - `analysis/phase2_handoff_md_20261002/remote_artifacts/runs/retry2/handoff_stability_heatmap` (recent; traceable)
  - `analysis/phase2_handoff_md_20261002/remote_artifacts/runs/retry2/handoff_timeseries` (recent; traceable)
  - `analysis/phase2_handoff_md_20261003/figures/phase2_handoff_timeseries` (recent; traceable)
  - `analysis/phase2_handoff_md_20261003/figures/phase2_stability_gate` (recent; traceable)
  - `analysis/phase2_handoff_md_20261002/remote_artifacts/runs/retry2/handoff_stability_heatmap` (recent; traceable)
  - `analysis/phase2_handoff_md_20261002/remote_artifacts/runs/retry2/handoff_timeseries` (recent; traceable)
  - `analysis/phase2_handoff_md_20261002/audit_20261003/figures/phase2_handoff_timeseries` (recent; traceable)
  - `analysis/phase2_handoff_md_20261002/audit_20261003/figures/phase2_stability_gate` (recent; traceable)
  - `analysis/phase2_handoff_md_20261002/remote_artifacts/runs/retry2/handoff_stability_heatmap` (recent; traceable)
  - `analysis/phase2_handoff_md_20261002/remote_artifacts/runs/retry2/handoff_timeseries` (recent; traceable)
- **文章位置 / article target:** Results: dynamic stability mechanism; SI or main diagnostic figure
- **证据链 / evidence chain:** Handoff snapshots and submission manifests -> 1,616-row `frame_aligned_metrics.csv` plus hash/anomaly/neighborhood audits -> `audit_phase2.py` and `render_phase2_atlas.py` -> canonical audit figures; retry2 PNG/SVGs are linked as superseded remote artifacts.
- **Caption / 图注:** a, frame-aligned temperature/energy/force/geometry traces for both input states and four models, with first-gate markers; b, model × input-state stability summary and first crossing steps.
- **Results / 结果描述:** Foundation-OMAT and from-scratch run away within 0.02–0.32 ps, while naive-FT and Multi-T remain finite for the short handoff; the first anomalous local force excursion precedes later contact/thermal runaway.
- **Discussion / 分析讨论:** This supports model-dependent local-PES amplification on identical initial states. It does not prove the initial state is a DFT minimum or identify the original long-trajectory mechanism; the 0 accepted formal DFT gate remains visible.
- **当前状态 / status:** recent, fully traceable with duplicate outputs
- **缺口 / gaps:** Designate `audit_20261003/figures` as canonical, record retry2 as superseded, and map the completed integrator sensitivity package into the paper if the stronger mechanism claim is used.

## P3-DFT — Phase 3 DFT recovery/validation diagnostics

- **Members (3 logical outputs):**
  - `analysis/phase3_dft_recovery_20261004/mace_results/figures/mace_reconstruction_diagnostic` (recent; partial)
  - `analysis/phase3_dft_validation_20261002/figures/mace_force_landscape_step_462000_81` (recent; partial)
  - `analysis/phase3_dft_validation_20261002/figures/mace_force_progression_81` (recent; partial)
- **文章位置 / article target:** Results/SI: DFT gate, force validation, and recovery boundary
- **证据链 / evidence chain:** Job ledgers/provenance manifests and ordered-block audits -> MACE reconstruction/force-landscape tables -> figure outputs; formal accepted target values are currently absent.
- **Caption / 图注:** Diagnostic force landscape/progression or reconstruction figure tied to the exact snapshot, ordered block, and job ledger; no panel should imply a converged DFT reference unless the explicit gate passes.
- **Results / 结果描述:** The figures document what was tested and where the DFT validation remains incomplete, rather than supplying a finalized target barrier/force value.
- **Discussion / 分析讨论:** Use these to explain a blocked DFT gate and preserve failed/recovered branches; do not substitute local-tile or cancelled-array evidence for identity-preserving full-cell DFT.
- **当前状态 / status:** partial, scientific gate blocked
- **缺口 / gaps:** Link each output to its exact Slurm job, raw QE input/output, and acceptance report; keep no-accepted-target status in any caption.

## PAPER-OTHER — Unclassified paper-repro output

- **Members (1 logical outputs):**
  - `paper/revision1/response_to_reviewers` (recent; partial)
- **文章位置 / article target:** Unmapped
- **证据链 / evidence chain:** Inventory-only until classified.
- **Caption / 图注:** Not drafted until source scope is identified.
- **Results / 结果描述:** Not used for claims.
- **Discussion / 分析讨论:** Requires manual review.
- **当前状态 / status:** unclassified
- **缺口 / gaps:** Assign a family and locate source data/script/caption.

## REV1-APPENDIX — Revision 1 endpoint perturbation appendix

- **Members (3 logical outputs):**
  - `paper/revision1/appendix/figures/fig1_campaign_overview` (recent; traceable)
  - `paper/revision1/appendix/figures/fig2_perturbation_response` (recent; traceable)
  - `paper/revision1/appendix/figures/fig3_interaction_and_site_maps` (recent; traceable)
- **文章位置 / article target:** SI endpoint-relaxation appendix
- **证据链 / evidence chain:** `appendix/data/endpoint_level_summary.csv/json` -> `appendix/scripts/plot_campaign_atlas.py` using Plot Atlas primitives -> three figure outputs -> `endpoint_perturbation_appendix.tex` captions and `multiarms_table.tex`.
- **Caption / 图注:** Campaign overview/relaxation outcomes; perturbation response across systems and endpoints; host–dopant interaction and relaxation-effort maps. The figures summarize measured 846-task multi-arm endpoint relaxation, not DFT or magnetic conclusions.
- **Results / 结果描述:** The captions and table define six host–dopant systems, perturbation arms, convergence criterion, final energy/force/displacement/steps, and the 846-task scientific-success gate.
- **Discussion / 分析讨论:** This appendix is one of the strongest closed evidence packages in the paper-repro tree because raw compact data, plotting source, table, and TeX placement are co-located.
- **当前状态 / status:** recent, traceable and article-linked
- **缺口 / gaps:** Visually verify the final compiled SI placement and keep the endpoint data scope tied to the exact six-system manifest.

## REV1-ARTICLE — Revision 1 article/SI audited figure assets

- **Members (5 logical outputs):**
  - `paper/revision1/figures/fig6_shap_revised` (recent; traceable)
  - `paper/revision1/figures/figS_dft_neb_mep_profiles` (recent; traceable)
  - `paper/revision1/figures/figS_latent_robustness` (recent; traceable)
  - `paper/revision1/figures/figS_neb_pes_local_audit` (recent; traceable)
  - `paper/revision1/figures/fig_md_msd_kappa_revised` (recent; traceable)
- **文章位置 / article target:** Exact locations are defined by `sn-article-final.tex` and `sn-article-SI.tex`
- **证据链 / evidence chain:** Revision1 processed tables/manifests and source ledger -> generator scripts or legacy source assets -> checked-in PDF/SVG/PNG figure -> TeX include/caption/label -> compiled article/SI.
- **Caption / 图注:** The canonical TeX captions are already present for `fig6_shap_revised`, `figS_dft_neb_mep_profiles`, `figS_latent_robustness`, `figS_neb_pes_local_audit`, and `fig_md_msd_kappa_revised`; the source ledger tracks claim-level evidence for SHAP, latent scores, DFT candidate barrier, and transport metrics.
- **Results / 结果描述:** The Revision1 text correctly distinguishes pointwise accuracy, fixed-path scoring, self-consistent MLFF-NEB, transport re-analysis, and latent/SHAP diagnostics. Several displayed values remain candidate/provisional and are explicitly caveated.
- **Discussion / 分析讨论:** This is the article-facing source of truth. New current-checkout diagnostic figures should be promoted only after their evidence package is linked to a specific TeX section and the source ledger is extended.
- **当前状态 / status:** article-facing and mostly traceable; not all assets have a co-located generator
- **缺口 / gaps:** Run/extend `audit_manuscript_sources.py` for the complete figure set; fill missing generator links for latent/NEB/transport assets and reconcile duplicate `figures/publication` copies.

## REV1-COMPILED — Revision1 compiled manuscript/SI PDFs

- **Members (2 logical outputs):**
  - `paper/revision1/sn-article-SI` (recent; compiled document)
  - `paper/revision1/sn-article-final` (recent; compiled document)
- **文章位置 / article target:** Whole Revision1 documents
- **证据链 / evidence chain:** Revision1 TeX -> compiled PDF.
- **Caption / 图注:** Not a standalone figure.
- **Results / 结果描述:** Used for placement/compilation QA only.
- **Discussion / 分析讨论:** Do not count as figure evidence.
- **当前状态 / status:** compiled document
- **缺口 / gaps:** None at figure level.

## REV1-FOUNDATION — Revision 1 grouped foundation fixed-path profiles

- **Members (2 logical outputs):**
  - `paper/revision1/figures/foundation_grouped/path_1-4_energy_profiles_corrected` (recent; traceable)
  - `paper/revision1/figures/foundation_grouped/path_1-4_energy_profiles_family_aggregated` (recent; traceable)
- **文章位置 / article target:** SI Detailed Migration Pathway Analysis → In-Gap Diffusion (`fig:foundation_profiles`)
- **证据链 / evidence chain:** `data/processed/foundation_grouped/dft_1-4_neb_energies.csv` + ten model JSON profiles -> `scripts/plot_foundation_neb_profiles.py` -> individual and family-aggregated PDF/SVG/PNG -> SI caption.
- **Caption / 图注:** a, ten foundation checkpoints on the same five DFT-path images; b, family means with pointwise ±1 SD bands. The DFT reference is a provisional 0.34 eV candidate and the profiles are fixed-image single-point evaluations, not model-relaxed NEB.
- **Results / 结果描述:** The four representatives show pathway-1–4 barriers of 1.50, 1.79, 1.60, and 1.84 eV against the provisional 0.34 eV reference; family profiles are systematically elevated and smooth.
- **Discussion / 分析讨论:** This is a clean example of fixed-path model-response evidence, but its physical barrier interpretation remains limited by the unconverged DFT candidate.
- **当前状态 / status:** recent, strong local chain
- **缺口 / gaps:** Confirm source tables are the exact versions used by the checked-in figure and retain candidate-DTF wording in the SI.

## TASK2-ROSTER — Cross-element NEB/profile and three-seed stability family

- **Members (10 logical outputs):**
  - `analysis/task2_cross_element/figures/placeholder_dft_gate_status` (recent; traceable)
  - `analysis/task2_cross_element/figures/provisional_task2_neb_model_profiles_9panel` (recent; traceable)
  - `analysis/task2_cross_element_broad_roster_20261003/figures/placeholder_dft_gate_status` (recent; traceable)
  - `analysis/task2_cross_element_broad_roster_20261003/figures/task2_broad_roster_diagnostic` (recent; traceable)
  - `analysis/task2_cross_element_broad_roster_20261003/figures/task2_broad_roster_profiles` (recent; traceable)
  - `analysis/task2_cross_element_broad_roster_20261003/figures/task2_broad_roster_profiles_9panel` (recent; traceable)
  - `analysis/task2_cross_element_broad_roster_final_20261006/figures/task2_three_seed_profiles` (recent; traceable)
  - `analysis/task2_cross_element_broad_roster_final_20261006/figures/task2_three_seed_stability` (recent; traceable)
  - `analysis/task2_cross_element_broad_roster_final_20261006/figures/task2_three_seed_validation` (recent; traceable)
  - `analysis/task2_cross_element/figures/placeholder_dft_gate_status` (recent; traceable)
- **文章位置 / article target:** Results: cross-element transfer, model coverage, and seed stability
- **证据链 / evidence chain:** Ten saved path inputs and DFT provenance manifest -> model/provenance manifests and validation/profile tables -> parent renderers (`render_task2_broad_roster_figures.py`, `render_three_seed_task2_figures.py`) -> provisional, broad-roster, and final three-seed figure sets -> bilingual manuscript/Obsidian notes.
- **Caption / 图注:** Parent caption: same-geometry QE path-table overlays, model profiles, seed/path stability, and epoch-999 validation. QE curves are retained as provisional path references; seed spread is diagnostic model sensitivity, not DFT uncertainty.
- **Results / 结果描述:** The final package contains 1,200 profile rows, 120 summaries, and explicit three-seed stability tables; mean path-wise seed spread is large enough that seed-invariant transferability is not established.
- **Discussion / 分析讨论:** Use the series to separate element coverage, profile shape, and seed sensitivity. Do not call any `<0.1 eV` screen or `JOB DONE` path a converged DFT barrier; current DFT scientific gate remains closed.
- **当前状态 / status:** recent, traceable package with scientific gate blocked
- **缺口 / gaps:** Remove/mark provisional and placeholder renders in the manuscript-facing index; integrate only the final three-seed family after confirming model hashes and DFT gate wording.

## TASK3-NEB — Endpoint-only and Cr-61/Cr-81 path diagnostic hierarchy

- **Members (47 logical outputs):**
  - `analysis/task3_endpoint_only_neb/endpoint_initialization_smoke` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/all_dft_foundation_models/figure_corrected_dft_foundation_barrier_comparison` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/all_dft_foundation_models/figure_corrected_dft_foundation_path_profiles` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/all_path_diagnostic_framework/figure_diagnostic_framework_overview` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/all_path_diagnostic_framework/figure_path_type_summary` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/all_path_structural_diagnostics/figure_structural_stage_diagnostics` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/corrected_dft_foundation_models/figure_corrected_dft_foundation_barrier_comparison` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/corrected_dft_foundation_models/figure_corrected_dft_foundation_path_profiles` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/cr_61_81_three_class_20261003/publication_figures_20261003/Cr61_gap_to_QL/path_1_2` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/cr_61_81_three_class_20261003/publication_figures_20261003/Cr61_gap_to_QL/path_1_3` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/cr_61_81_three_class_20261003/publication_figures_20261003/Cr61_gap_to_QL/path_1_4` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/cr_61_81_three_class_20261003/publication_figures_20261003/Cr61_gap_to_QL/path_1_5` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/cr_61_81_three_class_20261003/publication_figures_20261003/Cr61_gap_to_QL/path_1_7` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/cr_61_81_three_class_20261003/publication_figures_20261003/Cr81_grouped/path_neb_1` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/cr_61_81_three_class_20261003/publication_figures_20261003/Cr81_grouped/path_neb_2` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/cr_61_81_three_class_20261003/publication_figures_20261003/Cr81_grouped/path_neb_3` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/cr_61_81_three_class_20261003/publication_figures_20261003/Cr81_grouped/path_neb_4` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/cr_61_81_three_class_20261003/publication_figures_20261003/Cr81_grouped/path_neb_5` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/cr_61_81_three_class_20261003/publication_figures_20261003/figure_Cr61_Cr81_geometry_embedding` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/cr_61_81_three_class_20261003/publication_figures_20261003/figure_Cr61_Cr81_unified_diagnostic_framework` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/cr_61_81_three_class_20261003/publication_figures_20261003/figure_Cr61_gap_to_QL_all_foundation_profiles` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/cr_61_81_three_class_20261003/publication_figures_20261003/figure_Cr81_grouped_all_foundation_profiles` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/cr_61_81_three_class_20261003/publication_figures_20261003/figure_Cr81_in_gap_diffusion_foundation_profile` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/diagnostic_framework/figure_diagnostic_framework_overview` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/figure1_task_type_matrix/figure1_task_type_matrix` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/full_foundation_models/figure_all_foundation_model_path_profiles` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/full_foundation_models/figure_all_foundation_model_spread` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/neb_convergence_audit/figure_neb_residual_audit` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/processed/figures/figure_1_stability_parity_perturbation` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/processed/figures/figure_2_relative_energy_profiles` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/structural_diagnostics/figure_structural_stage_diagnostics` (recent; partial)
  - `analysis/task3_endpoint_only_neb/endpoint_initialization_smoke` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/all_dft_foundation_models/figure_corrected_dft_foundation_barrier_comparison` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/all_dft_foundation_models/figure_corrected_dft_foundation_path_profiles` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/all_path_diagnostic_framework/figure_diagnostic_framework_overview` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/all_path_diagnostic_framework/figure_path_type_summary` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/all_path_structural_diagnostics/figure_structural_stage_diagnostics` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/corrected_dft_foundation_models/figure_corrected_dft_foundation_barrier_comparison` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/corrected_dft_foundation_models/figure_corrected_dft_foundation_path_profiles` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/diagnostic_framework/figure_diagnostic_framework_overview` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/figure1_task_type_matrix/figure1_task_type_matrix` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/full_foundation_models/figure_all_foundation_model_path_profiles` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/full_foundation_models/figure_all_foundation_model_spread` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/neb_convergence_audit/figure_neb_residual_audit` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/processed/figures/figure_1_stability_parity_perturbation` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/processed/figures/figure_2_relative_energy_profiles` (recent; partial)
  - `analysis/task3_endpoint_only_neb/exploratory_energy_audit/structural_diagnostics/figure_structural_stage_diagnostics` (recent; partial)
- **文章位置 / article target:** Results/SI: path-family diagnostics, local model disagreement, and DFT-reference boundary
- **证据链 / evidence chain:** Path inputs, model registry, source inventories, and QE path tables -> processed image/profile/error tables and Cr-61/Cr-81 unified diagnostic dataset -> parent renderers (`plot_paper_figures.py`, `plot_cr_61_81_*`, `plot_*diagnostics.py`) -> 31 logical outputs, including 15 per-path children and multiple summary parents -> caption/discussion markdown.
- **Caption / 图注:** Use nested naming: `TASK3-NEB.61` (five Cr-61 gap→QL paths), `TASK3-NEB.81G` (five Cr-81 grouped/deep-penetration paths), `TASK3-NEB.81I` (additional in-gap candidate), `TASK3-NEB.S` (stability/profile summaries), and `TASK3-NEB.U` (unified diagnostic/embedding). Child captions identify the exact path and retain QE points plus ten model profiles.
- **Results / 结果描述:** The Cr-61 family is heterogeneous with multi-eV-to-tens-of-eV Foundation deviations on paths 1-2/1-3/1-5 and smaller but nonzero discrepancies on 1-4/1-7. The Cr-81 grouped family is less extreme but path/model dependent; the recovered in-gap candidate is only a last-complete-table reference. Geometry-only grouped-CV AUROC is 0.566 at 0.2 eV and 0.597 at 0.5 eV.
- **Discussion / 分析讨论:** The figures support consistent fixed-geometry model-response diagnostics and path-family separation, not certified barrier rankings. All Cr paths remain exploratory/unconverged/SCF-warning under the current DFT gate; PCA is not a training-distribution coverage test.
- **当前状态 / status:** recent, parent evidence chain traceable; per-path exports need child-level closure
- **缺口 / gaps:** Attach each per-path child to the exact processed CSV rows, hash the child input table, and recover the plotting invocation. Keep FT-600K model-coverage failure separate from energetic disagreement.

