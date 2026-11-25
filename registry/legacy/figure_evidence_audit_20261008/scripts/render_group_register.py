#!/usr/bin/env python3
"""Render the curated logical-figure group register from the inventory matrix."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/figure_evidence_audit_20261008")

GROUPS = {
    "MD-ABLATION-20PS": {
        "name": "From-scratch ablation MD screening and transport diagnostic",
        "article": "Results/SI: training-family ablation, short-trajectory stability, and transport-screen boundary",
        "chain": "Isolated worktree `/Users/alina/.codex/worktrees/bf6e/Wiley-MigrationBench/analysis/from_scratch_ablation_md_20261005` -> remote registry/status/checkpoint/trajectory artifacts -> screening/MSD/Dxy tables -> Plot Atlas-compatible screening and unified comparison renderers -> `figures_plot_atlas`, `figures_unified_atlas_20ps`, and remote audit figures.",
        "caption": "Parent caption: short-screen trajectory completeness, stability/failure state, MSD/Dxy screening, and model-family comparison. Each panel names its duration, accepted/failed branch, atom/PBC identity, and whether it is a transport screen or a formal transport estimate.",
        "result": "The worktree contains an independent from-scratch ablation family with explicit screening figures and failed/partial branches retained. A rendered screening figure is evidence of the analysis chain, not proof of a complete 1 ns or converged transport result.",
        "discussion": "Use this family to separate training-family effects from numerical/runtime failures. Do not pool incomplete or unstable trajectories into formal MSD/Dxy/HFACF/κ claims; retain training seed and MD velocity seed as separate provenance fields.",
        "status": "recent, traceable worktree package; branch-isolated and partly screening-only",
        "gaps": "Link each render to the exact status/checkpoint/trajectory manifest and designate the worktree branch as source-of-truth or superseded relative to current-checkout duplicates; no automatic promotion to the manuscript.",
    },
    "FIG-1-DIFFUSION": {
        "name": "Diffusion-aware evaluation / context–method–result composite",
        "article": "Introduction/Results transition; candidate main Fig. 1",
        "chain": "`figure1_nature.py` -> hard-coded audited summary values -> `figure1_diffusion_aware_evaluation.{pdf,svg,png}` -> `figure1_caption.md`.",
        "caption": "a, evaluation gap schematic; b, three-level diffusion-aware evaluation hierarchy; c, same-geometry MD geometry/force-integrity warning; d, exploratory fixed-geometry path sampling. The schematic is conceptual, and the quantitative panels are explicitly not model rankings or converged NEB barriers.",
        "result": "The source caption records a drop in minimum pair distance from 2.393 to 0.629 Å and a rise in model force spread from 1.29 to 4,565.24 eV Å⁻¹, plus Foundation/MPA0 median profile and barrier diagnostics over eight screened paths.",
        "discussion": "This is a useful framing figure: pointwise agreement is necessary but does not close dynamical or path-level reliability. Panel c must remain a warning about a nonphysical collapsed frame, not a model-accuracy ranking.",
        "status": "candidate, traceable to a renderer and caption but panel c/d values are embedded constants rather than linked input tables",
        "gaps": "Add a data manifest linking the hard-coded values to Phase-1 and processed-NEB CSVs; map the figure to the current revision1 TeX labels before calling it main Fig. 1.",
    },
    "MD-1NS": {
        "name": "MACE 1 ns benchmark family",
        "article": "Results: dynamic stability and diffusion; SI transport-method/results candidate; current main-text integration is pending",
        "chain": "Remote four-model archive -> local `data/*/{audit,metrics,msd,timeseries,vacf}.(json,csv)` and derived `data/thermo/*` -> `plot_figure.py`, `plot_final_benchmark.py`, `plot_strict_replica.py`, `plot_overleaf_panels.py` plus recomputation helpers -> three manifests -> 9 logical render variants.",
        "caption": "Parent caption: six-model-axis or evidence-aware benchmark at 600 K for 2,050 atoms, with temperature, pressure, stability/failure, MSD, VACF and explicitly qualified transport axes. The final benchmark removes HFACF and κ because no heat-flux time series is archived; the strict replica retains those slots only as `NOT AVAILABLE`.",
        "result": "FT-600K and FT-MultiT are stable for the archived diffusion analysis; Foundation-OMAT crosses a thermal threshold at 8.8 ps and from-scratch at 462.2 ps. The final package reports the stable models for diffusion and retains failures as evidence.",
        "discussion": "Use the final diffusion benchmark for diffusion/stability claims. Do not claim thermal conductivity from the strict replica. Training seeds and MD velocity seeds remain separate, and identical velocity seeds are not independent replicas.",
        "status": "recent, strongest current evidence package; scientific status is model-specific, not uniformly successful",
        "gaps": "Integrate the accepted variant into revision1 text/captions; preserve the explicit absence of HFACF/κ and verify the remote-to-local source hashes if the archive changes.",
    },
    "MD-1NS-LEGACY": {
        "name": "Legacy continuation-MD panels",
        "article": "SI transport/audit appendix only until lineage is reconciled with MD-1NS",
        "chain": "`md_continuation_20260930/analysis_publication_20261003/make_publication_artifacts.py` and local audit JSON -> four figure stems (`fig_stability`, `fig_msd`, `fig_vacf`, `fig_from_scratch_failure`).",
        "caption": "Stability, MSD, VACF, and from-scratch failure panels from the continuation-MD audit; each caption must name the exact trajectory segment and whether the value is a diagnostic or an accepted transport estimate.",
        "result": "The outputs preserve stable/failed branches and show why a failed from-scratch branch cannot be pooled into a formal 1 ns transport estimate.",
        "discussion": "These are useful audit/legacy panels but should not compete with MD-1NS final figures. They need a direct source-trajectory hash and exact window definition before publication reuse.",
        "status": "recent but legacy/duplicate family",
        "gaps": "Record exact raw trajectory paths, derived-window inputs, and a superseded-by relationship to MD-1NS.",
    },
    "MD-CONV": {
        "name": "MD time-window convergence",
        "article": "SI transport-method validation",
        "chain": "`blocks_50ps.csv`, `cumulative_windows.csv`, `time_convergence.json` -> PNG outputs; the plotting script is not co-located in the figure directory.",
        "caption": "Time-window convergence of block statistics for the from-scratch and multi-temperature trajectories, showing the selected analysis window and whether the estimator has stabilized.",
        "result": "The figure is a convergence diagnostic, not a transport result by itself; it should justify the window used by the accepted MD package.",
        "discussion": "Keep this as a method-support panel. A convergence-looking trace does not repair a trajectory that fails the physical stability gate.",
        "status": "partial evidence chain",
        "gaps": "Recover or identify the exact plotting script and commit the command/configuration used to produce the PNGs.",
    },
    "ENDPOINT-MACE": {
        "name": "Four-model endpoint relaxation robustness",
        "article": "Results: endpoint robustness / local relaxation sensitivity",
        "chain": "Raw model summary JSONL and relaxed extxyz archives -> `tidy_endpoint_relaxation_results.csv` plus six summary tables -> `analyze_and_plot.py` -> `fig1_endpoint_robustness` and `fig2_relative_energy_landscape` -> `discussion_draft.md`.",
        "caption": "A, convergence/effort and paired final-geometry robustness across 47 endpoints, three starting variants, and four models. B, model-internal relative-energy landscapes and endpoint-type dependence; energy zero points are not compared across models.",
        "result": "All models except three Scratch runs at `unique_151` satisfy the local relaxation gate; paired Cr displacements from FT-600K have medians of 0.225–0.284 Å, with only 3.5–15.6% within 0.10 Å.",
        "discussion": "The figure supports basin-sensitivity screening and DFT-priority selection, not model-independent endpoint geometries or cross-model absolute-energy ranking.",
        "status": "traceable current diagnostic",
        "gaps": "Confirm whether this remains in the journal revision or is an internal endpoint appendix; preserve three Scratch failures in any final table.",
    },
    "P1-FORCE": {
        "name": "Phase 1 same-geometry static force comparison",
        "article": "Results: model disagreement and trajectory-integrity boundary; candidate Fig. 1c/diagnostic SI",
        "chain": "Common extxyz snapshots and `snapshot_manifest.json` -> NPZ model evaluations -> `analysis/{model_step_summary,pairwise_force_comparison,atom_force_disagreement,geometry_diagnostics}.csv` -> `scripts/plot_phase1.py` -> 3 export variants -> `Results.md`.",
        "caption": "a, pairwise force disagreement and leave-one-out force RMSE across the four models; b, Cr-local force response and maximum atom spread; c, minimum distance and local-shell diagnostics across the shared snapshots.",
        "result": "The minimum pair distance reaches 0.629 Å and maximum force spread reaches 4,565.24 eV Å⁻¹ at step 462200. The collapsed frame is excluded from formal DFT/diffusion/barrier statistics.",
        "discussion": "The figure establishes a model-disagreement/trajectory-integrity warning. It does not identify the causal integrator/thermostat mechanism and does not rank model accuracy.",
        "status": "recent, fully traceable analysis package",
        "gaps": "No article include/caption is yet wired to these outputs; the paper-repro duplicate must be reconciled with the current checkout.",
    },
    "P2-HANDOFF": {
        "name": "Phase 2 short-handoff stability and gate family",
        "article": "Results: dynamic stability mechanism; SI or main diagnostic figure",
        "chain": "Handoff snapshots and submission manifests -> 1,616-row `frame_aligned_metrics.csv` plus hash/anomaly/neighborhood audits -> `audit_phase2.py` and `render_phase2_atlas.py` -> canonical audit figures; retry2 PNG/SVGs are linked as superseded remote artifacts.",
        "caption": "a, frame-aligned temperature/energy/force/geometry traces for both input states and four models, with first-gate markers; b, model × input-state stability summary and first crossing steps.",
        "result": "Foundation-OMAT and from-scratch run away within 0.02–0.32 ps, while naive-FT and Multi-T remain finite for the short handoff; the first anomalous local force excursion precedes later contact/thermal runaway.",
        "discussion": "This supports model-dependent local-PES amplification on identical initial states. It does not prove the initial state is a DFT minimum or identify the original long-trajectory mechanism; the 0 accepted formal DFT gate remains visible.",
        "status": "recent, fully traceable with duplicate outputs",
        "gaps": "Designate `audit_20261003/figures` as canonical, record retry2 as superseded, and map the completed integrator sensitivity package into the paper if the stronger mechanism claim is used.",
    },
    "TASK2-ROSTER": {
        "name": "Cross-element NEB/profile and three-seed stability family",
        "article": "Results: cross-element transfer, model coverage, and seed stability",
        "chain": "Ten saved path inputs and DFT provenance manifest -> model/provenance manifests and validation/profile tables -> parent renderers (`render_task2_broad_roster_figures.py`, `render_three_seed_task2_figures.py`) -> provisional, broad-roster, and final three-seed figure sets -> bilingual manuscript/Obsidian notes.",
        "caption": "Parent caption: same-geometry QE path-table overlays, model profiles, seed/path stability, and epoch-999 validation. QE curves are retained as provisional path references; seed spread is diagnostic model sensitivity, not DFT uncertainty.",
        "result": "The final package contains 1,200 profile rows, 120 summaries, and explicit three-seed stability tables; mean path-wise seed spread is large enough that seed-invariant transferability is not established.",
        "discussion": "Use the series to separate element coverage, profile shape, and seed sensitivity. Do not call any `<0.1 eV` screen or `JOB DONE` path a converged DFT barrier; current DFT scientific gate remains closed.",
        "status": "recent, traceable package with scientific gate blocked",
        "gaps": "Remove/mark provisional and placeholder renders in the manuscript-facing index; integrate only the final three-seed family after confirming model hashes and DFT gate wording.",
    },
    "TASK3-NEB": {
        "name": "Endpoint-only and Cr-61/Cr-81 path diagnostic hierarchy",
        "article": "Results/SI: path-family diagnostics, local model disagreement, and DFT-reference boundary",
        "chain": "Path inputs, model registry, source inventories, and QE path tables -> processed image/profile/error tables and Cr-61/Cr-81 unified diagnostic dataset -> parent renderers (`plot_paper_figures.py`, `plot_cr_61_81_*`, `plot_*diagnostics.py`) -> 31 logical outputs, including 15 per-path children and multiple summary parents -> caption/discussion markdown.",
        "caption": "Use nested naming: `TASK3-NEB.61` (five Cr-61 gap→QL paths), `TASK3-NEB.81G` (five Cr-81 grouped/deep-penetration paths), `TASK3-NEB.81I` (additional in-gap candidate), `TASK3-NEB.S` (stability/profile summaries), and `TASK3-NEB.U` (unified diagnostic/embedding). Child captions identify the exact path and retain QE points plus ten model profiles.",
        "result": "The Cr-61 family is heterogeneous with multi-eV-to-tens-of-eV Foundation deviations on paths 1-2/1-3/1-5 and smaller but nonzero discrepancies on 1-4/1-7. The Cr-81 grouped family is less extreme but path/model dependent; the recovered in-gap candidate is only a last-complete-table reference. Geometry-only grouped-CV AUROC is 0.566 at 0.2 eV and 0.597 at 0.5 eV.",
        "discussion": "The figures support consistent fixed-geometry model-response diagnostics and path-family separation, not certified barrier rankings. All Cr paths remain exploratory/unconverged/SCF-warning under the current DFT gate; PCA is not a training-distribution coverage test.",
        "status": "recent, parent evidence chain traceable; per-path exports need child-level closure",
        "gaps": "Attach each per-path child to the exact processed CSV rows, hash the child input table, and recover the plotting invocation. Keep FT-600K model-coverage failure separate from energetic disagreement.",
    },
    "P3-DFT": {
        "name": "Phase 3 DFT recovery/validation diagnostics",
        "article": "Results/SI: DFT gate, force validation, and recovery boundary",
        "chain": "Job ledgers/provenance manifests and ordered-block audits -> MACE reconstruction/force-landscape tables -> figure outputs; formal accepted target values are currently absent.",
        "caption": "Diagnostic force landscape/progression or reconstruction figure tied to the exact snapshot, ordered block, and job ledger; no panel should imply a converged DFT reference unless the explicit gate passes.",
        "result": "The figures document what was tested and where the DFT validation remains incomplete, rather than supplying a finalized target barrier/force value.",
        "discussion": "Use these to explain a blocked DFT gate and preserve failed/recovered branches; do not substitute local-tile or cancelled-array evidence for identity-preserving full-cell DFT.",
        "status": "partial, scientific gate blocked",
        "gaps": "Link each output to its exact Slurm job, raw QE input/output, and acceptance report; keep no-accepted-target status in any caption.",
    },
    "REV1-ARTICLE": {
        "name": "Revision 1 article/SI audited figure assets",
        "article": "Exact locations are defined by `sn-article-final.tex` and `sn-article-SI.tex`",
        "chain": "Revision1 processed tables/manifests and source ledger -> generator scripts or legacy source assets -> checked-in PDF/SVG/PNG figure -> TeX include/caption/label -> compiled article/SI.",
        "caption": "The canonical TeX captions are already present for `fig6_shap_revised`, `figS_dft_neb_mep_profiles`, `figS_latent_robustness`, `figS_neb_pes_local_audit`, and `fig_md_msd_kappa_revised`; the source ledger tracks claim-level evidence for SHAP, latent scores, DFT candidate barrier, and transport metrics.",
        "result": "The Revision1 text correctly distinguishes pointwise accuracy, fixed-path scoring, self-consistent MLFF-NEB, transport re-analysis, and latent/SHAP diagnostics. Several displayed values remain candidate/provisional and are explicitly caveated.",
        "discussion": "This is the article-facing source of truth. New current-checkout diagnostic figures should be promoted only after their evidence package is linked to a specific TeX section and the source ledger is extended.",
        "status": "article-facing and mostly traceable; not all assets have a co-located generator",
        "gaps": "Run/extend `audit_manuscript_sources.py` for the complete figure set; fill missing generator links for latent/NEB/transport assets and reconcile duplicate `figures/publication` copies.",
    },
    "REV1-FOUNDATION": {
        "name": "Revision 1 grouped foundation fixed-path profiles",
        "article": "SI Detailed Migration Pathway Analysis → In-Gap Diffusion (`fig:foundation_profiles`)",
        "chain": "`data/processed/foundation_grouped/dft_1-4_neb_energies.csv` + ten model JSON profiles -> `scripts/plot_foundation_neb_profiles.py` -> individual and family-aggregated PDF/SVG/PNG -> SI caption.",
        "caption": "a, ten foundation checkpoints on the same five DFT-path images; b, family means with pointwise ±1 SD bands. The DFT reference is a provisional 0.34 eV candidate and the profiles are fixed-image single-point evaluations, not model-relaxed NEB.",
        "result": "The four representatives show pathway-1–4 barriers of 1.50, 1.79, 1.60, and 1.84 eV against the provisional 0.34 eV reference; family profiles are systematically elevated and smooth.",
        "discussion": "This is a clean example of fixed-path model-response evidence, but its physical barrier interpretation remains limited by the unconverged DFT candidate.",
        "status": "recent, strong local chain",
        "gaps": "Confirm source tables are the exact versions used by the checked-in figure and retain candidate-DTF wording in the SI.",
    },
    "REV1-APPENDIX": {
        "name": "Revision 1 endpoint perturbation appendix",
        "article": "SI endpoint-relaxation appendix",
        "chain": "`appendix/data/endpoint_level_summary.csv/json` -> `appendix/scripts/plot_campaign_atlas.py` using Plot Atlas primitives -> three figure outputs -> `endpoint_perturbation_appendix.tex` captions and `multiarms_table.tex`.",
        "caption": "Campaign overview/relaxation outcomes; perturbation response across systems and endpoints; host–dopant interaction and relaxation-effort maps. The figures summarize measured 846-task multi-arm endpoint relaxation, not DFT or magnetic conclusions.",
        "result": "The captions and table define six host–dopant systems, perturbation arms, convergence criterion, final energy/force/displacement/steps, and the 846-task scientific-success gate.",
        "discussion": "This appendix is one of the strongest closed evidence packages in the paper-repro tree because raw compact data, plotting source, table, and TeX placement are co-located.",
        "status": "recent, traceable and article-linked",
        "gaps": "Visually verify the final compiled SI placement and keep the endpoint data scope tied to the exact six-system manifest.",
    },
    "ARCHIVE-NEB-BARRIER": {
        "name": "Archived NEB barrier atlas derivatives",
        "article": "SI DFT NEB barrier inventory / source ledger",
        "chain": "`barrier_atlas.csv/json` -> archived PNG derivative; no co-located plot generator.",
        "caption": "Per-path barrier-atlas summary with path identity, forward barrier, convergence/status flags, and source provenance. Unconverged/diverged paths remain retained but are not final references.",
        "result": "These assets preserve historical path-level values and flags used by the revision audit.",
        "discussion": "They are provenance records, not automatically publication-ready figures. A `JOB DONE` line or an optimizer value does not close SCF/force/NEB/path identity gates.",
        "status": "historical archive, partial chain",
        "gaps": "Recover the renderer and raw-input lineage; associate each child with `figS_dft_neb_mep_profiles` only through the source ledger, not filename similarity.",
    },
    "ARCHIVE-NEB-PROFILE": {
        "name": "Archived NEB energy-profile derivatives",
        "article": "SI DFT NEB profiles / fixed-path benchmark source evidence",
        "chain": "`neb_energy_summary.csv/json` and `neb_energy_snapshot.md` -> archived profile PNG; plotting generator not co-located.",
        "caption": "Relative energy along the exact archived path/images, with path identity and source job retained. Profiles are qualified as candidate or exploratory where the original convergence evidence is incomplete.",
        "result": "The profiles support path-family inventory and fixed-geometry comparison, but do not by themselves support converged barrier ranking.",
        "discussion": "Keep as source evidence; do not merge path families or infer equivalence from similar names.",
        "status": "historical archive, partial chain",
        "gaps": "Recover the plotting script/config and verify exact path/image order, units, endpoints, and hashes against the source ledger.",
    },
    "ARCHIVE-TRAJ-PORTFOLIO": {
        "name": "Archived representative-trajectory coverage/similarity plots",
        "article": "Methods/SI trajectory coverage audit",
        "chain": "Portfolio manifest plus coverage gaps, pairwise distances, and representative path tables -> archived image; generator not co-located.",
        "caption": "Coverage/similarity of the 61- and 81-atom trajectory portfolios, with the exact metric and sampled frame scope named.",
        "result": "These plots document data-selection/coverage decisions rather than model accuracy.",
        "discussion": "Useful for provenance and sampling justification; not a substitute for a formal train/test distribution audit unless the split source is linked.",
        "status": "historical archive, partial chain",
        "gaps": "Recover renderer and source-frame manifest; explicitly state whether the metric is geometry similarity or training-distribution coverage.",
    },
    "ARCHIVE-ENDPOINT": {
        "name": "Archived endpoint-relaxation curves",
        "article": "SI endpoint-relaxation evidence",
        "chain": "Published endpoint manifests/log summaries -> archived force-relaxation curves; renderer not co-located.",
        "caption": "Force relaxation histories for exact endpoint/iteration or geometry-transform comparisons, with model/job/path identity.",
        "result": "These are convergence-history records and should be read with the endpoint gate, not as cross-model energy comparisons.",
        "discussion": "Retain failed/unconverged curves as provenance. A smooth curve without identity and final gate evidence is incomplete.",
        "status": "historical archive, partial chain",
        "gaps": "Recover plot command and raw log hashes; connect each curve to the endpoint table and manuscript claim.",
    },
    "ARCHIVE-SELFNEB": {
        "name": "Historical self-consistent NEB profile",
        "article": "SI NEB stability audit / Fig. S3 lineage",
        "chain": "Historical self-NEB audit derivative -> archived profile image; source/renderer not co-located.",
        "caption": "Self-consistent MLFF-NEB profile, explicitly distinguished from fixed-path DFT scoring; any 0.41 eV value is model-reported on the model-optimized path.",
        "result": "The foundation model completed the archived self-consistent path while other models terminated before a converged band; this is a stability result, not fixed-path accuracy.",
        "discussion": "Preserve the operator distinction emphasized in Reviewer A1 response.",
        "status": "legacy audit derivative, partial chain",
        "gaps": "Link to the exact historical input/log and current SI Fig. S3 asset; do not blend with new fixed-path figures.",
    },
    "ARCHIVE-PRECOND": {
        "name": "Archived MLFF preconditioner residual smoke",
        "article": "SI NEB methods/audit only",
        "chain": "Residual summary -> archived residual plot; source generator not co-located.",
        "caption": "Residual progression for the specified preconditioner smoke test, labeled as a method diagnostic.",
        "result": "Shows numerical preconditioning behavior, not a physical barrier.",
        "discussion": "Keep outside main scientific narrative unless needed to explain the NEB implementation.",
        "status": "partial archive",
        "gaps": "Recover exact command, input, and acceptance rule.",
    },
    "ARCHIVE-LEAKAGE": {
        "name": "Training-leakage distance distribution",
        "article": "Methods/data provenance and reviewer evidence",
        "chain": "`leakage_audit.json`, nearest-neighbor CSV, distance summary -> archived distribution plot; generator is outside the figure directory.",
        "caption": "Distance distributions for the audited checkpoint versus reference/training samples, with split scope and distance definition.",
        "result": "Supports a data-leakage/proximity audit, not downstream task accuracy.",
        "discussion": "Must name the exact source scope and not generalize to unrelated four-model or Dual-X workflows.",
        "status": "partial but data artifacts present",
        "gaps": "Link exact renderer and audit commit; extend source ledger if a manuscript claim uses this plot.",
    },
    "LEGACY-NEURIPS": {
        "name": "Original NeurIPS-era manuscript assets",
        "article": "Original main/SI manuscript, retained for provenance only",
        "chain": "Image-only historical assets referenced by old TeX captions; modern source chain is not uniformly present.",
        "caption": "Use the original manuscript captions only as historical text, and label any reused panel with the current source/data scope.",
        "result": "These figures document earlier benchmark framing and may contain superseded model names, protocols, or transport claims.",
        "discussion": "Do not treat legacy captions as current evidence. The current revision must use the audited source ledger and corrected figures.",
        "status": "legacy/image-first",
        "gaps": "For every reuse candidate, recover raw data, generator, protocol, and current hash; otherwise mark `RECONSTRUCTION` or `REPRODUCED_SCOPE_LIMITED`.",
    },
    "LEGACY-SHAP": {
        "name": "Legacy SHAP/descriptor diagnostics",
        "article": "Original or reviewer-facing interpretability evidence",
        "chain": "Historical image assets and candidate SOAP/SHAP outputs; current canonical chain is `revision1/scripts/shap_analysis.py` + `shap_feature_labels.py` + `plot_shap_fig6.py` + compact inputs.",
        "caption": "Feature-importance, error-distribution, or model-comparison diagnostic with the surrogate target and validation scope explicitly named.",
        "result": "Only the revised cross-validated SHAP chain is article-ready; prior in-sample or image-only panels are superseded unless re-audited.",
        "discussion": "Interpret as surrogate error-predictor sensitivity, not causal MACE internals or physical mechanism.",
        "status": "mixed legacy/current candidates",
        "gaps": "One-to-one map old assets to the revised Fig. 6 and mark all unlinked images as historical.",
    },
    "LEGACY-SI": {
        "name": "Legacy supplementary image assets",
        "article": "Original SI; historical reference",
        "chain": "Image-only or historical supplementary assets; source linkage varies.",
        "caption": "Use original captions only after protocol, model identity, and source data are confirmed.",
        "result": "Not current evidence without re-audit.",
        "discussion": "Keep separate from Revision1 audited SI figures.",
        "status": "legacy/image-first",
        "gaps": "Recover source chain or mark unavailable.",
    },
    "LEGACY-PUBLISHED": {
        "name": "Duplicate packaged publication assets",
        "article": "Revision1 manuscript assets, duplicate location",
        "chain": "Copied/packaged PDF assets without an independent generator.",
        "caption": "Use the caption attached to the canonical Revision1 asset, not a duplicate filename.",
        "result": "No independent result; duplicate must hash-match or be explained.",
        "discussion": "Avoid counting duplicates as separate figures.",
        "status": "duplicate/legacy",
        "gaps": "Hash-compare against canonical revision1 figures and record the source-of-truth path.",
    },
    "COMPILED-DOCUMENT": {
        "name": "Compiled manuscript/response PDFs",
        "article": "Whole document, not an independent figure",
        "chain": "TeX sources -> compiler -> PDF.",
        "caption": "Not applicable at the logical-figure level.",
        "result": "Use only for visual placement verification; individual figures are audited separately.",
        "discussion": "Excluded from figure count in the human audit despite being present in the inventory.",
        "status": "compiled document",
        "gaps": "None as figures; verify included assets via TeX/source audit.",
    },
    "REV1-COMPILED": {
        "name": "Revision1 compiled manuscript/SI PDFs",
        "article": "Whole Revision1 documents",
        "chain": "Revision1 TeX -> compiled PDF.",
        "caption": "Not a standalone figure.",
        "result": "Used for placement/compilation QA only.",
        "discussion": "Do not count as figure evidence.",
        "status": "compiled document",
        "gaps": "None at figure level.",
    },
    "PAPER-OTHER": {
        "name": "Unclassified paper-repro output",
        "article": "Unmapped",
        "chain": "Inventory-only until classified.",
        "caption": "Not drafted until source scope is identified.",
        "result": "Not used for claims.",
        "discussion": "Requires manual review.",
        "status": "unclassified",
        "gaps": "Assign a family and locate source data/script/caption.",
    },
}


def main() -> None:
    rows = list(csv.DictReader((ROOT / "figure_evidence_matrix.tsv").open(encoding="utf-8"), delimiter="\t"))
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)

    lines = [
        "# Figure evidence group register (2026-10-08)",
        "",
        "This is the human-readable layer over `figure_evidence_matrix.tsv`. The matrix has one row per logical figure output (PNG/SVG/PDF siblings are one row); this register gives the parent/child naming, article mapping, caption, result, discussion, and evidence-boundary language. `candidate` means a proposed article placement, not an accepted manuscript change.",
        "",
        "## Naming hierarchy",
        "",
        "Use the group ID as the parent, then a semantic child suffix: for example `MD-1NS.A` = evidence-aware benchmark, `MD-1NS.B` = final diffusion benchmark, `TASK3-NEB.61.1-4` = one Cr-61 path child, and `TASK3-NEB.81G.neb_1` = one Cr-81 grouped child. Export formats never receive separate figure numbers.",
        "",
        "## Inventory coverage",
        "",
        f"The machine inventory contains **{len(rows)} logical figure outputs** across **{len(by_group)} groups**. The exact per-output paths and SHA256 hashes are in `figure_inventory.tsv` and `figure_evidence_matrix.tsv`; this register deliberately keeps compiled manuscript PDFs separate from scientific figures.",
        "",
    ]
    for group_id in sorted(by_group):
        info = GROUPS.get(group_id)
        if info is None:
            info = {"name": group_id, "article": "unmapped", "chain": "inventory only", "caption": "not drafted", "result": "not drafted", "discussion": "not drafted", "status": "unclassified", "gaps": "manual review"}
        lines.extend([
            f"## {group_id} — {info['name']}",
            "",
            f"- **Members ({len(by_group[group_id])} logical outputs):**",
        ])
        for row in by_group[group_id]:
            recent = "recent" if row["recent_since_2026_08_08"] == "True" else "historical"
            lines.append(f"  - `{row['logical_figure']}` ({recent}; {row['evidence_chain_status']})")
        lines.extend([
            f"- **文章位置 / article target:** {info['article']}",
            f"- **证据链 / evidence chain:** {info['chain']}",
            f"- **Caption / 图注:** {info['caption']}",
            f"- **Results / 结果描述:** {info['result']}",
            f"- **Discussion / 分析讨论:** {info['discussion']}",
            f"- **当前状态 / status:** {info['status']}",
            f"- **缺口 / gaps:** {info['gaps']}",
            "",
        ])

    (ROOT / "figure_group_register.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {ROOT / 'figure_group_register.md'} ({len(rows)} members, {len(by_group)} groups)")


if __name__ == "__main__":
    main()
