# MigrationBench figure audit summary — 2026-10-08

## Scope and counting rule

The audit scans the current `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench`
checkout, the related isolated ablation worktree
`/Users/alina/.codex/worktrees/bf6e/Wiley-MigrationBench`, plus the canonical
paper/repro checkout `/Users/alina/Project/migrationbench-paper-repro`. The recency cutoff is
2026-08-08. PNG, SVG, PDF, and other exports sharing the same path stem are
one logical figure; an export format is not a separate scientific result.

Current inventory:

- 204 logical figure outputs across 28 named groups;
- 425 rendered exports;
- 336 unique rendered-output hashes; 83 duplicate-hash groups / 172 duplicate export files are retained as branch/format provenance rather than separate scientific results;
- 173 logical outputs modified since 2026-08-08;
- 102 evidence-bearing candidate directories, of which 52 have no rendered descendant output;
- seven new Plot Atlas table-backed evidence packages under `generated/table_backed_packages/`;
- the seven generated packages are intentionally excluded from the 204/425 source inventory to prevent the audit from counting its own outputs; they add 7 logical composites and 21 export files in the separate audit layer;
- 6 compiled manuscript/response PDFs, which are not independent figures.

The per-output paths, timestamps, SHA256 values, and nearby evidence files are
in `figure_inventory.tsv`; the clickable every-source-figure gallery is in
`figure_gallery.md`; the article mapping and gap fields are in
`figure_evidence_matrix.tsv`; the readable parent/child register is in
`figure_group_register.md`; the table-backed candidate queue is in
`figure_candidate_register.md`; duplicate hashes are in
`duplicate_output_hashes.tsv`.

## Priority naming hierarchy

Use these parent IDs in manuscript notes and filenames, followed by semantic
child IDs where needed:

- `FIG-1-DIFFUSION`: context → method → result framing figure.
- `MD-1NS.A–D`: evidence-aware, final diffusion, strict six-axis replica, and
  Overleaf/template variants. Old single panels are `MD-1NS.L*`.
- `MD-ABLATION-20PS`: independent from-scratch ablation worktree family;
  distinguish screening-only 20 ps renders from pending 1 ns production.
- `P1-FORCE.a–c`: same-geometry force disagreement, Cr-local force, and
  geometry/step diagnostics.
- `P2-HANDOFF.a–b`: aligned short-handoff traces and stability gate; retry2
  renders are superseded children, not new figures.
- `TASK2-ROSTER`: provisional/broad-roster/final-three-seed branches; use
  `TASK2-ROSTER.final.*` for manuscript-facing children.
- `TASK3-NEB.61.*`, `TASK3-NEB.81G.*`, `TASK3-NEB.81I.*`, and
  `TASK3-NEB.U.*`: Cr-61 gap→QL, Cr-81 grouped, Cr-81 in-gap candidate, and
  unified diagnostics. Individual path plots are children of the family, not
  separate unrelated figures.
- `REV1-FOUNDATION`, `REV1-APPENDIX`, and `REV1-ARTICLE`: existing Revision 1
  article-facing assets with explicit TeX destinations.
- `ARCHIVE-*` and `LEGACY-*`: historical/source derivatives; never promote
  automatically to current article figures.

## Article mapping

The current Revision 1 article source is the mapping authority:

| Figure family | Proposed article destination | Current status |
|---|---|---|
| `FIG-1-DIFFUSION` | Introduction → Results transition; candidate main figure | Candidate; not yet wired to revision1 TeX |
| `MD-1NS` | Results and Discussions: equilibrium/dynamic stability and diffusion; SI transport candidate | Strong local package; article integration pending; HFACF/κ unavailable |
| `P1-FORCE` | Results: model disagreement and trajectory-integrity boundary; possibly diagnostic SI | Complete current analysis; no TeX include yet |
| `P2-HANDOFF` | Results: short-handoff mechanism and stability gate | Complete current analysis; keep DFT gate closed |
| `TASK2-ROSTER` | Results/SI: cross-element transfer, coverage, and seed stability | Final branch traceable; DFT scientific gate still blocked |
| `TASK3-NEB` | SI Detailed Migration Pathway Analysis, In-Gap Diffusion, Deep Penetration | Parent chain present; per-path child closure remains |
| `ENDPOINT-MACE` | Results or SI endpoint-robustness appendix | Traceable; final placement undecided |
| `REV1-FOUNDATION` | SI `fig:foundation_profiles` | Article-linked and source-scripted |
| `REV1-APPENDIX` | SI endpoint perturbation appendix | Article-linked and source-scripted |
| `REV1-ARTICLE` | Exact `sn-article-final.tex` / `sn-article-SI.tex` locations | Existing TeX captions and ledger; audit still has ledger failures |
| `LEGACY-SHAP` | Sec. 3.5 / Fig. 6 only through revised SHAP chain | Historical images are not current evidence |

The manuscript section names and existing figure labels are in
`/Users/alina/Project/migrationbench-paper-repro/paper/revision1/sn-article-final.tex`
and `sn-article-SI.tex`. New current-checkout figures are not assumed to be in
the manuscript merely because they have a publication-style filename.

## Most important evidence findings

1. The MACE 1 ns family is the most complete recent figure package. It has
   local derived CSV/JSON inputs, multiple renderers, manifests, captions, and
   output hashes. The final diffusion benchmark is scientifically usable only
   for the stable FT-600K and FT-MultiT branches; Foundation-OMAT and Scratch
   remain explicit failure evidence. Thermal conductivity is unavailable from
   the archived heat-flux data.
2. Phase 1 and Phase 2 provide a coherent same-geometry → short-handoff chain.
   The collapsed 0.629 Å frame and the 0 accepted formal DFT gate are
   boundaries, not footnotes.
3. Task 2 and Task 3 have rich profile/table/manifest chains, but their QE
   references are provisional or exploratory. No figure in those branches may
   be described as a converged DFT barrier benchmark without a new scientific
   gate pass.
4. Revision 1 already contains strong article-facing chains for the endpoint
   appendix and grouped foundation profiles. The source ledger and TeX captions
   are the current paper authority.
5. The archived NEB/path/profile and legacy manuscript images often have a
   table or manifest but no co-located plotting script. They are marked
   `partial`, `RECONSTRUCTION`, or `REPRODUCED_SCOPE_LIMITED` until the exact
   renderer, input order, units, and hashes are recovered.
6. The previous two-root inventory was incomplete: the isolated ablation
   worktree contains 30 additional logical figure outputs / 84 exports,
   including the from-scratch 20 ps screening and several Task3/Phase1/endpoint
   branches. These are now included and branch provenance is retained.
7. Table-backed analyses are now explicit candidates rather than invisible
   non-figures. Seven candidates have additive Plot Atlas packages; the
   remaining 52 unrendered candidates are listed with their evidence files and
   a render/no-render recommendation in `figure_candidate_register.md`.

## Explicit gaps to close

- Add a data manifest for `FIG-1-DIFFUSION`; its quantitative values are
  currently embedded in the plotting script rather than linked to input tables.
- Recover the plotting command for the two MD time-convergence PNGs.
- Attach every Task 3 per-path child to exact processed rows, path/image hashes,
  and the parent renderer invocation.
- Designate one canonical Phase 2 render and mark retry2/duplicate checkout
  outputs as superseded.
- Reconcile duplicate hashes across the Wiley and paper-repro checkouts before
  counting them as separate figures.
- Extend/fix the Revision 1 source ledger audit. The current read-only run of
  `revision1/scripts/audit_manuscript_sources.py` reports nine ledger claim
  tokens not found in current manuscript/table text, including SHAP CV values,
  the 0.336 eV DFT candidate, the 0.688 eV fixed-path error, and latent-space
  scores. This is a paper-source consistency gap, not evidence that the values
  are scientifically invalid.
- For all legacy image-only assets, recover raw data → processing → plotting
  provenance or mark the figure unavailable for current publication use.
- Review the 52 table-backed/unrendered candidates one by one. Do not render
  numerical scientific panels from schema-only, manifest-only, pending, or
  missing-output campaigns; use gate/status figures where the status itself is
  the evidence.

## Verification performed

The inventory was regenerated deterministically, the four audit helpers pass
Python compilation, `git diff --check` is clean for tracked changes, and output
SHA256 hashes were recorded. Duplicate hashes were explicitly detected for
Phase 2 renders, Task 2 placeholders, and duplicated Revision 1 publication
assets. The Scientific Plot Atlas registry/catalog and independent-panel
contract were read before classifying the plotting chains.

The candidate inventory/register were also regenerated. The Plot Atlas renderer
created seven additive packages, each with PNG/SVG/PDF output, a processed table,
source/output SHA256 manifest, and caption/results/discussion text. The seven new
PNG figures were visually inspected at high detail; they are readable and
consistent with their source tables, while their gate/screening boundaries are
preserved in the package manifests.

### Visual sanity check (2026-10-08)

Five representative recent renders were opened at high detail and were
non-empty, internally coherent, and visually consistent with their local
manifests and text:

- `analysis/mace_1ns_publication_figure_20261005/figures/mace_1ns_final_diffusion_benchmark.png`: six-panel MD benchmark with visible force/log-scale failure modes, MSD exclusions, block fits, and VACF.
- `analysis/phase1_static_force_comparison_20261002/figures/fig1_force_disagreement.png`: four-step force-disagreement series with the expected collapse and force-spread growth.
- `analysis/phase2_handoff_md_20261002/audit_20261003/figures/phase2_handoff_timeseries.png`: thermal, force, contact, and structural handoff traces with runaway trajectories visible.
- `analysis/task2_cross_element_broad_roster_final_20261006/figures/task2_three_seed_profiles.png`: provisional QE/NEB reference and three-seed profiles, including the SCF-warning boundary annotation.
- `analysis/task3_endpoint_only_neb/exploratory_energy_audit/cr_61_81_three_class_20261003/publication_figures_20261003/figure_Cr61_Cr81_unified_diagnostic_framework.png`: unified 61/81 diagnostic with heatmap, stage error, geometry warning, and grouped-CV AUROC values.

This visual check confirms that the selected evidence chains produce actual
readable figures; it does not upgrade any scientific status. The per-group
register and per-output matrix remain authoritative for unresolved provenance
and acceptance gaps.
