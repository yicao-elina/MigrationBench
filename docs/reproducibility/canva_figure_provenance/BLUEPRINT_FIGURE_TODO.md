# Atomic Migration blueprint: figure-by-figure TODO and release gates

Audit date: 2026-10-31 (America/Los_Angeles)

## Scope boundary

The supplied PDF (`/Users/alina/Downloads/Atomic_Migration_Figure_Blueprint.pdf`)
and the pasted checklist are treated as design/reference instructions. The
current request is to turn those instructions into an evidence-backed build and
review package using the local data paths catalogued by PR #14. A figure is not
scientifically complete because a Canva page exists, a scheduler job says
`COMPLETED`, or a PDF renders. Release requires the provenance, data, geometry,
and visual gates in `DATA_AND_VISUAL_ACCEPTANCE_CRITERIA.md`.

## Status vocabulary

- `DATA_BACKED_DIAGNOSTIC`: the plotted values come from identified local tables,
  but one or more publication claims remain gated.
- `PROVISIONAL`: the data are directly available and reproducible, but the
  source-unit, split, or coverage contract still needs an explicit audit.
- `BLOCKED`: a required geometry join, convergence record, or campaign coverage
  is absent; the generated red panel is a placeholder, not a result.
- `PUBLICATION_PASS`: only after an independent reviewer signs both data and
  visual gates and the manifest hash is frozen. No figure in this first build is
  promoted to this state.

## Figure 1 — research design and task distribution

Blueprint intent: task-conditioned benchmark, material/data layers, and the
configuration universe.

Current local inputs:

- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/migrationbench_universe_publication_20261026/derived/comprehensive_task_map_points.csv`
- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/migrationbench_universe_publication_20261026/derived/structure_universe_points.csv`
- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/migrationbench_universe_publication_20261026/derived/structure_universe_local_soap_vectors.npz`
- the corresponding `manifest*.json` files under
  `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/migrationbench_universe_publication_20261026`

Build TODO:

1. Freeze one canonical `geometry_id`/`configuration_id` for each point and
   document the task-to-configuration mapping.
2. Add task family, trajectory group, dopant, site, temperature, and split
   fields where the source supports them; do not infer them from PCA position.
3. Recompute counts from the frozen manifest and reconcile the current 2,724
   comprehensive points, 2,652 structure-universe points, and the 72 L1
   thermal points.
4. Keep the workflow schematic as a separate editable artifact; it is not
   evidence for the count plot.
5. Reviewer checks: no synthetic scatter, deterministic embedding seed,
   feature representation/normalization recorded, and every legend label has a
   source row.

Current deliverable: the map plus workflow composition in
`figures/figure01_task_conditioned_universe.*`; status `PARTIAL` until the
canonical IDs and task joins are frozen.

## Figure 2 — near-equilibrium parity and frame error

Blueprint intent: held-out energy/force parity, seed variability, and
distribution-shift checks.

Current local inputs:

- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/rmse_train_test_20261026/parity_summary.csv`
- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/rmse_train_test_20261026/raw/rockfish_parity`
  (the `*_per_frame.csv` files and their paired `*.npz` arrays)
- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/rmse_train_test_20261026/MODEL_REGISTRY.tsv`
- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/rmse_train_test_20261026/parity_manifest.tsv`

Build TODO:

1. Freeze the model family, seed, train/test split, and held-out trajectory
   group in a machine-readable manifest.
2. Audit the source energy field and units before replacing the explicit
   `source unit audit pending` label in panel A.
3. Recompute energy residual, energy RMSE, force RMSE, and frame-level
   distribution panels from the 24 summary rows and their per-frame sources.
4. Add the missing metadata join needed for a defensible cross-temperature or
   cross-task distribution-shift claim; the current package only supports the
   audited held-out 600 K groups.
5. Reviewer checks: no train/test leakage, correct per-atom versus total-energy
   denominator, finite values, one-to-one model/seed rows, and agreement between
   summary and raw recomputation.

Current deliverable: `figures/figure02_near_equilibrium_parity.*`; status
`PROVISIONAL`.

## Figure 3 — within-gap fixed-image evidence

Blueprint intent: paired fixed-image Cr paths, model curves, and a quantitative
error panel.

Current local inputs:

- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/data/Cr_fixed_path_representative_profiles.tsv`
- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/data/cr_three_path_plotted_profiles_R02.tsv`
- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/data/cr_three_path_coverage_audit_R02.tsv`
- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/tables/NORMALIZED_PROFILES.tsv`

Build TODO:

1. Keep the three paths, ten images, and fixed-image coordinate identical for
   QE and every model.
2. Preserve the mandatory G1 text: the QE path is `COMPLETED_BUT_SCF_WARNING`
   and is not a validated migration barrier.
3. Remove the legacy Scratch/endpoint-only proxy from any main-text acceptance
   claim unless a current full fixed-image evaluation replaces it.
4. Define the exact `0.47/0.68/1.20` metric before plotting or citing it; the
   current source does not define those numbers as a publication metric.
5. Recompute maximum absolute error, MAE, and path-wise residuals only after the
   reference status and unit contract are frozen.

Current deliverable: `figures/figure03_within_gap_fixed_images.*`; status
`G1_WARNING / LEGACY_DIAGNOSTIC`.

## Figure 4 — cross-environment transfer

Blueprint intent: compare the same fixed QE images across within-gap,
gap-to-QL/into-QL, and within-QL environments.

Current local inputs:

- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/data/Cr_fixed_path_representative_profiles.tsv`
- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/data/Cr_fixed_path_foundation_profiles.tsv`
- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/tables/NORMALIZED_PROFILES.tsv`

Build TODO:

1. Freeze the environment/path labels and verify the same ten image indices are
   present in every compared series.
2. Define the paired gain as `error_scratch - error_FT` from the same source
   images, not as a difference between separately generated paths.
3. Freeze seed pairing and the sign convention before any green/negative gain
   annotation.
4. Keep QE SCF warning and fixed-image limitations in the caption.
5. Reviewer checks: paired row counts, path identity, finite residuals, and no
   interpretation as a universal transfer ranking.

Current deliverable: `figures/figure04_cross_environment_transfer.*`; status
`DIAGNOSTIC` until the paired metric contract is signed.

## Figure 5 — generated NEB paths

Blueprint intent: model-generated NEB paths and barrier/recognition assessment.

Current local inputs:

- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/data/Cr_four_model_self_relaxed_neb_profiles.tsv`
- the campaign and repair manifests under
  `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024`

Build TODO:

1. Replace endpoint-only/precanonical initialization with full generated paths
   containing ten finite images, coordinates, forces, endpoint identity, and
   convergence logs.
2. Record per-run iteration count, final maximum force, residuals, convergence
   status, and the exact checkpoint hash.
3. Separate valid self-relaxed paths, failed paths, and reconstruction controls;
   do not pool them into a success-rate bar.
4. Only after the gate passes may barrier heights, recognition rates, or model
   comparisons be reported.

Current deliverable: `figures/figure05_generated_neb_diagnostic.*`; status
`BLOCKED`. The red release box is intentional.

## Figure 6 — matched MD

Blueprint intent: stability, structure, and motion over the nominal 1 ns plan.

Current local inputs:

- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/md_1ns_updated_publication_figure_20261025/MD15_FIGURE_DATA_R06G.tsv`
- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/md_1ns_updated_publication_figure_20261025/MD15_CN_GROUP_SUMMARY_R06G.tsv`
- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/md_1ns_updated_publication_figure_20261025/md15_analysis_r03/series`
- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/md_1ns_updated_publication_figure_20261025/md15_analysis_r03/rdf`
- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/md_1ns_updated_publication_figure_20261025/md15_analysis_r03/vacf`
- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/md_1ns_updated_publication_figure_20261025/md15_analysis_r03/msd`

Build TODO:

1. Use the recorded `last_step`/`last_time_ps` as the hard plotting horizon;
   never extend a stopped trajectory to 1 ns.
2. Reconcile the 15 model rows with the raw series and termination records;
   current audit shows only 4/15 reaching 1,000 ps and many failures near
   200–485 ps.
3. Keep failed runs as censored/observed-horizon diagnostics, not as diffusion
   estimates or model ranking.
4. Audit anomalous temperature maxima and invalid frames before any stability
   conclusion.
5. Reviewer checks: valid frames, actual horizon, termination reason, no raw
   trajectory truncation beyond recorded data, and exact CN cutoffs.

Current deliverable: `figures/figure06_matched_md_diagnostics.*`; status
`PROVISIONAL / OBSERVED-HORIZON ONLY`.

## Figure 7 — interlayer sliding

Blueprint intent: fixed-image versus self-relaxed sliding profiles across model
classes.

Current local inputs:

- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/foundation_interlayer_sliding_10model_20261024/results/tables/IMAGE_PROFILES.tsv`
- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/foundation_interlayer_sliding_10model_20261024/results/tables/QE_REFERENCE.tsv`
- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/foundation_interlayer_sliding_10model_20261024/results/tables/RESULT_MATRIX.tsv`
- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/foundation_interlayer_sliding_10model_20261024/results/tables/SLIDING_MATRIX.tsv`

Build TODO:

1. Keep fixed-image and generated/self-relaxed paths in separate panels and
   captions.
2. Label the 20 QE references `COMPLETED_BUT_SCF_WARNING`; the reference is a
   diagnostic, not a validated barrier.
3. Record model identity and checkpoint counts; the current class bands are
   model-identity spread, not training-seed uncertainty.
4. Replace endpoint-only proxy paths with fully converged self-relaxed paths
   before using generated-path conclusions.
5. Include the exact registry displacement coordinate and ten-image count in
   the data contract.

Current deliverable: `figures/figure07_interlayer_sliding_diagnostics.*`; status
`SCF_WARNING / PROXY_DIAGNOSTIC`.

## Figure 8 — QE diagnostic and prospective data need

Blueprint intent: use configuration space to identify where additional QE data
would be valuable.

Current local input:

- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/migrationbench_universe_publication_20261026/derived/structure_universe_points.csv`
- `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/migrationbench_universe_publication_20261026/derived/structure_universe_local_soap_vectors.npz`
- the Figure 3–4 QE error tables only after a validated geometry-keyed join is
  created.

Build TODO:

1. Freeze a geometry-keyed join between descriptor rows, task labels, model
   errors, and prospective held-out paths.
2. Define the high-error threshold before looking at the held-out result.
3. Validate the proposed selection on a held-out path and report recognition
   rate, calibration, and uncertainty.
4. Do not substitute PCA distance for measured error and do not claim an
   acquisition advantage from the current map.

Current deliverable: `figures/figure08_qe_diagnostic_conditional.*`; status
`CONDITIONAL / SUPPLEMENTAL MAP ONLY`, with explicit B–D placeholders.

## Tables 1–2

- `tables/TABLE1_MODEL_REFERENCE_AUDIT.tsv`: rebuild a single model/reference
  audit with exact checkpoint, seed, split, reference source, unit, SCF status,
  and scientific status. Do not merge current Foundation, legacy Scratch,
  endpoint-only proxies, or QE-warning rows into one accepted class.
- `tables/TABLE2_TASK_DECISION_MATRIX.tsv`: rebuild task decisions from the
  figure-level gates, using conservative values `PASS`, `PROVISIONAL`,
  `DIAGNOSTIC`, or `BLOCKED`; every decision must point to a figure and source
  row.

## Recommended execution order

1. Freeze Figure 2 units, split, and seed manifest.
2. Freeze Figure 3–4 path identity, reference status, and paired error metric.
3. Reconcile Figure 5 generated-path convergence and Figure 6 observed-horizon
   records.
4. Re-audit Figure 7 fixed/self separation and checkpoint coverage.
5. Build the Figure 8 geometry-keyed QE join only after Figures 3–4 are frozen.
6. Run the independent reviewer brief, repair all FAIL items, regenerate the
   figures, and only then assign `PUBLICATION_PASS`.
