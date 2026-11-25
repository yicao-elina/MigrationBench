# Migration-path diagnostic framework: acceptance matrix

This matrix records what is currently demonstrated by the retained 10-path
endpoint-only dataset and what is still required before the results can be
described as a formal DFT/Foundation-model validation.

## Scope and fixed evidence

- Dataset: 10 retained QE path-table traces, 10 images per path.
- Foundation evaluation: 9 hashed local Foundation models, 900/900 successful
  same-geometry single-point evaluations.
- Path classes: 3 `in_gap_diffusion`, 3 `gap_to_QL`, 3 `within_QL`, and one
  `unclassified_control` (bare Cu control).
- Energy convention: QE `sb2te3.path` Hartree values converted to eV with
  `27.211386245988`; path coordinates converted from Bohr to Angstrom with
  `0.529177210903`; model energies normalized within each path.
- Reference boundary: this is a screened QE path-table reference, not a
  converged DFT barrier benchmark.

## Requirement-by-requirement status

| Layer | Acceptance question | Current status | Evidence | Missing or limiting evidence |
|---|---|---|---|---|
| DFT-NEB reliability | Are all retained images finite and structurally traceable? | **Conditional pass** | `all_path_diagnostic_framework/dft_reference_qualification.csv`; `all_path_structural_diagnostics/qe_path_structure_stage_metrics.csv` | This proves finite path-table records and path identity, not force convergence. |
| DFT-NEB reliability | Are SCF and NEB convergence formally established? | **Fail for formal reference; exploratory only** | `scf_neb_iteration_energy_audit.csv`; `audit_summary.json` | All 10 logs contain at least one SCF warning; inner electronic SCF iteration counts are not recorded; 8/10 pass the last-table `<0.1 eV` screen, 2/10 do not; no force-convergence record is available in this audit. |
| Energy comparison | Are QE and Foundation values evaluated on identical frames and units? | **Pass for same-geometry diagnostic** | `all_dft_foundation_models/corrected_dft_foundation_manifest.json`; `corrected_dft_foundation_image_profiles.csv` | The result is valid for profile comparison, but not a claim that the QE trace is a converged barrier. |
| Energy comparison | Are alignment sensitivity and per-image/barrier errors reported? | **Pass** | `all_path_diagnostic_framework/alignment_profile_errors.csv`; `model_path_metrics_by_alignment.csv`; `corrected_dft_foundation_path_barriers.csv` | Add fresh reconverged DFT/AEB references before using absolute barrier error as a final accuracy metric. |
| Failure mechanism | Are errors resolved by migration stage and path class? | **Pass for current NEB-only scope** | `model_stage_metrics.csv`; `path_type_stage_error_summary.csv`; `figure_path_type_summary.{png,pdf,svg}` | Stage labels are derived from the screened QE path maximum; they are not independent saddle-point validation. |
| Failure mechanism | Are physically interpretable geometry descriptors linked to error? | **Conditional pass** | `model_frame_errors_with_structure.csv`; `geometry_only_failure_predictor_metrics.csv`; `figure_structural_stage_diagnostics.{png,pdf,svg}` | Current descriptors are geometry/image-index features; local chemistry and provenance-qualified training-distribution distances are not yet included. |
| Representation space | Are training, test, AIMD and NEB configurations compared in a common representation space? | **Not yet demonstrated** | `diagnostic_framework_manifest.json` explicitly records `embedding_scope: NEB frames only` | Need immutable train/test/AIMD manifests, model/version hashes, feature extraction settings, and a leakage-safe distance calculation. |
| Usage boundary | Can high-error path images be identified before deployment? | **Provisional only** | `geometry_only_failure_predictor_oof.csv`; `provisional_usage_policy_oof.csv`; `provisional_usage_policy_summary.csv` | Grouped 4-fold CV is NEB-only; external trajectory validation and calibration are still required. Do not call the bands certified. |

## Current scientific conclusion

The current evidence supports a diagnostic statement: Foundation-model error is
path- and stage-dependent, and the largest disagreement is concentrated in
specific high-distortion/transition-state environments rather than being a
uniform offset. It does **not** support a statement that the screened QE
path-table values are converged DFT barriers or that the provisional early-warning
policy is deployment-ready.

## Required completion gates

1. Finish the separate historical Cr-Sb2Te3 audit, preserving the distinct
   `2D_421_Diffusion_traj` and `neuromorphic-filament/dft/04_neb` branches.
2. Reconcile every historical figure/table against its raw QE output, model
   input frame, unit convention, and barrier definition.
3. Recover or rerun QE evidence sufficient to record per-image SCF iterations,
   final SCF residual/force information, and path-level convergence for any
   result promoted to a reference answer.
4. Add provenance-qualified training, test, and AIMD configuration manifests to
   the embedding analysis; keep NEB frames as a held-out evaluation set.
5. Refit and calibrate the direct-use/validation/DFT-fallback rule on grouped
   splits that include independent non-NEB configurations, then regenerate all
   publication figures and tables from the final manifest.

## Historical grouped Foundation benchmark checkpoint

The independent audit of
`/Users/alina/Documents/Codex/foundation_grouped_benchmark_20261001` has now
been added to the evidence chain:

- 61-atom branch: 5 QE paths and 50/50 finite Foundation profiles. All five
  retained QE outputs reproduce their historical barrier arithmetic, but none
  retains `JOB DONE` or an explicit NEB/BFGS convergence marker. Formal DFT
  reference candidates: 0/5.
- 81-atom branch: 5 QE paths and 50/50 finite Foundation profiles. All five
  retain `JOB DONE`, but all five contain `SCF convergence NOT achieved`
  warnings. Formal DFT reference candidates: 0/5.
- Therefore the old 61-atom `dft_quality=pass` label is downgraded to
  `exploratory/conditional`; the 81-atom `SCF convergence warning` label is
  retained as a stress-test qualifier, not a convergence certificate.

Evidence is preserved in
`historical_grouped_benchmark_audit_20261003/`, including raw-output hashes,
re-extracted path tables, Foundation JSON hashes, and profile identity checks.
