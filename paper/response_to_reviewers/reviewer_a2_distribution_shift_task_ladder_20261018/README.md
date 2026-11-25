# Reviewer A2: task-distribution and direct-overlap evidence

This additive package corrects a common ambiguity in the original response: source-level NEB exclusion, direct near-duplicate auditing, and broad descriptor-space novelty are distinct claims.

## Evidence structure

| Evidence | What it establishes | Status |
|---|---|---|
| Grouped FT-600K source split | The declared training data consist of whole AIMD trajectories, not NEB output; the 25 target DFT-path images are evaluation inputs. | `VERIFIED` from the grouped provenance manifest |
| Direct A2 leakage audit | Across the 25 target images, zero exact matches, zero SOAP-near matches, and zero comparable local-RMSD-near matches occur in either grouped training split under the declared metrics. | `VERIFIED` |
| SOAP/PCA task map | Held-out AIMD, fixed-path NEB, and the separate large-cell failure-attribution sample occupy distinct regions of the stored historical descriptor fit. | `DESCRIPTIVE` |
| Failure-stage distance panel | The 639 failure-attribution sample is outside the historical training q99 envelope in every stage, but its kNN distance is not monotonic as failure approaches. | `DESCRIPTIVE` / counterexample to a scalar OOD ladder |
| Self-relaxed NEB in a grouped-train SOAP map | Not generated from the current grouped FT-600K raw train file. | `VALIDATION_GAP` |

The main scientific conclusion is therefore not that a single descriptor radius orders every process. Rather, the evidence supports task-conditioned evaluation: held-out AIMD, fixed-path NEB, and large-cell failure branches are distinct directions in configuration space and must be evaluated with their own measurement operators.

## Build

```text
MIGRATIONBENCH_HISTORICAL_ROOT=/path/to/Wiley-MigrationBench \
MIGRATIONBENCH_PLOT_ATLAS=/path/to/plot_library \
PYTHONPATH=/path/to/plot_library/src \
  python \
  paper/response_to_reviewers/reviewer_a2_distribution_shift_task_ladder_20261018/plot_a2_task_distribution.py
```

Outputs are written in `figures/`, `tables/`, and `distribution_task_ladder_manifest.json`.

`proposed_a2_response_and_insertions.tex` contains the ready-to-paste English reviewer response, SI table and captions, and bounded main-text insertions. `analysis_and_placement.md` records why the current task map is SI-ready but not yet the definitive grouped-FT-600K main-text map.

## Figure use

- `fig_a2_task_distribution_map.*`: candidate main-text figure or main-text panel. Its axes are a two-dimensional visualization of the retained 20-component SOAP-PCA diagnostic; distances in this projection are not used as a calibrated OOD score.
- `fig_a2_distance_audits.*`: SI/reviewer-response figure. Panel a responds directly to leakage; panel b reports the 639-stage diagnostic without claiming a monotonic causal mechanism.

No model-family colours are used in these figures because they encode configuration/task classes rather than models. JHU MACE colours remain reserved for model-family comparisons.

The historical descriptor inputs are intentionally not copied into this repository: they belong to
the separate historical 639-screen audit and are referenced by SHA256 in the manifest. The grouped
overlap evidence and the grouped retraining figure are fully self-contained in the A2 package; the
task-distribution figures are therefore marked descriptive rather than a release-level OOD score.
