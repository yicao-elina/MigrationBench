# Manuscript comment and narrative audit — 2026-11-07

## Status

Eight unresolved Overleaf comment threads were visible in the project review panel. The corresponding changes below are applied to the local main manuscript and Supplementary Information (SI). Overleaf has **not** been updated and the threads have **not** been resolved online: the browser session became unavailable while attempting to download the current source, so the server-side source could not be re-read and reconciled before upload.

## Comment-by-comment changes in the local sources

| # | Underlying issue | Local change |
|---|---|---|
| 1 | Figure citations should identify the evidence panel, not only the figure. | Updated prose citations to include the relevant panels throughout the main text and SI; aligned figure captions with the displayed panel lettering. |
| 2 | Avoid defending the story against an earlier non-archival report. | Removed the prior-report discussion from the Introduction; the study question is now framed directly around the tests in this manuscript. Necessary data-lineage notes remain in the SI where they affect provenance or convergence status. |
| 3 | The 600 K and MultiT pools are not a four-condition temperature-only experiment. | Methods now explain that 600 K samples near the MD target and MultiT tests broader thermal coverage. It states that size and test composition also change, so the comparison cannot isolate a temperature effect or establish a general extrapolation advantage. |
| 4 | Protocol/design detail was interrupting the Results. | Moved model-lineage, split-size, pointwise-centering, diagnostic and run-protocol detail into Methods. Results now leads with the observed outcome and its interpretation. |
| 5 | Results should report findings rather than restate how each evaluation was run. | Reorganized the migration, MD, projection and SHAP sections around what the measurements show; moved operator definitions and diagnostic settings to Methods. |
| 6 | Scratch and fine-tuning need an explicit within-regime comparison, with the design limits stated. | Compared scratch and fine-tuned routes within each data regime, stated that routes share the target split within a regime, and qualified differences as comparisons of practical implementations because architecture and optimization are not fully matched. Added a polished SI training-design figure and cited it from Methods. |
| 7 | Main-to-SI and SI-to-main cross-references showed `??`. | Retained the reciprocal external-document labels and checked both compile directions by compiling SI and main in sequence. The final local PDFs have no undefined references or `??` text. |
| 8 | The MD horizon should be stated as a 1 ns target without confusing archive language. | Results reports the 1 ns target and observed endpoints directly. The SI method now calls the initial configuration a saved state and describes recorded momenta; stopped runs are explicitly distinguished from runs reaching the target. |

## Cross-manuscript scan

The narrative now runs from pointwise energy/force errors to model-generated MD, fixed versus model-relaxed migration paths, collective sliding, then descriptive projections and error-surrogate associations. Claims are bounded where the evidence does not support stronger conclusions: the migration references retain convergence warnings; a low self-relaxed peak is not treated as a validated barrier; trajectory survival is not treated as physical accuracy; projection separation is not treated as mechanism; SHAP features are not treated as causal channels; and the tested routes are not presented as a causal, architecture-matched comparison.

## Local verification

- Compiled SI and main repeatedly with the project class, bibliography and external-document auxiliaries in place.
- Final compile logs contain no undefined references or citations. The remaining warnings are the template's geometry over-specification notices.
- Extracted text from both final PDFs contains zero `??` pairs.
- Visually inspected the new training-design figure after adjusting its legend so 600 K, MultiT, scratch and fine-tuned are all distinct.

## Files in the sync payload

The ZIP contains updated main and SI `.tex` files, the new SI training-design PDF, and its editable TikZ source, arranged under `revision1/`. It is a local handoff payload; it has not been uploaded to Overleaf.
