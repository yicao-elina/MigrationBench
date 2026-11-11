# Feature-map candidate selection

## Decision

The publication-facing map is the **feature-guided spaced display**:

- common Cr-local SOAP descriptors;
- the existing shared PCA fit, followed by UMAP on the first 30 common PCA coordinates;
- marker area = measured Cr-shell RMS bond-length distortion;
- color = the four physical task regimes, with saturation/lightness for DFT/MLIP sublayers;
- solid/dashed path lines = fixed/sliding versus self-relaxed protocols;
- the four task groups are spaced for legibility, while the relative UMAP geometry inside each task group is retained.

The task-conditioned spacing is a communication layer, not a new physical distance. Raw feature-space candidates are retained for quantitative proximity claims.

## Candidate comparison

| Candidate | Physical input | Visual result | Decision |
|---|---|---|---|
| SOAP/PCA → UMAP | common SOAP/PCA | Most informative nonlinear feature map; task groups still overlap and the reference manifold is sparse | retain as raw audit |
| SOAP/PCA → densMAP | common SOAP/PCA | Better density fidelity but too compact for the comprehensive universe figure | reject for main |
| SOAP/PCA + measured Cr-shell geometry → densMAP | SOAP plus RMS distortion, coordination/species deltas, bond-length spread | physically interpretable but over-compresses the task cloud | reject for main |
| SOAP/PCA + measured Cr-shell geometry → UMAP | SOAP plus the same measured geometry | meaningful distortion sensitivity, but less clean and less separated visually | retain as diagnostic |
| SOAP/PCA direct 2D | first two common PCA axes | linear and transparent, but collapses the useful manifold into long bands/outliers | reject for main |
| local UMAP + rank equalization | local SOAP UMAP with monotonic display calibration | visually expanded, but marginal equalization weakens metric interpretation | reject for main |

## Energy handling

Energy is not used as a shared embedding coordinate in this release. The reference structures, MLIP-MD frames, DFT fixed paths, self-relaxed paths, and sliding paths do not provide a single comparable energy zero and model-consistent coverage. Energy can be added later as an independent color/contour overlay within a path family after the energy provenance is frozen; it should not define cross-family map distance here.

## Outputs

- Main map: `figures/migrationbench_configuration_universe_comprehensive.png`, `.pdf`, `.svg`
- Common feature audit candidates: `figures/feature_map_candidates/`
- Candidate parameters: `figures/feature_map_candidates/candidate_manifest.json`
- Point-level provenance and coordinates: `derived/comprehensive_task_map_points.csv`
- Fit manifest: `derived/comprehensive_task_map_fit.json`
