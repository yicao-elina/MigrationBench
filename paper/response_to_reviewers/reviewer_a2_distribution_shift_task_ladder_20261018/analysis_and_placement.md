# A2 distribution analysis and recommended placement

## What the data now establish

The reviewer’s two alternatives address two different risks. The first is literal leakage: a target NEB image (or an almost identical geometry) may appear in the fine-tuning file. The second is ordinary task proximity: the target can be absent from training but still have a close homologue. The evidence answers both at their proper level.

1. **Source provenance — verified.** The grouped FT-600K training file is assembled from 600 K AIMD trajectories; no NEB image or NEB iteration history is a training source. The 25 retained DFT images are evaluation inputs.
2. **Direct near-duplicate audit — verified.** For full grouped FT-600K (1,786 frames) and Scratch-5% (89 frames), the 25 fixed DFT-path images have no exact global/local fingerprint match, no SOAP distance at or below `1e-4`, and no species-compatible Cr-local RMSD at or below `0.05 A`. The closest SOAP distances are `5.43e-4` and `6.37e-4`; the closest comparable RMSDs are `2.055` and `2.057 A`.
3. **Historical descriptor map — descriptive.** A 521-structure shared SOAP fit maps 250 historical AIMD train, 246 historical held-out AIMD, and 25 historical fixed-path NEB structures. The NEB structures are separated from the AIMD reference region in the retained PCA view. This is a visualization of a historical reference fit, not a recalculation on the current grouped A2 input files.
4. **Large-cell failure attribution — descriptive counterexample.** All 639 stage-balanced failure-attribution configurations lie outside that historical training q99 kNN envelope, but their median kNN distance moves from 12.69 (early stable) to 12.54 (pre-onset), 12.21 (precursor), and 11.17 (threshold). It is therefore wrong to describe this dataset as a monotonic ``move outward until explosion'' trajectory. The valid result is that this MD branch starts in a different descriptor region and that task difficulty is not a scalar descriptor radius.

## Recommended figure roles

| Asset | Placement now | What it supports | Boundary |
|---|---|---|---|
| `fig_a2_distance_audits.pdf` | Reviewer response and SI, new ``Data splitting and configuration-space audit'' subsection | Direct response to A2; no detected target-image leakage under declared metrics | The dashed lines are audit thresholds, not physical accuracy thresholds. |
| `fig_a2_task_distribution_map.pdf` | SI now; cite once from the Evaluation Framework | Distinct directions in the available common historical descriptor fit | Not current grouped FT-600K; not calibrated uncertainty; not a ranking. |
| Current fixed-path/self-relaxed NEB matrix | Results discussion, kept protocol-separated | Fixed-path versus model-in-the-loop NEB are different operators | DFT anchors remain provisional; 18 self-relaxed runs converged, 17 are finite but non-converged, and 5 legacy rows are diagnostic. |

## Main-text plan

Put only the source-level NEB exclusion plus the direct overlap conclusion in Methods Sec. 2.1. Add one sentence in the Evaluation Framework that configuration shift is multidirectional and point readers to the SI map. Do **not** call the current historical map the final grouped FT-600K distribution map in the main text.

The upgrade that makes a single complete main-text task map defensible is post-processing rather than a new physical campaign: make the archived `data_grouped/train.xyz`, `valid.xyz`, and `test.xyz` files available locally, transform those plus the retained fixed DFT paths, selected equilibrium MD frames, and final self-relaxed NEB images using one frozen SOAP/PCA/kNN fit, and write their hashes into a manifest. Then replace the SI descriptor map with that current common-space map. The final self-relaxed images must remain labelled as model-relaxed geometries rather than DFT path images.
