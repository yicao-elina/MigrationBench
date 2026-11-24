# 639 candidates｜12-model static MLIP evidence package

> [!summary] Verified status
> The same 639 periodic 2050-atom structures were evaluated with 12 canonical MLIP checkpoints. The paired matrix contains 7,668 finite successful rows with zero duplicate candidate–model pairs.

## One-line conclusion

The candidate set is a multi-trajectory, stage-balanced failure-attribution probe—not a time-weighted MD population—and it reveals increasing model-force divergence from stable controls to precursors and threshold frames. All candidates lie outside the training SOAP q99 envelope and are separated from the NEB reference region in the stored PCA diagnostic.

## Navigation

- [[sampling-and-representativeness]] — where the 639 structures came from and what they represent
- [[model-provenance]] — canonical 12-model roster, seeds, hashes and labels
- [[distribution-shift]] — SOAP/PCA comparison with training, validation, test and NEB references
- [[force-divergence]] — stage-wise 12-model static-force comparison
- [[figures-and-tables]] — figure, tables and publication interpretation
- [[reproduce]] — local rebuild and Rockfish provenance
- [[gaps-and-boundaries]] — verified, reconstruction, pending and excluded claims

## Evidence anchors

- Combined manifest: `../analysis/combined_analysis_manifest.json`
- Candidate manifest: `../candidate_manifest_639.csv`
- Model registry: `../MODEL_REGISTRY.tsv`
- Model hash audit: `../MODEL_HASH_AUDIT.tsv`
- Claim matrix: `../CLAIM_EVIDENCE_MATRIX.tsv`
- Release manifest: `../PAPER_RELEASE_MANIFEST.json`
- Results and discussion: `../RESULTS_AND_DISCUSSION.md`

## Figure

![[../figures/fig_multimodel_639_overview.png]]

The figure is a new Plot Atlas reconstruction from verified processed data; it is not a pixel-parity reconstruction of a previously published figure.
