# Figure 1A clean raw-PCA revision

The first Figure 1A foreground point contract is restored:

- `train`, `valid`, `test`, and `md_sample` points remain visible;
- only the five direct `dft_neb` rows and ten direct `dft_interlayer` rows are shown as NEB/sliding foreground markers;
- the 245 `mlip_self_neb` rows and 30 `mlip_interlayer_self` rows remain fully represented in panel C and the source table, but are not rendered as hundreds of foreground markers;
- the primary coordinates are the source-provided `x_pca/y_pca` values with no task translation;
- task hulls are very light visual guides only;
- the complete source path-line set is omitted from the primary PCA panel because it forms a spiderweb at this density; path-aware rows remain in the source table and the pre-cleanup snapshot;
- marker area uses `distance_to_training` for OOD and does not relabel `rms_delta_A` as OOD.
- the legend uses the four task-family colors together with their task-specific marker shapes; OOD marker area remains a separate encoding, while source-split coverage is kept in panel C.
- the compact gray DFT key retains `DFT - train`, `DFT - valid`, and `DFT - test` as circle, triangle, and square handles.

Superseded versions are preserved in the pre-cleanup snapshot commit recorded by `archive/OLD_UNIVERSE_VERSIONS_SNAPSHOT.md`; they are removed from the active tree. This R06 package is additive and has not been imported into Canva.
