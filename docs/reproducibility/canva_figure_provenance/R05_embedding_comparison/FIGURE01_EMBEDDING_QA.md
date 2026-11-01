# Figure 1 embedding comparison

This audit compares the source-provided common SOAP/PCA, UMAP, and t-SNE coordinates from `comprehensive_task_map_points.csv`. No task-specific anchors, translations, rank equalization, or post-hoc point movement are used.

## Selection result

The balanced task-label diagnostics are:

| embedding | centroid accuracy | approximate balanced silhouette |
|---|---:|---:|
| raw SOAP PCA | 0.377 | -0.078 |
| existing UMAP | 0.268 | -0.174 |
| existing t-SNE | 0.580 | -0.016 |

The existing t-SNE is the relatively best task-separation view among these saved embeddings, but its axes and global distances are not metric quantities. It is therefore retained as an optional visual embedding, not substituted for the raw PCA axes in Figure 1A.

The main Figure 1A revision keeps `x_pca/y_pca` unchanged and adds only measured OOD marker area (`distance_to_training`), very light task hulls, and source-provided path lines. If the t-SNE view is used later, it must be labelled `t-SNE`, with its original fit metadata and the same caveat.

The attempted fresh UMAP/densMAP regeneration was not accepted because the provider-backed SOAP-vector import did not materialize within the bounded check. The comparison here uses only coordinates already present in the audited CSV.
