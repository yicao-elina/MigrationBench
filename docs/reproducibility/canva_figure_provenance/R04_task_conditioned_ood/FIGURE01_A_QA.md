# Figure 1A task-conditioned OOD revision

## Coordinate semantics

The previous R03 Figure 1A plotted the raw `x_pca`/`y_pca` columns, so its axes represented the first two coordinates of the common SOAP PCA projection. It used task hue, source-split marker shape, and nearly fixed marker areas.

The Canva slide-1 map uses the existing `display_x`/`display_y` columns instead. These are produced from the common Cr-local SOAP feature map through the shared PCA and UMAP representation, followed by task-local centering/scaling. Within-task geometry is preserved, but the task-conditioned display is not a new metric distance; the axes are therefore labelled `Feature-map axis 1/2` rather than `SOAP PCA 1/2`.

## R04 encoding

- The source table and all plotted rows are unchanged.
- Task-group hulls are convex envelopes of the measured `task_group` rows; their fill uses the task-map hue.
- Existing `path_key` rows are connected in `frame/sample_frame` order. Solid lines denote fixed/sliding paths; dashed lines denote self-relaxed paths.
- Point area is mapped to the measured `distance_to_training`, with larger points farther from the training feature distribution. The legend uses near-train, intermediate, and far-OOD examples.
- Point hue retains the existing sublevel colors. The task-group hue is reserved for hulls and task labels.
- `rms_delta_A` remains a structural Cr-shell distortion field in the source table; it is not relabelled as OOD in this revision.

## Visual QA

- PNG inspected after rendering for clipping, overlap, hull visibility, path visibility, and legend placement.
- Authored SVG text sizes are 18, 20, or 22 px-equivalent points; no authored text is below the 18 pt target.
- PDF is one-page vector output; SVG parses with `xmllint`.
- This package is an additive review output. It does not overwrite the historical R03 figure package or import anything into Canva.
