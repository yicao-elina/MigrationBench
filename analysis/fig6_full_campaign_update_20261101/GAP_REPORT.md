# Figure 6 update gap report

## Resolved in this update

- The analysis now uses the complete historical 12-model static screen rather than a partial current-model subset.
- The heatmap uses explicit family strips and readable log-color cell labels.
- The stage-distribution panel preserves the three-arm encoding and renders the one stable-terminal observation as a point.
- The embedding panel retains the NEB reference cluster and adds a near-origin inset for the candidate/reference cloud.
- Independent PNG/PDF/SVG exports are available for every panel and for the composite.

## Remaining boundaries

- This source package is a static MACE energy/force evaluation, not an MD transport campaign and not a DFT validation.
- FT-600K and FT-MultiT are same-family seed replicas. Foundation variants are not seeds. Scratch includes two Lmax2 branches and a different baseline branch.
- The legacy compiled OMAT checkpoint is retained for historical comparison but is excluded from current publication scope in the SQLite registry.
- The figure is therefore marked `RECONSTRUCTION`, not a bit-identical legacy reproduction and not a current publication release.
- The recovered Plot Atlas package was not hydrated in the local filesystem. The script applies the documented Atlas style contract locally and records this fallback in the manifest.
