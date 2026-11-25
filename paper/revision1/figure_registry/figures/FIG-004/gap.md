# FIG-004 provenance gaps

- The new all-data from-scratch seeds 123, 234, and 345 all fail the smoke scientific threshold near 202 ps; short pre-failure MSD fits are retained, but this campaign contributes no accepted 1 ns Dxy row.
- The locally archived all-data segments begin at 200.0 ps and end at the first threshold near 202--203.8 ps, so their Dxy values are terminal short-lag estimates rather than full pre-failure trajectory transport coefficients.
- Historical Foundation-OMAT and legacy from-scratch raw position trajectories are not locally archived; their processed pre-threshold MSD-derived values are retained where available, while exact raw time-origin recomputation remains a provenance gap.
- Historical parent and additive FT rows retain source data and corrected metrics, but exact model SHA256 values are not present in the local historical manifests used here. They are labelled provenance-partial rather than silently inferred.
- Pressure is plotted only for rows with real stress-derived pressure files. The historical seed-234/345 FT pressure fields are not locally archived, so no pressure uncertainty band is fabricated for those rows.
- HFACF and thermal conductivity remain unavailable because no matching heat-flux time series was recorded.
