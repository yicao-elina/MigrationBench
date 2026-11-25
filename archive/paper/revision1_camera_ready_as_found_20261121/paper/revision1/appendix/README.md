# Endpoint perturbation appendix

This directory is self-contained with respect to the endpoint-level data,
summary table, plotting source, and checked-in figure assets used by the
Supplementary Information.

The endpoint overview (`figures/fig1_campaign_overview.{pdf,svg,png}`) was
redrawn from `data/endpoint_level_summary.csv` by
`scripts/render_fig1_campaign_overview_palette_20261109.py`. It uses the
2026-11-09 MigrationBench palette and publication typography. It retains the
endpoint-level final-force, displacement, FIRE-step, and force--geometry views.
The original all-panel renderer is retained as
`scripts/plot_campaign_atlas.py`. Figures 2–3 and the summary table retain
their existing source and data.
The companion summary table is `table/multiarms_table.tex`, sourced from
`data/multiarms_summary.csv`.

The standard Overleaf compiler uses the checked-in PDF figures. To regenerate the endpoint overview, run the new renderer with Python, pandas,
NumPy, and Matplotlib. To regenerate the older atlas figures, use the Plot Atlas
package as described in `scripts/plot_campaign_atlas.py`, then compile
`revision1/sn-article-SI.tex`.
