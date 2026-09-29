# Endpoint perturbation appendix

This directory is self-contained with respect to the endpoint-level data,
summary table, plotting source, and checked-in figure assets used by the
Supplementary Information.

The figures were generated from `data/endpoint_level_summary.csv` and
`data/endpoint_level_summary.json` with `scripts/plot_campaign_atlas.py`.
The companion summary table is `table/multiarms_table.tex`, sourced from
`data/multiarms_summary.csv`.

The standard Overleaf compiler uses the checked-in PDF figures. To regenerate
the assets locally, run the plotting script in the project environment with the
Plot Atlas package available, then compile `revision1/sn-article-SI.tex`.
