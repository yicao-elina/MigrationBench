# Seed-resolved clustered boxplot — gallery case

This gallery case freezes the preferred visual grammar from Figure 2 v16,
commit `941baf8`. It is the project reference for clustered boxplots that
show three independent seeds per model family while retaining every finite
frame-level point.

## Reproduce

From the repository root:

```bash
python docs/reproducibility/canva_figure_provenance/gallery/boxplot_cluster_reference_v04/reproduce.py
```

The default output directory is `outputs/` beside this file. A different
directory can be supplied with `--out-dir`.

The wrapper calls the canonical Figure 2 builder rather than copying or
reimplementing its plotting code. It writes a manifest with SHA256 hashes for
the scripts, packaged inputs, and generated outputs so the exact provenance of
the gallery render remains inspectable.

## Reference outputs

The generated directory contains the standalone temperature-resolved figure,
the square overview, and the square composite, each in PDF, SVG, and PNG
forms. PDF and SVG are the editable vector sources for downstream layout work;
PNG is retained only for visual QA.

The canonical source scripts remain in
`docs/reproducibility/canva_figure_provenance/scripts/` and are listed in the
generated manifest. The packaged frame-level inputs remain under
`docs/reproducibility/canva_figure_provenance/data/figure02/`.
