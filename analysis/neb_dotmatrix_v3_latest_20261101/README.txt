NEB dot-matrix v3

Rebuild:
python source/build_neb_dotmatrix.py --input data/NEB_TOTAL_PERFORMANCE_SEED_EXPANDED_METRICS.tsv --output regenerated

Dependencies: numpy, pandas, matplotlib, reportlab.
Color encodes predicted energy (0-27 eV, global PowerNorm gamma .40); circle area encodes absolute error (square-root area scale plus a minimum visible point). All 120 supplied finite values are drawn as product metrics. Original source statuses remain in plotted_values.tsv and audit.json for provenance, but no diagnostic-only glyph is placed in the product figure.
The matrix size legend is drawn by the same size function. All source values and DFT references are retained. See figure_caption.txt, style_spec.json and audit.json.
