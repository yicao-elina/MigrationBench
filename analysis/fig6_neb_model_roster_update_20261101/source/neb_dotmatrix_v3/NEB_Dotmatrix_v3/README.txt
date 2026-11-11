NEB dot-matrix v3

Rebuild:
python source/build_neb_dotmatrix.py --input data/NEB_TOTAL_PERFORMANCE_SEED_EXPANDED_METRICS.tsv --output regenerated

Dependencies: numpy, pandas, matplotlib, reportlab.
Color encodes predicted energy (0-27 eV, global PowerNorm gamma .40); circle area encodes absolute error (square-root area scale plus a minimum visible point). NC is a grey open circle with a cross, without quantitative color or size.
The matrix size legend is drawn by the same size function. All source values and DFT references are retained. See figure_caption.txt, style_spec.json and audit.json.
