"""Build all six MigrationBench schematics into out/ (SVG, PDF, 600 dpi PNG).

Run from this folder:  ../../.venv-fig/bin/python build_all.py
"""

from figs import fig1_method, fig2_static, fig3_md, fig4_local, fig5_sliding, fig6_attribution

if __name__ == "__main__":
    for module in (fig1_method, fig2_static, fig3_md, fig4_local, fig5_sliding, fig6_attribution):
        module.build()
